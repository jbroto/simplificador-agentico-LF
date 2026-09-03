# Agentes/supervisor_agente.py

import logging
import os
import uuid
import httpx
import uvicorn

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from Agentes.agente import BaseA2AAgent
from utils.a2a import (
    AgentCard,
    AgentSkill,
    Message,
    TaskResult,
    TaskSendParams,
    TextPart,
)
from utils.registro_agentes import AGENTES_SUPERVISOR
from utils.logger_config import setup_logging

load_dotenv()

PORT = 8003
TIMEOUT = 300

logger = logging.getLogger(__name__)
setup_logging("supervisor")
logger.info("Iniciando Supervisor A2A en puerto %d.",PORT)


async def descubrir_agentes() -> dict:
    agentes = {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        for nombre, url in AGENTES_SUPERVISOR.items():
            response = await client.get(
                f"{url}/.well-known/agent.json"
            )
            response.raise_for_status()
            card = response.json()
            agentes[nombre] = {
                "url": url,
                "card": card,
            }
    return agentes


async def llamar_agente(url: str, texto: str, informe: str | None = None) -> str:

    message = Message(
        role="user",
        parts=[
            TextPart(text=texto)],
        informe=informe)

    params = TaskSendParams(
        id=str(uuid.uuid4()),
        message=message)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        response = await client.post(f"{url}/tasks/send", json=params.model_dump())
        response.raise_for_status()
        result = TaskResult(**response.json())

    if result.status == "completed":
        return result.output or ""
    
    else:
        raise RuntimeError(f"El agente {url} falló: {result.error}")


class SupervisorAgent(BaseA2AAgent):

    card = AgentCard(
        name="Supervisor",
        description=(
            "Orquestador principal del proceso de adaptación a Lectura Fácil. "
            "Coordina los agentes de simplificación y crítica, decide cuándo "
            "continuar o finalizar el proceso de simplificación y solicita "
            "el glosario final cuando el texto está preparado."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="simplify-and-validate",
                name="Orquestación de Lectura Fácil",
                description=(
                    "Coordina el proceso completo de adaptación de un texto "
                    "a Lectura Fácil. Solicita la simplificación, evalúa el "
                    "resultado mediante el agente crítico y decide si es "
                    "necesario realizar nuevas iteraciones."
                ),
                input_description=(
                    "Texto que se desea adaptar a Lectura Fácil."
                ),
                output_description=(
                    "Texto final adaptado a Lectura Fácil, incluyendo el "
                    "glosario cuando sea necesario."
                ),
            )
        ],
    )

    llm = ChatGroq(
        model=os.environ.get("MODEL"),
        api_key=os.environ.get("GROQ"),
    )

    async def process(self, message: Message) -> str:

        texto = message.parts[0].text
        logger.info("Solicitud recibida: %d caracteres.",len(texto))
        
        # 1. DESCUBRIR AGENTES
        agentes = await descubrir_agentes()

        logger.info("Agentes descubiertos: %s",list(agentes.keys()))

        # 2. IDENTIFICAR LOS ORQUESTADORES
        agente_simplificador = None
        agente_critico = None
        agente_definidor = None

        for nombre, datos in agentes.items():

            card = datos["card"]
            nombre_card = card["name"].lower()

            if ("simplificador" in nombre_card):
                agente_simplificador = datos

            elif ("crítico" in nombre_card or "critico" in nombre_card):
                agente_critico = datos

            elif "definidor" in nombre_card:
                agente_definidor = datos

        if not agente_simplificador:
            raise RuntimeError("No se ha encontrado el orquestador de simplificación.")

        if not agente_critico:
            raise RuntimeError("No se ha encontrado el orquestador crítico.")

        # 3. CICLO DE SIMPLIFICACIÓN
        limite_iteraciones = 5

        for iteracion in range(1, limite_iteraciones + 1):

            logger.info("Iniciando iteración %d/%d.",iteracion,limite_iteraciones)

            # CRÍTICA

            logger.info("Enviando texto al orquestador crítico.")

            informe_critico = await llamar_agente(agente_critico["url"],texto)

            MARCADOR = "=== CONCLUSIÓN GLOBAL ==="
            if MARCADOR in informe_critico:
                conclusion_global = informe_critico.split(MARCADOR, 1)[1].strip()
            else:
                conclusion_global = informe_critico.strip()

            logger.info("Informe crítico recibido: %d caracteres.",len(informe_critico))

            # DECISIÓN DEL SUPERVISOR
            decision_prompt = f"""
            Eres el supervisor principal de un proceso de adaptación a Lectura Fácil según la norma UNE 153101:2018.

            Debes decidir si el proceso debe continuar o puede finalizar.
            El agente crítico ha evaluado el texto y ha generado el siguiente informe:

            INFORME DEL CRÍTICO:
            {conclusion_global}

            REGLAS DE DECISIÓN:

            - Si el texto es globalmente adecuado y los problemas detectados
              son inexistentes, leves o no justifican otra modificación,
              puedes aprobar el texto.

            - Si existen problemas relevantes que dificultan la comprensión,
              debe realizarse otra iteración de simplificación.

            - No seas excesivamente estricto.
              El objetivo no es conseguir un texto perfecto, sino un texto
              suficientemente claro y adecuado para Lectura Fácil.

            Responde ÚNICAMENTE con una de estas dos opciones:

            CONTINUAR
            FINALIZAR
            """

            decision_response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content=decision_prompt
                    ),
                    HumanMessage(
                        content="Decide el siguiente paso."
                    ),
                ]
            )

            decision = decision_response.content.strip().upper()

            logger.info("Decisión del Supervisor: %s",decision)

            # FINALIZAR

            if "FINALIZAR" in decision:
                logger.info("El Supervisor considera que el texto está preparado.")
                break

            # CONTINUAR
            if iteracion == limite_iteraciones:
                logger.info("Se ha alcanzado el límite máximo de iteraciones.")
                break

            logger.info("Se requiere una nueva simplificación.")

            texto = await llamar_agente(
                agente_simplificador["url"],
                texto,
                informe=informe_critico,
            )

            logger.info("Nueva simplificación completada: %d caracteres.",len(texto))

        # 5. DEFINIDOR
        if agente_definidor:
            logger.info("Enviando texto final al agente Definidor.")

            definiciones = await llamar_agente(agente_definidor["url"],texto)

            logger.info("Glosario generado.")

            return (
                f"{texto}\n\n"
                f"--- GLOSARIO ---\n"
                f"{definiciones}"
            )

        logger.info("No se encontró agente Definidor. "
                    "Se devuelve únicamente el texto.")

        return texto


agent = SupervisorAgent()
app = agent.build_app()


if __name__ == "__main__":


    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )