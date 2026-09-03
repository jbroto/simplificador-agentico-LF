# Agentes/critico_agent.py

import logging
import os
import uuid
import httpx
import uvicorn
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from utils.logger_config import setup_logging
from utils.registro_agentes import AGENTES_CRITICO

from Agentes.agente import BaseA2AAgent
from utils.a2a import (
    AgentCard,
    AgentSkill,
    Message,
    TaskSendParams,
    TaskResult,
    TextPart,
    CriticReport,
)

load_dotenv()

PORT = 8002
TIMEOUT = 300

logger = logging.getLogger(__name__)
setup_logging("critico")
logger.info("Iniciando agente A2A en puerto %d.",PORT)


async def descubrir_agentes_criticos() -> list:
    """
    Descubre dinámicamente los agentes críticos mediante sus Agent Cards.
    """

    agentes = []

    async with httpx.AsyncClient(timeout=10.0) as client:

        for nombre, url in AGENTES_CRITICO.items():

            try:
                response = await client.get(
                    f"{url}/.well-known/agent.json"
                )

                response.raise_for_status()

                card = AgentCard(**response.json())

                # Solo nos interesan agentes especializados en crítica.
                if "critico" not in card.name.lower():
                    continue

                agentes.append(
                    {
                        "name": card.name,
                        "url": card.url,
                        "description": card.description,
                        "skills": card.skills,
                    }
                )

                logger.info(
                    "Agente crítico descubierto: %s",
                    card.name,
                )

            except Exception:
                logger.exception(
                    "No se pudo descubrir el agente en %s",
                    url,
                )

    if not agentes:
        raise RuntimeError(
            "No se ha encontrado ningún agente crítico."
        )

    return agentes


async def llamar_agente(
    url: str,
    texto: str,
) -> CriticReport:
    """
    Envía una tarea A2A a un agente crítico y convierte su respuesta
    en un CriticReport estructurado.
    """

    params = TaskSendParams(
        id=str(uuid.uuid4()),
        message=Message(
            role="user",
            parts=[
                TextPart(text=texto)
            ],
        ),
    )

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        response = await client.post(
            f"{url}/tasks/send",
            json=params.model_dump(),
        )

        response.raise_for_status()

        result = TaskResult(**response.json())

    if result.status != "completed":
        raise RuntimeError(
            f"El agente crítico en {url} falló: {result.error}"
        )

    if not result.output:
        raise RuntimeError(
            f"El agente crítico en {url} no devolvió ningún informe."
        )

    try:
        return CriticReport.model_validate_json(result.output)

    except Exception as e:
        raise RuntimeError(
            f"El agente crítico en {url} devolvió un informe "
            f"con formato incorrecto: {e}"
        )


class CriticoAgent(BaseA2AAgent):

    card = AgentCard(
        name="Critico",
        description=(
            "Agente coordinador especializado en verificar textos conforme "
            "a la norma UNE 153101:2018 de Lectura Fácil. Coordina agentes "
            "críticos especializados en vocabulario, oraciones y estructura, "
            "analiza conjuntamente sus informes y determina si el texto "
            "necesita nuevas simplificaciones o puede considerarse adecuado."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="critique-text",
                name="Evaluación integral de Lectura Fácil",
                description=(
                    "Coordina la evaluación léxica, sintáctica y estructural "
                    "de un texto. Recoge los informes de los agentes críticos "
                    "especializados, analiza la gravedad y relevancia de los "
                    "problemas detectados y genera una conclusión global "
                    "sobre si es necesario continuar simplificando."
                ),
                input_description=(
                    "Texto que debe ser evaluado conforme a los criterios "
                    "de Lectura Fácil."
                ),
                output_description=(
                    "Informe consolidado con los resultados de los críticos "
                    "especializados y una conclusión global que indica si "
                    "el texto puede aprobarse o necesita nuevas "
                    "simplificaciones."
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

        logger.info(
            "Solicitud recibida. Texto: %d caracteres.",
            len(texto),
        )

        # 1. Descubrir agentes críticos
        agentes = await descubrir_agentes_criticos()
        logger.info("Agentes críticos descubiertos: %s",[a["name"] for a in agentes])

        # 2. Ejecutar los críticos especializados
        informes: list[CriticReport] = []

        for agente in agentes:
            logger.info("Ejecutando crítico: %s",agente["name"])

            try:
                informe = await llamar_agente(agente["url"],texto)
                informes.append(informe)

                logger.info("Informe recibido de %s: estado=%s, problemas=%d",agente["name"],informe.estado,len(informe.problemas))
                await asyncio.sleep(60)

            except Exception:
                logger.exception("Error ejecutando el crítico %s.",agente["name"])

        if not informes:
            raise RuntimeError(
                "No se ha podido obtener ningún informe crítico."
            )

        # 3. Preparar los informes para el LLM coordinador
        informes_texto = "\n\n".join(
            f"=== INFORME DEL CRÍTICO {informe.especialidad.upper()} ===\n"
            f"{informe.model_dump_json(indent=2)}"
            for informe in informes
        )

        logger.info(informes_texto)

        # 4. Analizar conjuntamente los informes
        system_prompt = f"""Eres el coordinador de evaluación de Lectura Fácil conforme a la norma UNE 153101:2018.

        Has recibido informes independientes de agentes críticos especializados.
        Cada agente ha evaluado exclusivamente su especialidad:
        - vocabulario
        - oraciones
        - estructura

        Tu responsabilidad es analizar los informes conjuntamente y decidir si el texto necesita otra ronda de simplificación o si puede darse por terminado.

        INFORMES RECIBIDOS:
        {informes_texto}

        CRITERIO DE DECISIÓN:
        No debes ser excesivamente estricto.

        Debes valorar conjuntamente:
        - La cantidad de problemas.
        - La relevancia de las pautas incumplidas.
        - El impacto de los problemas sobre la comprensión.
        - Si los problemas son aislados o repetidos.
        - Si las recomendaciones aportadas producirían una mejora significativa.

        Debes recomendar continuar con la simplificación cuando existan problemas relevantes, numerosos, graves o que puedan dificultar claramente la comprensión.

        Si el texto presenta un nivel global adecuado y los problemas restantes son menores, recomienda APROBARLO.

        No debes inventar problemas.
        No debes realizar una nueva evaluación del texto.
        No debes modificar el texto.

        Basa tu decisión exclusivamente en los informes recibidos.

        FORMATO DE LA CONCLUSIÓN:

        1. DECISIÓN:
           - APROBADO
           - NECESITA_SIMPLIFICACION

        2. VALORACIÓN GLOBAL:
           Explica brevemente por qué has tomado esa decisión. Detallando si de manera general el texto es adecuado o no.


        La decisión debe representar una valoración global y no una simple suma de incumplimientos.
        """

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=(
                    "Analiza los informes de los críticos y determina "
                    "si el texto puede aprobarse."
                )
            ),
        ]

        logger.info("Analizando conjuntamente los informes críticos.")

        try:
            response = await self.llm.ainvoke(messages)

        except Exception:
            logger.exception("Error analizando los informes críticos.")
            raise

        conclusion = response.content

        logger.info("Conclusión global generada correctamente.\n"+conclusion)

        # 5. Construir respuesta
        resultado = (
            "=== INFORMES DE LOS CRÍTICOS ESPECIALIZADOS ===\n\n"
            f"{informes_texto}\n\n"
            "=== CONCLUSIÓN GLOBAL ===\n"
            f"{conclusion}"
        )

        logger.info("Informe crítico consolidado generado: %d caracteres.",len(resultado))

        return resultado


agent = CriticoAgent()
app = agent.build_app()


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )