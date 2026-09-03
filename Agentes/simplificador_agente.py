# Agentes/simplificador_agente.py
import sys
import os
import uuid
import httpx
import logging
import uvicorn
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from utils.registro_agentes import AGENTES_SIMPLIFICACION

from Agentes.agente import BaseA2AAgent
from utils.a2a import (
    AgentCard,
    AgentSkill,
    Message,
    TaskResult,
    TaskSendParams,
    TextPart,
)
from utils.logger_config import setup_logging

load_dotenv()

PORT = 8001
TIMEOUT = 300

logger = logging.getLogger(__name__)
setup_logging("simplificador")
logger.info("Iniciando Simplificador A2A en puerto %d...",PORT)


# Llamada A2A
async def _call_agent(url: str,texto: str,informe_critico: str | None = None) -> str:

    message = Message(
        role="user",
        parts=[
            TextPart(text=texto)
        ],
        informe=informe_critico,
    )

    params = TaskSendParams(
        id=str(uuid.uuid4()),
        message=message,
    )

    logger.info("Enviando tarea A2A a %s",url)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        response = await client.post(f"{url}/tasks/send",json=params.model_dump())
        response.raise_for_status()
        result = TaskResult(**response.json())

    if result.status == "completed":
        if result.output is None:
            raise RuntimeError(f"El agente {url} terminó correctamente pero no devolvió nada.")
        return result.output

    raise RuntimeError(
        f"Agente en {url} falló: {result.error}"
    )

# Limpiar informe por especialidad
def obtener_informe_especialidad(informe, especialidad):
    if not informe:
        return None

    marcador = f"=== INFORME DEL CRÍTICO {especialidad.upper()} ==="

    for bloque in informe.split("=== INFORME DEL CRÍTICO "):
        if bloque.startswith(especialidad.upper()):
            return marcador + "\n" + bloque.split("=== CONCLUSIÓN GLOBAL ===")[0]

    return None


# Descubrimiento de agentes (se mantiene: sigue siendo A2A real,
# solo que ya no se usa para construir StructuredTools)
async def descubrir_agentes() -> dict:
    """Devuelve {nombre: {url, description}} consultando cada Agent Card."""
    info = {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        for nombre, url in AGENTES_SIMPLIFICACION.items():

            logger.info("Descubriendo Agent Card de %s", nombre)
            response = await client.get(f"{url}/.well-known/agent.json")
            response.raise_for_status()
            card = response.json()

            info[card["name"]] = {
                "url": url,
                "description": card.get("description", ""),
            }

    return info


# Simplificador
class SimplificadorAgent(BaseA2AAgent):

    # Agent Card
    card = AgentCard(
        name="Simplificador",
        description=(
            "Suborquestador especializado en coordinar la simplificación "
            "de textos conforme a la norma UNE 153101:2018 de Lectura Fácil. "
            "No modifica directamente el texto. Analiza el texto y el "
            "informe del agente Crítico y decide qué agentes especializados "
            "de simplificación deben intervenir y en qué orden. Coordina "
            "mediante el protocolo A2A los agentes de simplificación "
            "sintáctica, léxica y estructural."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="orchestrate-simplification",
                name="Orquestación de agentes de simplificación",
                description=(
                    "Coordina los agentes especializados de simplificación "
                    "sintáctica, léxica y estructural. Utiliza el texto "
                    "recibido y, cuando está disponible, el informe del "
                    "agente Crítico para determinar qué agentes son "
                    "necesarios y en qué orden deben ejecutarse."
                ),
                input_description=(
                    "Mensaje A2A que contiene el texto y opcionalmente "
                    "un informe del agente Crítico con los cambios "
                    "recomendados."
                ),
                output_description=(
                    "Texto final después de ejecutar los agentes de "
                    "simplificación necesarios."
                ),
            )
        ],
    )

    # LLM
    llm = ChatGroq(
        model=os.environ.get("MODEL"),
        api_key=os.environ.get("GROQ"),
        temperature=0,
        max_tokens=256,
    )

    # Process
    async def process(self, message: Message) -> str:
        try:
            texto = message.parts[0].text
            informe_critico = message.informe

            logger.info(
                "Texto recibido: %d caracteres. Informe crítico: %s",
                len(texto),
                "sí" if informe_critico else "no",
            )

            # Descubrir agentes especializados (A2A real, solo para info)
            agentes_info = await descubrir_agentes()
            logger.info("Agentes descubiertos: %s", list(agentes_info.keys()))

            # Prompt del suborquestador
            if informe_critico:
                contexto_informe = f"""{informe_critico}"""
            else:
                contexto_informe = "No se ha recibido ningún informe del agente Crítico."

            lista_agentes = "\n".join(
                f"- {nombre}: {datos['description']}"
                for nombre, datos in agentes_info.items()
            )

            system_prompt = f"""Eres un suborquestador experto en Lectura Fácil conforme a la norma UNE 153101:2018.
Tu función NO es simplificar directamente el texto.
Tu única tarea es decidir qué agentes especializados deben intervenir, y en qué orden,
basándote en el informe del bloque <INFORME>.

<INFORME>
{contexto_informe}
</INFORME>

AGENTES DISPONIBLES:
{lista_agentes}

REGLAS:
1. Analiza los problemas del texto y el informe del Crítico.
2. Utiliza únicamente los agentes necesarios.
3. No incluyas un agente si no tiene trabajo que realizar.
4. Puedes incluir varios agentes, en el orden en que deben ejecutarse.

Responde ÚNICAMENTE con los nombres de los agentes a usar, separados por comas,
en el orden en que deben ejecutarse. No incluyas explicaciones ni texto adicional.

Ejemplo de respuesta válida:
simplificador_oraciones, simplificador_estructural, simplificador_lexico

Si no hace falta ningún agente, responde exactamente:
NINGUNO
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"TEXTO: {texto}"),
            ]

            logger.info("Pidiendo decisión de orquestación al LLM.")
            response = await self.llm.ainvoke(messages)

            decision = response.content.strip()
            logger.info("Decisión del orquestador: %s", decision)

            if decision.upper() == "NINGUNO":
                agentes_a_llamar = []
            else:
                agentes_a_llamar = [a.strip() for a in decision.split(",") if a.strip()]

            texto_actual = texto

            for tool_name in agentes_a_llamar:
                if tool_name not in AGENTES_SIMPLIFICACION:
                    logger.warning("Agente desconocido ignorado: %s", tool_name)
                    continue

                logger.info("Ejecutando agente especializado: %s", tool_name)
                especialidad = {
                    "simplificador_lexico": "lexico",
                    "simplificador_oraciones": "sintactico",
                    "simplificador_estructural": "estructural",
                }[tool_name.lower()]

                informe_especializado = obtener_informe_especialidad(
                    informe_critico,
                    especialidad,
                )

                resultado = await _call_agent(
                    url=AGENTES_SIMPLIFICACION[tool_name],
                    texto=texto_actual,
                    informe_critico=informe_especializado,
                )

                logger.info("Resultado de %s: %d caracteres.", tool_name, len(resultado))
                texto_actual = resultado
                await asyncio.sleep(60)

            # Resultado
            logger.info("Simplificación finalizada. Resultado: %d caracteres.", len(texto_actual))
            return texto_actual

        except Exception as e:
            logger.exception("Error durante la simplificación: %s", str(e))
            raise


# Aplicación A2A
agent = SimplificadorAgent()
app = agent.build_app()


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )