# Agentes/definicion_agente.py

import logging
import os
import uvicorn

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from Agentes.agente import BaseA2AAgent
from utils.a2a import AgentCard, AgentSkill, Message
from utils.logger_config import setup_logging

load_dotenv()

PORT = 8004

logger = logging.getLogger(__name__)
setup_logging("definidor")
logger.info("Iniciando agente A2A en puerto %d.",PORT)


class DefinidorAgent(BaseA2AAgent):

    card = AgentCard(
        name="definidor",
        description=(
            "Agente especializado en identificar palabras difíciles, "
            "técnicas o poco habituales de un texto de Lectura Fácil y "
            "generar definiciones sencillas en español. "
            "Se utiliza para facilitar la comprensión de términos que "
            "no pueden sustituirse por alternativas más sencillas."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="define-difficult-terms",
                name="Definición de términos difíciles",
                description=(
                    "Identifica los términos técnicos, difíciles o poco "
                    "habituales que pueden dificultar la comprensión de "
                    "un texto y proporciona una definición breve y "
                    "comprensible para cada uno."
                ),
                input_description=(
                    "Texto final adaptado a Lectura Fácil del que se "
                    "deben identificar los términos que necesitan "
                    "una explicación."
                ),
                output_description=(
                    "Lista de términos difíciles acompañados de "
                    "definiciones sencillas en español."
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

        system_prompt = """
        Eres un experto en lenguaje sencillo y Lectura Fácil.

        Tu única responsabilidad es identificar palabras o expresiones del texto que puedan resultar difíciles de comprender y proporcionar una definición sencilla de cada una.

        Debes:
        - Identificar únicamente palabras o expresiones que realmente puedan resultar difíciles, técnicas o poco habituales.
        - No definir palabras comunes que cualquier lector pueda comprender.
        - No modificar el texto original.
        - No inventar términos que no aparezcan en el texto.
        - Crear definiciones breves, claras y sencillas.
        - Utilizar palabras habituales para explicar los términos.
        - Evitar utilizar en la definición el mismo término que estás definiendo sin explicarlo.
        - Mantener el significado correcto del término según el contexto del texto.
        - Si un término tiene varios significados, utilizar únicamente el significado que corresponde al contexto.

        Devuelve ÚNICAMENTE el glosario.

        Utiliza exactamente este formato:

        PALABRA: definición sencilla

        Si no hay ninguna palabra que necesite definición, devuelve: NINGUNA
        """

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=f"TEXTO:\n{texto}"
            ),
        ]

        logger.info("Ejecutando modelo para generar definiciones.")

        try:
            response = await self.llm.ainvoke(messages)

        except Exception:
            logger.exception("Error al ejecutar el modelo de definición.")
            raise

        resultado = response.content.strip()

        logger.info("Glosario generado correctamente: %d caracteres.",len(resultado),)

        return resultado


agent = DefinidorAgent()
app = agent.build_app()


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )