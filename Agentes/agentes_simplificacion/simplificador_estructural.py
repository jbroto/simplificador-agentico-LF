# Agentes/simplificador_estructural.py

import logging
import os
import uvicorn

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from utils.logger_config import setup_logging
from Agentes.agente import BaseA2AAgent
from utils.a2a import AgentCard, AgentSkill, Message
from utils.reglas import reglas

load_dotenv()

PORT = 8006

logger = logging.getLogger(__name__)
setup_logging("simplificador_estructural")
logger.info("Iniciando agente A2A en puerto %d.",PORT)

class SimplificadorEstructuralAgent(BaseA2AAgent):

    card = AgentCard(
        name="simplificador_estructural",
        description=(
            "Agente especializado en la simplificación de la estructura "
            "y organización de textos conforme a la norma UNE 153101:2018 "
            "de Lectura Fácil. Reorganiza la información en párrafos claros "
            "y coherentes, mejora la cohesión global, utiliza conectores "
            "sencillos y reduce redundancias cuando sea necesario, "
            "manteniendo siempre toda la información y el significado "
            "original. Puede recibir un texto directamente o un texto "
            "acompañado de un informe elaborado por el agente Crítico."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="simplify-structure",
                name="Simplificación estructural de textos",
                description=(
                    "Adapta exclusivamente la estructura y organización "
                    "global de un texto a Lectura Fácil. Reorganiza el "
                    "contenido en párrafos cortos y claros, mejora la "
                    "cohesión entre ideas, utiliza conectores sencillos "
                    "y elimina redundancias innecesarias sin eliminar "
                    "información relevante. Si recibe un informe del "
                    "Crítico, aplica las modificaciones estructurales "
                    "indicadas en él."
                ),
                input_description=(
                    "Mensaje que contiene el texto que necesita "
                    "simplificación estructural y, opcionalmente, un "
                    "informe del agente Crítico con las modificaciones "
                    "estructurales recomendadas."
                ),
                output_description=(
                    "Texto reorganizado y estructurado conforme a las "
                    "pautas de Lectura Fácil, manteniendo toda la "
                    "información y el significado original. Devuelve "
                    "únicamente el texto modificado."
                ),
            )
        ],
    )

    llm = ChatGroq(
        model=os.environ.get("MODEL"),
        api_key=os.environ.get("GROQ"),
        temperature=0,
        max_tokens=2048,
    )

    async def process(self, message: Message) -> str:

        # Extraer texto e informe
        texto = message.parts[0].text
        informe_critico = message.informe

        logger.info("Solicitud recibida. Texto: %d caracteres. Informe crítico: %s",len(texto),"sí" if informe_critico else "no")
        logger.info(informe_critico if informe_critico else "No se ha proporcionado informe crítico.")

        # Construir contexto
        if informe_critico:
            contexto_informe = f"""{informe_critico}"""
            logger.info("Se ha recibido un informe del agente Crítico.")
        else:
            contexto_informe = """No se ha proporcionado ningún informe."""
            logger.info( "No se ha recibido informe del agente Crítico.")

        # System prompt
        system_prompt = f"""Eres un experto en Lectura Fácil conforme a la norma UNE 153101:2018.
        Tu única responsabilidad es simplificar las características ESTRUCTURALES del texto recibido siguiendo las reglas en el bloque <REGLAS> y las recomendaciones en <INFORME>.

        <REGLAS>
        {reglas['estructura']}
        </REGLAS>

        <INFORME>
        {contexto_informe}
        </INFORME>

        IMPORTANTE:
        - No elimines información.
        - No inventes información.
        - No realices cambios léxicos innecesarios.

        Devuelve ÚNICAMENTE el texto final reestructurado.
        No incluyas explicaciones, comentarios, listas de cambios ni ningún otro contenido adicional.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""TEXTO ORIGINAL: 
            
            
            {texto}"""),
        ]

        # Ejecutar modelo
        logger.info("Ejecutando modelo de simplificación estructural.")

        try:
            response = await self.llm.ainvoke(messages)
        except Exception:
            logger.exception(
                "Error al ejecutar el modelo de simplificación estructural."
            )
            raise

        resultado = response.content

        logger.info("Resultado generado correctamente: %d caracteres.",len(resultado))

        return resultado


agent = SimplificadorEstructuralAgent()
app = agent.build_app()


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )