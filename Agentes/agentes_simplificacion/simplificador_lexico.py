# Agentes/simplificador_lexico_agente.py

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

PORT = 8007

logger = logging.getLogger(__name__)
setup_logging("simplificador_lexico")
logger.info("Iniciando agente A2A en puerto %d.", PORT)

class SimplificadorLexicoAgent(BaseA2AAgent):

    card = AgentCard(
        name="simplificador_lexico",
        description=(
            "Agente especializado en la simplificación del vocabulario de "
            "textos conforme a la norma UNE 153101:2018 de Lectura Fácil. "
            "Sustituye palabras difíciles, tecnicismos, términos abstractos, "
            "siglas no explicadas y expresiones ambiguas por alternativas más "
            "sencillas cuando estas puedan dificultar la comprensión. "
            "Mantiene siempre el significado y toda la información original. "
            "Puede recibir un texto directamente o un texto acompañado de un "
            "informe elaborado por el agente Crítico."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="simplify-vocabulary",
                name="Simplificación del vocabulario",
                description=(
                    "Adapta exclusivamente el vocabulario de un texto a "
                    "Lectura Fácil. Detecta y sustituye palabras poco "
                    "frecuentes, tecnicismos, términos abstractos, siglas "
                    "sin explicar y expresiones ambiguas por alternativas "
                    "más sencillas, siempre que esto no altere el significado. "
                    "Si recibe un informe del Crítico, aplica las modificaciones "
                    "léxicas indicadas en él."
                ),
                input_description=(
                    "Texto que necesita simplificación léxica. Puede incluir "
                    "únicamente el texto original o el texto acompañado de "
                    "un informe del agente Crítico con cambios sugeridos."
                ),
                output_description=(
                    "Texto con el vocabulario adaptado a Lectura Fácil, "
                    "manteniendo toda la información y el significado original. "
                    "Devuelve únicamente el texto modificado."
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

        logger.info("Solicitud recibida. Texto: %d caracteres. Informe crítico: %s", len(texto), "sí" if informe_critico else "no",)
        logger.info(informe_critico if informe_critico else "No se ha proporcionado informe crítico.")
        
        # Construir contexto
        if informe_critico:
            contexto_informe = f"""{informe_critico}"""
            logger.info("Se ha recibido un informe del agente Crítico.")
        else:
            contexto_informe = """No se ha proporcionado ningún informe."""
            logger.info( "No se ha recibido informe del agente Crítico.")

        # Prompt
        prompt = f"""Eres un experto en Lectura Fácil conforme a la norma UNE 153101:2018.
        Tu única responsabilidad es simplificar las características LÉXICAS del texto recibido aplicando las reglas en el bloque <REGLAS> y siguiendo las recomendaciones del bloque <INFORME>.

        <REGLAS>
        {reglas['vocabulario']}
        </REGLAS>

        <INFORME>
        {contexto_informe}
        </INFORME>

        IMPORTANTE:
        - No elimines información.
        - No inventes información.

        Devuelve ÚNICAMENTE el texto final simplificado.
        No incluyas explicaciones, comentarios, listas de cambios ni ningún otro contenido adicional.
        """

        messages = [
            SystemMessage(
                content=prompt
            ),
            HumanMessage(content=f"""TEXTO ORIGINAL: 
            
            
            {texto}"""),
        ]

        # Ejecutar modelo
        logger.info("Ejecutando modelo de simplificación léxica.")

        try:
            response = await self.llm.ainvoke(messages)
        except Exception:
            logger.exception(
                "Error al ejecutar el modelo de simplificación léxica."
            )
            raise

        resultado = response.content

        logger.info(
            "Resultado generado correctamente: %d caracteres.",
            len(resultado),
        )

        return resultado


agent = SimplificadorLexicoAgent()
app = agent.build_app()


if __name__ == "__main__":


    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )