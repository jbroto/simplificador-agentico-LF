# Agentes/simplificador_sintactico.py

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

PORT = 8005

logger = logging.getLogger(__name__)
setup_logging("simplificador_sintactico")
logger.info("Iniciando agente A2A en puerto %d.", PORT)


class SimplificadorSintacticoAgent(BaseA2AAgent):

    card = AgentCard(
        name="simplificador_oraciones",
        description=(
            "Agente especializado en la simplificación sintáctica de "
            "textos conforme a la norma UNE 153101:2018 de Lectura Fácil. "
            "Simplifica y divide oraciones largas o complejas, reduce "
            "estructuras sintácticas difíciles y mejora la claridad de "
            "las oraciones manteniendo siempre el significado y toda la "
            "información original. Puede recibir un texto directamente "
            "o un texto acompañado de un informe elaborado por el agente "
            "Crítico."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="simplify-sentences",
                name="Simplificación sintáctica de oraciones",
                description=(
                    "Adapta exclusivamente la estructura sintáctica de las "
                    "oraciones a Lectura Fácil. Detecta y simplifica oraciones "
                    "largas o complejas, subordinaciones encadenadas, "
                    "construcciones pasivas y otras estructuras que dificulten "
                    "la comprensión. Puede dividir oraciones cuando sea "
                    "necesario, manteniendo siempre toda la información y "
                    "el significado original. Si recibe un informe del "
                    "Crítico, aplica las modificaciones sintácticas indicadas."
                ),
                input_description=(
                    "Mensaje que contiene el texto que necesita simplificación "
                    "sintáctica y, opcionalmente, un informe del agente Crítico "
                    "con las modificaciones sintácticas recomendadas."
                ),
                output_description=(
                    "Texto con las oraciones adaptadas a Lectura Fácil, "
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

        logger.info("Solicitud recibida. Texto: %d caracteres. Informe crítico: %s",len(texto), "sí" if informe_critico else "no")
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
        Tu única responsabilidad es simplificar las características SINTÁCTICAS de las ORACIONES del texto recibido aplicando las reglas del bloque <REGLAS> y siguiendo las recomendaciones del bloque <INFORME>.

        <REGLAS>
        {reglas['oracion']}
        </REGLAS>

        <INFORME>
        {contexto_informe}
        </INFORME>

        IMPORTANTE:
        - No elimines información.
        - No inventes información.
        - No alteres el significado.

        Devuelve ÚNICAMENTE el texto final simplificado.
        No incluyas explicaciones, comentarios, listas de cambios ni ningún otro contenido adicional.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""TEXTO ORIGINAL: 
            
            
            {texto}"""),
        ]

        # Ejecutar modelo
        logger.info("Ejecutando modelo de simplificación sintáctica.")

        try:
            response = await self.llm.ainvoke(messages)
        except Exception:
            logger.exception(
                "Error al ejecutar el modelo de simplificación sintáctica."
            )
            raise

        resultado = response.content

        logger.info("Resultado generado correctamente: %d caracteres.",len(resultado))

        return resultado


agent = SimplificadorSintacticoAgent()
app = agent.build_app()


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )