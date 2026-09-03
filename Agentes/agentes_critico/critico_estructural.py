# Agentes/critico_estructural.py

import logging
import os
import uvicorn

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from utils.logger_config import setup_logging
from Agentes.agente import BaseA2AAgent
from utils.a2a import AgentCard, AgentSkill, Message, CriticReport, CriticReportLLM
from utils.reglas import reglas, formatear_reglas

load_dotenv()

PORT = 8009

logger = logging.getLogger(__name__)
setup_logging("critico_estructural")
logger.info("Iniciando agente A2A en puerto %d.",PORT)


class CriticoEstructuralAgent(BaseA2AAgent):

    card = AgentCard(
        name="Critico Estructural",
        description=(
            "Agente especializado en evaluar la estructura de textos "
            "conforme a la norma UNE 153101:2018 de Lectura Fácil. "
            "Detecta problemas relacionados con la organización de los "
            "párrafos, la distribución de las ideas, los conectores y "
            "la cohesión global del texto. No modifica el texto; "
            "genera un informe con los problemas estructurales detectados "
            "y recomendaciones de modificación."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="criticize-structure",
                name="Evaluación de la estructura",
                description=(
                    "Evalúa exclusivamente la estructura de un texto. "
                    "Detecta párrafos demasiado largos, párrafos que "
                    "mezclan varias ideas, problemas de organización, "
                    "conectores poco adecuados y otros incumplimientos "
                    "de las reglas estructurales de Lectura Fácil."
                ),
                input_description=(
                    "Texto que debe ser evaluado desde el punto de vista "
                    "de su estructura y organización."
                ),
                output_description=(
                    "Informe estructurado con los problemas estructurales "
                    "detectados, las pautas incumplidas y recomendaciones "
                    "concretas para mejorar la estructura."
                ),
            )
        ],
    )

    llm = ChatGroq(
        model=os.environ.get("MODEL"),
        api_key=os.environ.get("GROQ"),
        temperature=0.0,
        max_tokens=2048,
    )

    async def process(self, message: Message) -> str:

        texto = message.parts[0].text

        logger.info(
            "Solicitud recibida. Texto: %d caracteres.",
            len(texto),
        )

        system_prompt = f"""Eres un crítico experto en Lectura Fácil conforme a la norma UNE 153101:2018.
        Tu única responsabilidad es evaluar las características ESTRUCTURALES del texto recibido.

        No debes modificar el texto.
        Tu tarea consiste en detectar incumplimientos de las REGLAS dentro del bloque <REGLAS>.
        Debes ser exhaustivo y comprobar cada regla para todo el texto.
        Ten en cuenta que que las reglas pueden incumplirse más de una vez en el mismo texto.

        <REGLAS>
        {formatear_reglas(reglas['estructura'])}
        </REGLAS>

        Para cada problema relevante debes indicar:
        - elemento: párrafo, sección o fragmento problemático.
        - pauta_id: identificador de la pauta incumplida.

        El pauta_id debe corresponder exactamente a uno de los identificadores
        de las reglas proporcionadas. No inventes identificadores.

        Si no detectas problemas estructurales relevantes:
        - estado debe ser "adecuado".
        - problemas debe ser una lista vacía.
        - conclusion debe indicar que no es necesario seguir simplificando
          la estructura.

        Si detectas problemas:
        - estado debe ser "necesita_simplificacion".
        - problemas debe contener los incumplimientos detectados.
        - conclusion debe indicar que la estructura necesita simplificación.

        La respuesta debe ser un JSON válido que siga exactamente la estructura de CriticReport.
        """

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=f"""TEXTO A EVALUAR:
                
                
                {texto}"""
            ),
        ]

        logger.info("Ejecutando modelo de crítica estructural.")

        try:
            structured_llm = self.llm.with_structured_output(
                CriticReportLLM,
                method="json_schema"
            )
            informe_llm = await structured_llm.ainvoke(messages)

        except Exception:
            logger.exception("Error al ejecutar el modelo de crítica estructural.")
            raise
        
        informe = CriticReport(
            especialidad="estructural",
            estado=informe_llm.estado,
            problemas=informe_llm.problemas,
            conclusion=informe_llm.conclusion,
        )

        logger.info("Informe estructural generado: estado=%s, problemas=%d",informe.estado,len(informe.problemas))

        return informe.model_dump_json()


agent = CriticoEstructuralAgent()
app = agent.build_app()


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )