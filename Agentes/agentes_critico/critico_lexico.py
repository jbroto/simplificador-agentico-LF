# Agentes/critico_lexico.py

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

PORT = 8010

logger = logging.getLogger(__name__)
setup_logging("critico_lexico")
logger.info("Iniciando agente A2A en puerto %d.",PORT)


class CriticoLexicoAgent(BaseA2AAgent):

    card = AgentCard(
        name="Critico Lexico",
        description=(
            "Agente especializado en evaluar el vocabulario de textos "
            "conforme a la norma UNE 153101:2018 de Lectura Fácil. "
            "Detecta palabras o expresiones demasiado complejas, "
            "tecnicismos, términos abstractos, siglas o expresiones "
            "que puedan dificultar la comprensión. No modifica el texto; "
            "genera un informe con los problemas léxicos detectados y "
            "recomendaciones de modificación."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="criticize-vocabulary",
                name="Evaluación del vocabulario",
                description=(
                    "Evalúa exclusivamente las características léxicas "
                    "de un texto. Detecta palabras poco frecuentes, "
                    "tecnicismos, términos abstractos, siglas no explicadas "
                    "y expresiones complejas o ambiguas que puedan "
                    "dificultar la comprensión en Lectura Fácil."
                ),
                input_description=(
                    "Texto que debe ser evaluado desde el punto de vista "
                    "del vocabulario."
                ),
                output_description=(
                    "Informe conciso con los problemas léxicos detectados "
                    "y recomendaciones concretas para simplificar el "
                    "vocabulario. No devuelve el texto modificado."
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
        Tu única responsabilidad es evaluar las características LÉXICAS del texto recibido.

        No debes modificar el texto.
        Tu tarea consiste en detectar incumplimientos de las REGLAS dentro del bloque <REGLAS>.
        Debes ser exhaustivo y comprobar cada regla para todo el texto.
        Ten en cuenta que que las reglas pueden incumplirse más de una vez en el mismo texto.

        <REGLAS>
        {formatear_reglas(reglas['vocabulario'])}
        </REGLAS>

        Para cada problema relevante debes indicar:
        - elemento: término o expresión problemática.
        - pauta_id: identificador de la pauta incumplida (por ejemplo, G01, G02, G03).

        El pauta_id debe corresponder exactamente a uno de los identificadores
        de las reglas proporcionadas. No inventes identificadores.

        Si no detectas problemas:
        - estado debe ser "adecuado".
        - problemas debe ser una lista vacía.
        - conclusion debe indicar que no es necesario seguir simplificando
          el vocabulario.

        Si detectas problemas:
        - estado debe ser "necesita_simplificacion".
        - problemas debe contener los incumplimientos detectados.
        - conclusion debe indicar que el vocabulario necesita simplificación.

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

        logger.info( "Ejecutando modelo de crítica léxica.")

        try:
            structured_llm = self.llm.with_structured_output(
                CriticReportLLM,
                method="json_schema"
            )
            informe_llm = await structured_llm.ainvoke(messages)

        except Exception:
            logger.exception("Error al ejecutar el modelo de crítica léxica.")
            raise
        
        informe = CriticReport(
            especialidad="lexico",
            estado=informe_llm.estado,
            problemas=informe_llm.problemas,
            conclusion=informe_llm.conclusion,
        )
        
        logger.info("Informe léxico generado: estado=%s, problemas=%d",informe.estado,len(informe.problemas))

        return informe.model_dump_json()


agent = CriticoLexicoAgent()
app = agent.build_app()


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )