# Agentes/critico_sintactico.py

import logging
import os
import uvicorn

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from utils.logger_config import setup_logging
from Agentes.agente import BaseA2AAgent
from utils.a2a import AgentCard, AgentSkill, Message, CriticReport, CriticReportLLM
from utils.reglas import formatear_reglas, reglas

load_dotenv()

PORT = 8008

logger = logging.getLogger(__name__)
setup_logging("critico_sintactico")
logger.info("Iniciando agente A2A en puerto %d.",PORT)


class CriticoSintacticoAgent(BaseA2AAgent):

    card = AgentCard(
        name="Critico Sintactico",
        description=(
            "Agente especializado en evaluar las oraciones de textos "
            "conforme a la norma UNE 153101:2018 de Lectura Fácil. "
            "Detecta problemas relacionados con la longitud, complejidad "
            "y construcción de las oraciones. No modifica el texto; "
            "genera un informe con los problemas sintácticos detectados "
            "y recomendaciones de modificación."
        ),
        url=f"http://localhost:{PORT}",
        skills=[
            AgentSkill(
                id="criticize-sentences",
                name="Evaluación de las oraciones",
                description=(
                    "Evalúa exclusivamente las características sintácticas "
                    "de las oraciones de un texto. Detecta oraciones largas "
                    "o complejas, subordinaciones, construcciones pasivas "
                    "y otros problemas contemplados en las reglas de "
                    "oraciones de Lectura Fácil."
                ),
                input_description=(
                    "Texto que debe ser evaluado desde el punto de vista "
                    "de las oraciones y su construcción sintáctica."
                ),
                output_description=(
                    "Informe estructurado con los problemas sintácticos "
                    "detectados, las pautas incumplidas y recomendaciones "
                    "concretas para simplificar las oraciones."
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
        Tu única responsabilidad es evaluar las características SINTÁCTICAS y las ORACIONES del texto recibido.

        No debes modificar el texto.

        Tu tarea consiste en detectar incumplimientos de las REGLAS dentro del bloque <REGLAS>.
        Debes ser exhaustivo y comprobar cada regla para todo el texto.
        Ten en cuenta que que las reglas pueden incumplirse más de una vez en el mismo texto.
        
        <REGLAS>
        {formatear_reglas(reglas['oracion'])}
        </REGLAS>

        Para cada problema relevante debes indicar:
        - elemento: oración o fragmento problemático.
        - pauta_id: identificador de la pauta incumplida.

        El pauta_id debe corresponder exactamente a uno de los identificadores
        de las reglas proporcionadas. No inventes identificadores.

        Si no detectas problemas sintácticos relevantes:
        - estado debe ser "adecuado".
        - problemas debe ser una lista vacía.
        - conclusion debe indicar que no es necesario seguir simplificando
          las oraciones.

        Si detectas problemas:
        - estado debe ser "necesita_simplificacion".
        - problemas debe contener los incumplimientos detectados.
        - conclusion debe indicar que las oraciones necesitan simplificación.

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

        logger.info("Ejecutando modelo de crítica sintáctica.")

        try:
            structured_llm = self.llm.with_structured_output(
                CriticReportLLM,
                method="json_schema"
            )
            informe_llm = await structured_llm.ainvoke(messages)

        except Exception as e:
            logger.error("Tipo de excepción: %s", type(e))
            logger.error("Contenido completo: %s", str(e))
            # Si es un error de openai/httpx, suele traer .response
            response = getattr(e, "response", None)
            if response is not None:
                logger.error("Status: %s", response.status_code)
                logger.error("Body crudo: %s", response.text)
            raise
        
        informe = CriticReport(
            especialidad="sintactico",
            estado=informe_llm.estado,
            problemas=informe_llm.problemas,
            conclusion=informe_llm.conclusion,
        )

        logger.info("Informe sintáctico generado: estado=%s, problemas=%d",informe.estado,len(informe.problemas))

        return informe.model_dump_json()


agent = CriticoSintacticoAgent()
app = agent.build_app()


if __name__ == "__main__":


    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )