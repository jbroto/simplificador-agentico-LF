# main.py

import subprocess
import asyncio
import sys
import os
import logging
import uuid
import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.a2a import Message, TextPart, TaskSendParams, TaskResult
from utils.registro_agentes import AGENTES_REGISTRADOS
from dotenv import load_dotenv
from utils.logger_config import setup_logging
load_dotenv()

logger = logging.getLogger(__name__)

TIMEOUT = None

AGENTES = [

    ("Agentes.simplificador_agente:app", 8001),
    ("Agentes.critico_agente:app",       8002),
    ("Agentes.supervisor_agente:app",    8003),
    ("Agentes.definicion_agente:app",    8004),

    # Agentes simplificadores especializados
    ("Agentes.agentes_simplificacion.simplificador_sintactico:app",   8005),
    ("Agentes.agentes_simplificacion.simplificador_estructural:app", 8006),
    ("Agentes.agentes_simplificacion.simplificador_lexico:app",      8007),

    # Agentes críticos especializados
    ("Agentes.agentes_critico.critico_sintactico:app",   8008),
    ("Agentes.agentes_critico.critico_estructural:app", 8009),
    ("Agentes.agentes_critico.critico_lexico:app",      8010),
]


api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimplificarRequest(BaseModel):
    texto: str

@api.post("/simplificar")
async def simplificar(req: SimplificarRequest):
    params = TaskSendParams(
        id=str(uuid.uuid4()),
        message=Message(role="user", parts=[TextPart(text=req.texto)])
    )
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{AGENTES_REGISTRADOS['supervisor']}/tasks/send",
            json=params.model_dump()
        )
        result = TaskResult(**response.json())

    if result.status == "completed":
        logger.info("Tarea Completada: %s", result)
        return {"resultado": result.output}
    else:
        logger.error("Error en la tarea: %s", result.error)
        return {"error": result.error}

def lanzar_agentes() -> list:
    procesos = []
    for modulo, puerto in AGENTES:
        p = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            modulo,
            "--host", "0.0.0.0",
            "--port", str(puerto),
        ])
        procesos.append(p)
        logger.info("Agente lanzado: %s en puerto %d (pid %d)",modulo,puerto,p.pid,)
    return procesos


async def esperar_agentes():
    logger.info("Esperando a que los agentes arranquen...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        for nombre, url in AGENTES_REGISTRADOS.items():
            for _ in range(20):
                try:
                    r = await client.get(f"{url}/.well-known/agent.json")
                    if r.status_code == 200:
                        logger.info("Agente '%s' listo en %s",nombre,url)
                        break
                except Exception:
                    await asyncio.sleep(1)
            else:
                raise RuntimeError(f"El agente '{nombre}' en {url} no arrancó a tiempo")


async def main():
    procesos = lanzar_agentes()
    
    try:
        await esperar_agentes()
        logger.info("Todos los agentes están listos. "
            "API escuchando en http://localhost:8000")


        config = uvicorn.Config(api, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    finally:
        logger.info("Apagando agentes...")
        for p in procesos:
            p.terminate()


if __name__ == "__main__":
    setup_logging("main")
    asyncio.run(main())