from pydantic import BaseModel
from typing import Optional, Literal

##Mensaje 

class TextPart(BaseModel):
    """Parte de texto dentro de un mensaje A2A."""
    type: Literal["text"] = "text"
    text: str


class Message(BaseModel):
    """Mensaje A2A intercambiado entre agentes."""
    role: Literal["user", "agent"]
    parts: list[TextPart]
    informe: Optional[str] = None


##Tareas

class TaskSendParams(BaseModel):
    """Parámetros para enviar una tarea a un agente A2A."""
    id: str
    message: Message


class TaskResult(BaseModel):
    """Resultado de una tarea A2A."""
    id: str
    status: Literal["submitted", "working", "completed", "failed"]
    output: Optional[str] = None
    error: Optional[str] = None


##Agent Card

class AgentSkill(BaseModel):
    """Lo que puede hacer el agente"""
    id: str
    name: str
    description: str
    input_description: Optional[str] = None
    output_description: Optional[str] = None



class AgentCard(BaseModel):
    """
    Agent Card: descriptor público del agente, accesible en /.well-known/agent.json
    Permite al orquestador descubrir qué hace el agente sin conocer su implementación.
    """
    name: str
    description: str
    url: str
    version: str = "1.0"
    #capabilities: AgentCapabilities = AgentCapabilities()
    skills: list[AgentSkill] = []

##Informe para los Críticos

class CriticFinding(BaseModel):
    elemento: str
    pauta_id: str

class CriticReportLLM(BaseModel):
    estado: Literal[
        "adecuado",
        "necesita_simplificacion",
    ]
    problemas: list[CriticFinding]
    conclusion: str


class CriticReport(BaseModel):
    especialidad: Literal[
        "lexico",
        "sintactico",
        "estructural",
    ]

    estado: Literal[
        "adecuado",
        "necesita_simplificacion",
    ]

    problemas: list[CriticFinding]

    conclusion: str