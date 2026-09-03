# agents/base_a2a_agent.py

import sys
import os
from abc import ABC, abstractmethod

from fastapi import FastAPI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from utils.a2a import AgentCard, TaskSendParams, TaskResult
from dotenv import load_dotenv
load_dotenv()

class BaseA2AAgent(ABC):

    card: AgentCard

    def build_app(self) -> FastAPI:
        app = FastAPI(title=self.card.name)

        @app.get("/.well-known/agent.json")
        async def get_agent_card():
            return self.card.model_dump()

        @app.post("/tasks/send", response_model=TaskResult)
        async def send_task(params: TaskSendParams) -> TaskResult:
            try:
                output = await self.process(params.message)
                return TaskResult(id=params.id, status="completed", output=output)
            except Exception as exc:
                return TaskResult(id=params.id, status="failed", error=str(exc))

        return app

    @abstractmethod
    async def process(self, message_text: str) -> str:
        raise NotImplementedError