"""Callback handlers used in the app."""
import logging
from typing import Any, Dict, List, Optional
from langchain.callbacks.base import AsyncCallbackHandler
from schemas import ChatResponse
from uuid import UUID
from langchain.schema import LLMResult


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket, client_id):
        self.websocket = websocket
        self.client_id = client_id

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        await self.websocket.send_json(resp.dict())

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running."""
        logging.debug(f"[{self.client_id}] [stream-tutor] [PROMPT] - {prompts}")

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        logging.debug(f"[{self.client_id}] [stream-tutor] [RESPONSE] - {response}")


class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket, client_id):
        self.websocket = websocket
        self.client_id = client_id

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running."""
        resp = ChatResponse(sender="bot", message="Exercise generation...", type="info")
        logging.debug(f"[{self.client_id}] - Question generation started...")
        logging.debug(f"[{self.client_id}] - {prompts}")
        await self.websocket.send_json(resp.dict())

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        logging.debug(f"[{self.client_id}] - Response: {response}")
        logging.debug(f"[{self.client_id}] - Question generation end...")
