"""FastAPI WebSocket protocol compatible with the assignment's HTML test client."""
import asyncio
import json
import logging
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.chat import ChatService
from app.config import Settings, get_settings
from app.security import AuthenticationError, authenticate

log = logging.getLogger(__name__)


class AuthFrame(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["auth"]
    token: str = Field(min_length=1)


class MessageFrame(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["message"]
    text: str = Field(min_length=1)


async def _make_service(settings: Settings) -> ChatService:
    from app.embeddings import make_embedder
    from app.store import make_store
    from app.llm import make_llm

    # Persistent local Qdrant initialization uses disk I/O: offload from the event loop.
    embedder = make_embedder(settings)
    store = await asyncio.to_thread(make_store, settings)
    llm = make_llm(settings)
    return ChatService(embedder, store, llm, settings.top_k)


def create_app(settings: Settings | None = None, service: ChatService | None = None) -> FastAPI:
    app = FastAPI(title="Mini RAG Chatbot Backend", version="1.0.0")
    app.state.settings = settings or get_settings()
    app.state.service = service
    app.state.service_lock = asyncio.Lock()

    async def service_for_request() -> ChatService:
        if app.state.service is None:
            async with app.state.service_lock:
                if app.state.service is None:
                    app.state.service = await _make_service(app.state.settings)
        return app.state.service

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/chat")
    async def chat_socket(ws: WebSocket) -> None:
        await ws.accept()
        settings: Settings = app.state.settings
        try:
            try:
                # Auth MUST be the first message; reject unauthenticated sessions.
                raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
                frame = AuthFrame.model_validate(json.loads(raw))
                user = authenticate(frame.token, settings.jwt_secret.get_secret_value())
            except (asyncio.TimeoutError, json.JSONDecodeError, ValidationError, AuthenticationError):
                await ws.send_json({"type": "auth_failed", "message": "Invalid, expired, or missing authentication token."})
                await ws.close(code=1008)
                return
            await ws.send_json({
                "type": "auth_success", "user_id": user.sub,
                "department": user.department, "level": user.level,
            })
            while True:
                try:
                    raw = await ws.receive_text()
                    frame = MessageFrame.model_validate(json.loads(raw))
                    question = frame.text.strip()
                    if not question or len(question) > settings.max_question_chars:
                        raise ValueError("Question cannot be blank or exceed the configured length.")
                except (json.JSONDecodeError, ValidationError, ValueError):
                    await ws.send_json({"type": "stream", "text": "Please send a valid non-empty message within the size limit."})
                    await ws.send_json({"type": "done"})
                    continue

                try:
                    service = await service_for_request()
                    async for delta in service.answer(user, question):
                        await ws.send_json({"type": "stream", "text": delta})
                except Exception:
                    # Preserve WebSocket and never expose stack traces, tokens, or provider credentials.
                    log.exception("Chat request failed")
                    await ws.send_json({"type": "stream", "text": "Sorry, the request could not be completed. Please try again."})
                finally:
                    await ws.send_json({"type": "done"})
        except WebSocketDisconnect:
            return

    return app


app = create_app()
