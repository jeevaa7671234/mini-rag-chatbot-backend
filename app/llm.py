"""OpenAI-compatible, *asynchronous* streaming adapters for Groq and Ollama."""
import json
import re
from collections.abc import AsyncIterator
from typing import Protocol

import httpx

from app.tools import TOOL_SPEC


class LLMError(Exception):
    """Redacted provider error suitable for application-level handling."""


class ChatLLM(Protocol):
    async def select_tool(self, messages: list[dict]) -> dict | None: ...
    def stream(self, messages: list[dict]) -> AsyncIterator[str]: ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float, *, is_groq: bool):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.is_groq = is_groq

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key or 'ollama'}", "Content-Type": "application/json"}

    async def select_tool(self, messages: list[dict]) -> dict | None:
        """Short nonstreaming tool-choice pass; the actual answer is streamed separately."""
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": [TOOL_SPEC],
            "stream": False,
            "temperature": 0,
        }
        # Ollama's OpenAI-compatible endpoint does not support tool_choice.
        if self.is_groq:
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, headers=self._headers(), json=payload)
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            raise LLMError("The language model is temporarily unavailable.") from exc
        calls = message.get("tool_calls") or []
        if len(calls) > 1:
            raise LLMError("The language model requested unsupported multiple tools.")
        return calls[0] if calls else None

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.2,
        }
        emitted = False
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.url, headers=self._headers(), json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        content = line[5:].strip()
                        if content == "[DONE]":
                            break
                        if not content:
                            continue
                        packet = json.loads(content)
                        if "error" in packet:
                            raise LLMError("The language model returned an error while streaming.")
                        choices = packet.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            token = delta.get("content")
                            if token:
                                emitted = True
                                yield token  # forward actual provider deltas; don't simulate streaming
        except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError) as exc:
            raise LLMError("The language model stream was interrupted.") from exc
        if not emitted:
            raise LLMError("The language model returned an empty response.")


class MockLLM:
    """Offline tests and demo ONLY. Not an LLM, and does not produce policy answers."""

    async def select_tool(self, messages: list[dict]) -> dict | None:
        message = messages[-1]["content"]
        query = message.lower().split("question:", 1)[-1].split("retrieved policy excerpts", 1)[0]
        if any(phrase in query for phrase in ("my manager", "my grade", "my team", "given my grade")):
            # Generate the exact same structure the OpenAI tool API returns.
            authenticated_id = re.search(r"^Authenticated user ID: (\S+)$", message, re.MULTILINE)
            if not authenticated_id:
                raise LLMError("Test provider could not identify the authenticated user.")
            return {
                "id": "test-call-1",
                "type": "function",
                "function": {
                    "name": "get_employee_context",
                    "arguments": json.dumps({"user_id": authenticated_id.group(1)}),
                },
            }
        return None

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        tool_data = next((m["content"] for m in messages if m.get("role") == "tool"), None)
        if tool_data:
            value = json.loads(tool_data)
            answer = f"[DEMO ONLY] Manager: {value['manager']}; grade: {value['grade']}; team: {value['team_name']}."
        else:
            answer = "[DEMO ONLY] Retrieval and access checks completed. Configure Ollama or Groq for actual answers."
        for word in answer.split(" "):
            yield word + " "


def make_llm(settings) -> ChatLLM:
    if settings.llm_provider == "mock":
        return MockLLM()
    if settings.llm_provider == "groq":
        key = settings.groq_api_key.get_secret_value()
        if not key:
            raise LLMError("Set GROQ_API_KEY in .env to use Groq.")
        return OpenAICompatibleLLM(
            "https://api.groq.com/openai/v1", settings.groq_model, key,
            settings.request_timeout_seconds, is_groq=True
        )
    return OpenAICompatibleLLM(
        settings.ollama_base_url, settings.ollama_model, "ollama",
        settings.request_timeout_seconds, is_groq=False
    )
