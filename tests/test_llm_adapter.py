import json

import httpx
import pytest

from app.llm import OpenAICompatibleLLM


@pytest.mark.asyncio
async def test_groq_tool_selection_and_real_sse_delta_stream(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["stream"] is False:
            return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{
                "id": "call-123", "type": "function", "function": {
                    "name": "get_employee_context", "arguments": '{"user_id":"emp-001"}'
                }
            }]}}]})
        return httpx.Response(200, content=(
            'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
            'data: [DONE]\n\n'
        ), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def client_with_transport(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr("app.llm.httpx.AsyncClient", client_with_transport)
    llm = OpenAICompatibleLLM("https://example.test/v1", "sample-model", "test-token", 5, is_groq=True)
    conversation = [{"role": "user", "content": "my manager"}]
    selected = await llm.select_tool(conversation)
    assert selected["function"]["name"] == "get_employee_context"
    chunks = [delta async for delta in llm.stream(conversation)]
    assert chunks == ["Hello ", "world"]
    assert requests[0]["tools"][0]["function"]["name"] == "get_employee_context"
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[1]["stream"] is True
