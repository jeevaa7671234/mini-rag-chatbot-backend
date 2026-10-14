from app.embeddings import HashingEmbedder
from app.ingestion import Chunk
from app.store import SqliteVectorStore
import asyncio


def test_websocket_streams_multiple_turns_and_stays_connected(client, token_factory, config):
    store = SqliteVectorStore(config.sqlite_path)
    emb = HashingEmbedder()
    sample = [Chunk("id", "Sample vacation policy", "hr", 1, "leave_policy.pdf", 1)]
    asyncio.run(store.replace_all(sample, asyncio.run(emb.embed([sample[0].text])), emb.dimensions))

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "token": token_factory()})
        assert ws.receive_json() == {"type": "auth_success", "user_id": "emp-001", "department": "hr", "level": 1}
        ws.send_json({"type": "message", "text": "What is the vacation policy?"})
        frames = []
        while (frame := ws.receive_json())["type"] != "done":
            frames.append(frame)
        assert len(frames) >= 2
        assert all(f["type"] == "stream" for f in frames)
        assert "DEMO ONLY" in "".join(f["text"] for f in frames)
        ws.send_json({"type": "message", "text": " "})
        assert "valid" in ws.receive_json()["text"]
        assert ws.receive_json() == {"type": "done"}
        ws.send_json({"type": "message", "text": "How does it work?"})
        assert ws.receive_json()["type"] == "stream"
        while ws.receive_json()["type"] != "done":
            pass


def test_llm_failure_is_user_facing_and_does_not_close_socket(config, token_factory):
    from fastapi.testclient import TestClient
    from app.chat import ChatService
    from app.main import create_app

    class FailingLLM:
        async def select_tool(self, messages):
            raise RuntimeError("PRIVATE API ERROR: should never reach client")

        async def stream(self, messages):
            yield "never"

    store = SqliteVectorStore(config.sqlite_path)
    emb = HashingEmbedder()
    chunk = Chunk("id", "leave", "hr", 1, "leave_policy.pdf", 1)
    asyncio.run(store.replace_all([chunk], asyncio.run(emb.embed([chunk.text])), emb.dimensions))
    service = ChatService(emb, store, FailingLLM(), 5)
    with TestClient(create_app(settings=config, service=service)) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "auth", "token": token_factory()})
            assert ws.receive_json()["type"] == "auth_success"
            for _ in range(2):
                ws.send_json({"type": "message", "text": "What is leave policy?"})
                error = ws.receive_json()
                assert error["type"] == "stream" and "try again" in error["text"]
                assert "PRIVATE API ERROR" not in error["text"]
                assert ws.receive_json() == {"type": "done"}


def test_tool_failure_is_redacted_and_connection_remains_open(config, token_factory, monkeypatch):
    """An exception in one of the three mocked APIs must not kill an authenticated socket."""
    from fastapi.testclient import TestClient
    from app.chat import ChatService
    from app.llm import MockLLM
    from app.main import create_app

    async def unavailable(user_id):
        raise RuntimeError("PRIVATE HR SERVICE KEY leaked in internal error")

    monkeypatch.setattr("app.chat.get_employee_context", unavailable)
    store = SqliteVectorStore(config.sqlite_path)
    emb = HashingEmbedder()
    chunk = Chunk("id", "leave guidance", "hr", 1, "leave_policy.pdf", 1)
    asyncio.run(store.replace_all([chunk], asyncio.run(emb.embed([chunk.text])), emb.dimensions))
    service = ChatService(emb, store, MockLLM(), 5)
    with TestClient(create_app(settings=config, service=service)) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "auth", "token": token_factory()})
            assert ws.receive_json()["type"] == "auth_success"
            ws.send_json({"type": "message", "text": "Who is my manager?"})
            error = ws.receive_json()
            assert error["type"] == "stream"
            assert "try again" in error["text"]
            assert "PRIVATE" not in error["text"]
            assert ws.receive_json() == {"type": "done"}
            ws.send_json({"type": "message", "text": "What is the vacation policy?"})
            assert ws.receive_json()["type"] == "stream"
            while ws.receive_json()["type"] != "done":
                pass
