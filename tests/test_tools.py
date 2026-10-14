import time

import pytest

from app.chat import ChatService
from app.embeddings import HashingEmbedder
from app.ingestion import Chunk
from app.llm import MockLLM
from app.security import User
from app.store import SqliteVectorStore
from app.tools import get_employee_context


@pytest.mark.asyncio
async def test_three_tool_services_run_in_parallel():
    start = time.perf_counter()
    result = await get_employee_context("emp-001")
    elapsed = time.perf_counter() - start
    assert 0.9 <= elapsed < 2.1
    assert result == {
        "user_id": "emp-001", "name": "Jane Doe", "grade": "Senior",
        "manager": "John Smith", "team_size": 8, "team_name": "Platform",
    }


@pytest.mark.asyncio
async def test_model_cannot_request_someone_elses_profile(config):
    class MaliciousLLM(MockLLM):
        async def select_tool(self, messages):
            return {"id": "bad", "function": {"name": "get_employee_context", "arguments": '{"user_id":"other-user"}'}}

    store = SqliteVectorStore(config.sqlite_path)
    await store.replace_all([Chunk("id", "leave", "hr", 1, "leave_policy.pdf", 1)], [[1.0] * 384], 384)
    service = ChatService(HashingEmbedder(), store, MaliciousLLM(), 5)
    user = User(sub="emp-001", department="hr", level=1, iat=1, exp=2)
    with pytest.raises(PermissionError):
        async for _ in service.answer(user, "Who is my manager?"):
            pass
