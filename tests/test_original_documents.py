"""Contract tests against the employer-supplied seven PDFs (not fabricated fixtures)."""
import asyncio
from pathlib import Path

import pytest

from app.chat import ChatService
from app.embeddings import HashingEmbedder
from app.ingestion import FILE_ACCESS_LEVELS, collect_chunks, pages_from_pdf
from app.security import User
from app.store import SqliteVectorStore

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def user(department: str, level: int) -> User:
    return User(sub="test-001", department=department, level=level, iat=1, exp=2)


def test_real_corpus_is_complete_and_classified():
    actual = {p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.pdf")}
    assert actual == set(FILE_ACCESS_LEVELS)
    chunks = collect_chunks(DOCS)
    assert len(chunks) >= 14
    assert all(c.text and c.page == 1 for c in chunks)
    for path, level in FILE_ACCESS_LEVELS.items():
        selected = [c for c in chunks if c.source_file == Path(path).name]
        assert selected, path
        assert all(c.access_level == level and c.department == path.split('/')[0] for c in selected)
        assert all(len(c.text) <= 900 for c in selected)


def test_real_employer_policy_facts_extracted():
    texts = {p.name: "\n".join(t for _, t in pages_from_pdf(p)) for p in DOCS.rglob("*.pdf")}
    assert "22 working days" in texts["leave_policy.pdf"]
    assert "10 days of paid sick leave" in texts["leave_policy.pdf"]
    assert "five-point scale" in texts["performance_review.pdf"]
    assert "USD 500" in texts["expense_policy.pdf"]
    assert "Grade 5" in texts["travel_reimbursement.pdf"]
    assert "60th percentile" in texts["compensation_committee.pdf"]
    assert "Strategic Plan 2026" in texts["strategic_plan.pdf"]
    assert "APAC" in texts["strategic_plan.pdf"]
    assert "45 million" in texts["strategic_plan.pdf"]
    assert "AI-first" in texts["strategic_plan.pdf"], "3x OCR should read AI-first"


@pytest.mark.asyncio
async def test_real_policy_index_retrieval_acl_with_sql_layer(tmp_path):
    chunks = await asyncio.to_thread(collect_chunks, DOCS)
    embedder = HashingEmbedder()
    store = SqliteVectorStore(tmp_path / "original_policy_test.sqlite")
    vectors = await embedder.embed([c.text for c in chunks])
    await store.replace_all(chunks, vectors, embedder.dimensions)

    async def files_for(department, level, question, k=20):
        v = (await embedder.embed([question]))[0]
        return {c.source_file for c in await store.search(v, user(department, level), k)}

    emp = await files_for("hr", 1, "executive pay and manager calibration")
    assert emp == {"leave_policy.pdf", "code_of_conduct.pdf"}
    mgr = await files_for("hr", 2, "manager rating merit compensation")
    assert mgr == {"leave_policy.pdf", "code_of_conduct.pdf", "performance_review.pdf"}
    executive = await files_for("exec", 3, "board strategic plan and performance review")
    assert executive == {"leave_policy.pdf", "code_of_conduct.pdf", "performance_review.pdf", "strategic_plan.pdf", "compensation_committee.pdf"}
    finance = await files_for("finance", 1, "travel receipts leave")
    assert finance == {"leave_policy.pdf", "code_of_conduct.pdf", "expense_policy.pdf", "travel_reimbursement.pdf"}
    # Even when restricted chunks are highly similar, top-1 is searched inside permitted material.
    assert (await files_for("hr", 1, "board strategic plan compensation", 1)).issubset(emp)


@pytest.mark.asyncio
async def test_only_authorised_original_policy_text_reaches_llm(tmp_path):
    chunks = await asyncio.to_thread(collect_chunks, DOCS)
    emb = HashingEmbedder()
    store = SqliteVectorStore(tmp_path / "policy_context.sqlite")
    await store.replace_all(chunks, await emb.embed([c.text for c in chunks]), emb.dimensions)

    class RecordingLLM:
        def __init__(self):
            self.prompt = ""

        async def select_tool(self, messages):
            self.prompt = messages[-1]["content"]
            return None

        async def stream(self, messages):
            yield "Test token"
            yield " two"

    llm = RecordingLLM()
    service = ChatService(emb, store, llm, top_k=30)
    received = [v async for v in service.answer(user("hr", 1), "What does the leave policy say?")]
    assert received == ["Test token", " two"]
    assert "22 working days" in llm.prompt
    assert "60th percentile" not in llm.prompt
    assert "merit budget is 4%" not in llm.prompt
    assert "subscription revenue" not in llm.prompt
    assert "[Source: leave_policy.pdf p.1;" in llm.prompt
