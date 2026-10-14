import pytest

from app.embeddings import HashingEmbedder
from app.ingestion import Chunk
from app.security import User
from app.store import SqliteVectorStore


@pytest.mark.asyncio
async def test_role_access_filtered_at_sql_query_time(config):
    chunks = [
        Chunk("1", "vacation leave entitlement", "hr", 1, "leave_policy.pdf", 1),
        Chunk("2", "manager private review details", "hr", 2, "performance_review.pdf", 1),
        Chunk("3", "company receipts reimbursement", "finance", 1, "expense_policy.pdf", 1),
        Chunk("4", "confidential compensation private plan", "exec", 3, "compensation_committee.pdf", 1),
    ]
    embedding = HashingEmbedder()
    store = SqliteVectorStore(config.sqlite_path)
    await store.replace_all(chunks, await embedding.embed([c.text for c in chunks]), embedding.dimensions)
    query = (await embedding.embed(["private review compensation"]))[0]

    def make_user(dept, level):
        return User(sub="test", department=dept, level=level, iat=1, exp=2)

    hr_employee = await store.search(query, make_user("hr", 1), 20)
    assert {c.id for c in hr_employee} == {"1"}  # NOT top-K then filter
    hr_manager = await store.search(query, make_user("hr", 2), 20)
    assert {c.id for c in hr_manager} == {"1", "2"}
    finance_employee = await store.search(query, make_user("finance", 1), 20)
    assert {c.id for c in finance_employee} == {"1", "3"}
    executive = await store.search(query, make_user("exec", 3), 20)
    assert {c.id for c in executive} == {"1", "2", "4"}
    # Lower-limit query still gets an authorized result even when best overall was forbidden.
    assert {c.id for c in await store.search(query, make_user("hr", 1), 1)} == {"1"}


@pytest.mark.asyncio
async def test_mismatched_embedding_count_rejected(config):
    store = SqliteVectorStore(config.sqlite_path)
    with pytest.raises(ValueError):
        await store.replace_all([Chunk("x", "x", "hr", 1, "x.pdf", 1)], [], 384)
