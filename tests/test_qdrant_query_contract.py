"""Validate Qdrant query construction without requiring external Qdrant downloads."""
from dataclasses import dataclass
from types import SimpleNamespace
import sys
import types

import pytest

from app.security import User
from app.store import QdrantVectorStore


@dataclass
class FakeFilter:
    must: list | None = None
    should: list | None = None


@dataclass
class FakeRange:
    lte: int


@dataclass
class FakeMatch:
    value: str


@dataclass
class FakeField:
    key: str
    range: FakeRange | None = None
    match: FakeMatch | None = None


@pytest.mark.asyncio
async def test_qdrant_receives_filter_inside_query_points(monkeypatch):
    fake_lib = types.ModuleType("qdrant_client")
    fake_lib.models = SimpleNamespace(
        Filter=FakeFilter, Range=FakeRange, MatchValue=FakeMatch, FieldCondition=FakeField
    )
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_lib)

    class FakeClient:
        call = None

        def collection_exists(self, collection):
            return True

        def query_points(self, **kwargs):
            self.call = kwargs
            return SimpleNamespace(points=[SimpleNamespace(
                id="3c10614b-5b86-4827-9acf-d4987005663e",
                payload={"text": "approved leave", "department": "hr", "access_level": 1,
                         "source_file": "leave_policy.pdf", "page": 1},
            )])

    store = object.__new__(QdrantVectorStore)
    store.client = FakeClient()
    store.collection = "company_policies"
    user = User(sub="emp-001", department="finance", level=1, iat=1, exp=2)
    result = await store.search([0.3] * 384, user, 5)
    assert len(result) == 1
    assert result[0].source_file == "leave_policy.pdf"
    assert store.client.call["limit"] == 5
    f = store.client.call["query_filter"]
    assert isinstance(f, FakeFilter)
    assert len(f.must) == 2
    assert f.must[0].key == "access_level" and f.must[0].range.lte == 1
    assert isinstance(f.must[1], FakeFilter)
    assert {v.match.value for v in f.must[1].should} == {"finance", "hr"}
    assert all(v.key == "department" for v in f.must[1].should)
    assert store.client.call["with_payload"] is True
