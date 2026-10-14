"""No cloud keys, vector server, or model downloads needed for the automated tests."""
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from app.chat import ChatService
from app.config import Settings
from app.embeddings import HashingEmbedder
from app.llm import MockLLM
from app.main import create_app
from app.store import SqliteVectorStore


@pytest.fixture
def config(tmp_path):
    return Settings(
        _env_file=None,
        jwt_secret="test-secret-strong-enough-for-fixtures",
        llm_provider="mock", embedding_provider="hashing", vector_store="sqlite",
        sqlite_path=tmp_path / "test_vectors.sqlite3", docs_dir=tmp_path / "docs",
    )


@pytest.fixture
def token_factory(config):
    def mint(sub="emp-001", department="hr", level=1, expired=False):
        now = datetime.now(timezone.utc)
        data = {
            "sub": sub, "department": department, "level": level,
            "iat": now - timedelta(hours=3) if expired else now,
            "exp": now - timedelta(seconds=30) if expired else now + timedelta(hours=1),
        }
        return jwt.encode(data, config.jwt_secret.get_secret_value(), algorithm="HS256")
    return mint


@pytest.fixture
def client(config):
    service = ChatService(HashingEmbedder(), SqliteVectorStore(config.sqlite_path), MockLLM(), 5)
    with TestClient(create_app(settings=config, service=service)) as result:
        yield result
