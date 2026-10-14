"""Vector search with server-side / SQL-layer RBAC, never Python result filtering."""
import asyncio
import json
import math
import sqlite3
from pathlib import Path
from typing import Protocol

from app.ingestion import Chunk
from app.security import User


class VectorStore(Protocol):
    async def search(self, vector: list[float], user: User, limit: int) -> list[Chunk]: ...

    async def replace_all(self, chunks: list[Chunk], vectors: list[list[float]], dims: int) -> None: ...


class QdrantVectorStore:
    """Persistent local Qdrant; no Qdrant server or Docker needed."""

    def __init__(self, path: Path, collection: str):
        from qdrant_client import QdrantClient

        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.collection = collection

    async def replace_all(self, chunks: list[Chunk], vectors: list[list[float]], dims: int) -> None:
        from qdrant_client import models

        if len(chunks) != len(vectors):
            raise ValueError("Number of vectors must match chunks")

        def _replace() -> None:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE),
            )
            for start in range(0, len(chunks), 64):
                self.client.upsert(
                    collection_name=self.collection,
                    wait=True,
                    points=[
                        models.PointStruct(id=c.id, vector=vector, payload=c.payload())
                        for c, vector in zip(chunks[start : start + 64], vectors[start : start + 64])
                    ],
                )

        await asyncio.to_thread(_replace)

    async def search(self, vector: list[float], user: User, limit: int) -> list[Chunk]:
        from qdrant_client import models

        # A mandatory AND over role threshold and an OR over user's department / public HR.
        # The query engine restricts candidates BEFORE similarity ranking / limit.
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="access_level", range=models.Range(lte=user.level)),
                models.Filter(
                    should=[
                        models.FieldCondition(key="department", match=models.MatchValue(value=user.department)),
                        models.FieldCondition(key="department", match=models.MatchValue(value="hr")),
                    ]
                ),
            ]
        )

        def _query() -> list[Chunk]:
            if not self.client.collection_exists(self.collection):
                raise RuntimeError("No vector index found. Run: python -m app.ingestion")
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return [Chunk(id=str(p.id), **p.payload) for p in response.points]

        return await asyncio.to_thread(_query)


class SqliteVectorStore:
    """Portable SQL vector backend for offline tests/demo. No ANN index; not for production scale."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.create_function("cosine", 2, self._cosine, deterministic=True)
        return connection

    @staticmethod
    def _cosine(v_json: str, q_json: str) -> float:
        v, q = json.loads(v_json), json.loads(q_json)
        if len(v) != len(q):
            return 0.0
        n1 = math.sqrt(sum(a * a for a in v))
        n2 = math.sqrt(sum(b * b for b in q))
        return sum(a * b for a, b in zip(v, q)) / (n1 * n2) if n1 and n2 else 0.0

    async def replace_all(self, chunks: list[Chunk], vectors: list[list[float]], dims: int) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Number of vectors must match chunks")

        def _replace() -> None:
            with self._connect() as db:
                db.execute("""CREATE TABLE IF NOT EXISTS chunks
                    (id TEXT PRIMARY KEY, text TEXT NOT NULL, department TEXT NOT NULL,
                     access_level INTEGER NOT NULL, source_file TEXT NOT NULL,
                     page INTEGER NOT NULL, embedding TEXT NOT NULL)""")
                db.execute("DELETE FROM chunks")
                db.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (c.id, c.text, c.department, c.access_level, c.source_file, c.page,
                         json.dumps(v))
                        for c, v in zip(chunks, vectors)
                    ],
                )

        await asyncio.to_thread(_replace)

    async def search(self, vector: list[float], user: User, limit: int) -> list[Chunk]:
        def _query() -> list[Chunk]:
            with self._connect() as db:
                try:
                    # ACL is in SQL WHERE, before ORDER BY/LIMIT; not filtered in Python.
                    rows = db.execute(
                        """SELECT id, text, department, access_level, source_file, page
                        FROM chunks
                        WHERE (department = ? OR department = 'hr') AND access_level <= ?
                        ORDER BY cosine(embedding, ?) DESC LIMIT ?""",
                        (user.department, user.level, json.dumps(vector), limit),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    raise RuntimeError("No vector index found. Run: python -m app.ingestion") from exc
                return [Chunk(*row) for row in rows]

        return await asyncio.to_thread(_query)


def make_store(settings) -> VectorStore:
    if settings.vector_store == "sqlite":
        return SqliteVectorStore(settings.sqlite_path)
    return QdrantVectorStore(settings.qdrant_path, settings.collection_name)
