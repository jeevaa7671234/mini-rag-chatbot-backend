"""Embedders share the same API, enabling offline tests without model downloads."""
import asyncio
import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol


class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Production option: CPU-friendly local semantic embedding model."""

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    @lru_cache(maxsize=2)
    def _load(name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name)

    @property
    def dimensions(self) -> int:
        return int(self._load(self.name).get_sentence_embedding_dimension())

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def _encode() -> list[list[float]]:
            array = self._load(self.name).encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return array.tolist()

        return await asyncio.to_thread(_encode)


class HashingEmbedder:
    """Deterministic *lexical* vectors for offline integration tests, NOT production RAG."""
    dimensions = 384

    @staticmethod
    def _one(text: str) -> list[float]:
        vector = [0.0] * 384
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            # Stable across processes, unlike built-in hash().
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % 384] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(lambda: [self._one(t) for t in texts])


def make_embedder(settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
