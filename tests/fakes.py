import hashlib
from datetime import date

from hotel_rag.models import (
    Chunk,
    Dept,
    DocMetadata,
    DocType,
    RetrievedChunk,
)


class FakeEmbedder:
    """Deterministic 8-dimension embedder for tests (no model download)."""

    dimension = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[: self.dimension]]


def make_retrieved(doc_id: str, score: float = 1.0) -> RetrievedChunk:
    """Minimal RetrievedChunk for tests that only care about doc ids."""
    metadata = DocMetadata(
        doc_id=doc_id,
        title=doc_id,
        dept=Dept.COPA,
        doc_type=DocType.SOP,
        version="1.0",
        date=date(2026, 1, 1),
        source_path=f"{doc_id}.md",
    )
    chunk = Chunk(
        chunk_id=f"{doc_id}:0",
        doc_id=doc_id,
        text=f"texto de {doc_id}",
        char_start=0,
        char_end=len(f"texto de {doc_id}"),
        metadata=metadata,
    )
    return RetrievedChunk(chunk=chunk, score=score, source="test")
