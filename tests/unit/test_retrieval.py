from datetime import date

import pytest

from hotel_rag.bm25 import BM25Index
from hotel_rag.models import Chunk, Dept, DocMetadata, DocType, RetrievedChunk
from hotel_rag.retrieval import Retriever


def _chunk(cid: str, text: str) -> Chunk:
    meta = DocMetadata(
        doc_id=cid.split(":")[0],
        title=cid,
        dept=Dept.COPA,
        doc_type=DocType.SOP,
        version="1.0",
        date=date(2026, 1, 1),
        source_path="a.md",
    )
    return Chunk(
        chunk_id=cid,
        doc_id=meta.doc_id,
        text=text,
        char_start=0,
        char_end=len(text),
        metadata=meta,
    )


class StubStore:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits

    def search(self, vector, k: int = 10, dept=None) -> list[RetrievedChunk]:
        return self.hits[:k]


class StubEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


class ReverseReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return list(reversed(chunks))[:top_k]


@pytest.mark.parametrize("mode", ["vector", "hybrid", "hybrid_rerank"])
def test_all_modes_return_top_k(mode):
    chunk = _chunk("a:0", "camara fria")
    retriever = Retriever(
        StubEmbedder(),
        StubStore([RetrievedChunk(chunk=chunk, score=1.0, source="vector")]),
        BM25Index([chunk]),
        reranker=ReverseReranker(),
        mode=mode,
    )
    assert len(retriever.retrieve("camara fria", top_k=1)) == 1


def test_rewrite_is_applied_before_retrieval():
    seen: dict[str, str] = {}

    class RecordingEmbedder(StubEmbedder):
        def embed_query(self, text: str) -> list[float]:
            seen["q"] = text
            return [0.0] * 8

    chunk = _chunk("a:0", "x")
    retriever = Retriever(
        RecordingEmbedder(),
        StubStore([RetrievedChunk(chunk=chunk, score=1.0, source="vector")]),
        BM25Index([chunk]),
        mode="vector",
    )

    retriever.retrieve("o AC ta pingando", rewrite=lambda q: "ar condicionado a pingar agua")

    assert seen["q"] == "ar condicionado a pingar agua"


def test_unknown_mode_is_rejected():
    chunk = _chunk("a:0", "x")
    with pytest.raises(ValueError):
        Retriever(
            StubEmbedder(),
            StubStore([RetrievedChunk(chunk=chunk, score=1.0, source="vector")]),
            BM25Index([chunk]),
            mode="magic",
        )


def test_hybrid_rerank_requires_a_reranker():
    chunk = _chunk("a:0", "x")
    with pytest.raises(ValueError):
        Retriever(
            StubEmbedder(),
            StubStore([RetrievedChunk(chunk=chunk, score=1.0, source="vector")]),
            BM25Index([chunk]),
            mode="hybrid_rerank",
        )
