from collections.abc import Callable
from typing import Literal, get_args

from hotel_rag.bm25 import BM25Index, reciprocal_rank_fusion
from hotel_rag.embeddings import Embedder
from hotel_rag.models import Dept, RetrievedChunk
from hotel_rag.rerank import Reranker
from hotel_rag.store import ChunkStore

RetrievalMode = Literal["vector", "hybrid", "hybrid_rerank"]
MODES: tuple[str, ...] = get_args(RetrievalMode)


class Retriever:
    """Retrieval strategies: dense vector, hybrid (RRF), and hybrid + rerank."""

    def __init__(
        self,
        embedder: Embedder,
        store: ChunkStore,
        bm25: BM25Index,
        reranker: Reranker | None = None,
        mode: RetrievalMode = "hybrid_rerank",
        k_candidates: int = 20,
    ) -> None:
        _validate_mode(mode)
        if mode == "hybrid_rerank" and reranker is None:
            raise ValueError("mode 'hybrid_rerank' requires a reranker")

        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.mode = mode
        self.k_candidates = k_candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dept: Dept | None = None,
        rewrite: Callable[[str], str] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[RetrievedChunk]:
        effective_mode = mode or self.mode
        _validate_mode(effective_mode)
        if effective_mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("mode 'hybrid_rerank' requires a reranker")

        if rewrite is not None:
            query = rewrite(query)

        vector = self.embedder.embed_query(query)
        vector_hits = self.store.search(vector, k=self.k_candidates, dept=dept)

        if effective_mode == "vector":
            return vector_hits[:top_k]

        bm25_hits = self.bm25.search(query, k=self.k_candidates, dept=dept)
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits])

        if effective_mode == "hybrid":
            return fused[:top_k]

        assert self.reranker is not None
        return self.reranker.rerank(query, fused, top_k)


def _validate_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode: {mode!r} (expected one of {sorted(MODES)})")
