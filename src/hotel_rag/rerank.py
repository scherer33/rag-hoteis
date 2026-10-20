from collections.abc import Sequence
from typing import Protocol

from hotel_rag.models import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    """Cross-encoder reranker; weights are only loaded on first use."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, hit.chunk.text) for hit in chunks]
        scores: Sequence[float] = self._load().predict(pairs)
        ranked = sorted(
            zip(chunks, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True
        )

        return [
            RetrievedChunk(chunk=hit.chunk, score=float(score), source="rerank")
            for hit, score in ranked[:top_k]
        ]
