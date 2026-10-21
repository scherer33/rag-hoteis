import re
from collections import defaultdict

from rank_bm25 import BM25Plus

from hotel_rag.models import Chunk, Dept, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Lexical retrieval over chunks, with optional department filtering."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [_tokenize(chunk.text) for chunk in chunks]
        # BM25Plus keeps positive IDF on small corpora, where Okapi's IDF can
        # collapse to zero (term present in half the documents).
        self._bm25 = BM25Plus(corpus) if corpus else None

    def search(self, query: str, k: int = 10, dept: Dept | None = None) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            (
                (float(score), chunk)
                for score, chunk in zip(scores, self.chunks, strict=True)
                if score > 0 and (dept is None or chunk.metadata.dept is dept)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )

        return [
            RetrievedChunk(chunk=chunk, score=score, source="bm25", relevance=0.0)
            for score, chunk in ranked[:k]
        ]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Fuse ranked lists with Reciprocal Rank Fusion (RRF).

    The fused `score` is the RRF rank score (tiny, not a probability). The
    calibrated `relevance` is carried over from the best contributing hit, so
    abstention never compares an RRF value against a probability threshold.
    """
    scores: dict[str, float] = defaultdict(float)
    relevance: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}

    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] += 1.0 / (k + rank)
            relevance[chunk_id] = max(relevance[chunk_id], hit.relevance)
            chunks.setdefault(chunk_id, hit.chunk)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        RetrievedChunk(
            chunk=chunks[chunk_id],
            score=score,
            source="hybrid",
            relevance=relevance[chunk_id],
        )
        for chunk_id, score in ordered
    ]
