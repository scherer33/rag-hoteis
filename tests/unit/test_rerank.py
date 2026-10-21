from datetime import date

from hotel_rag.models import Chunk, Dept, DocMetadata, DocType, RetrievedChunk
from hotel_rag.rerank import CrossEncoderReranker, sigmoid


def test_sigmoid_normalizes_logits_to_probabilities():
    assert sigmoid(0.0) == 0.5
    assert 0.0 < sigmoid(-5.0) < 0.5 < sigmoid(5.0) < 1.0


def _hit(cid: str, text: str) -> RetrievedChunk:
    meta = DocMetadata(
        doc_id=cid.split(":")[0],
        title=cid,
        dept=Dept.COPA,
        doc_type=DocType.SOP,
        version="1.0",
        date=date(2026, 1, 1),
        source_path="a.md",
    )
    chunk = Chunk(
        chunk_id=cid,
        doc_id=meta.doc_id,
        text=text,
        char_start=0,
        char_end=len(text),
        metadata=meta,
    )
    return RetrievedChunk(chunk=chunk, score=0.5, source="hybrid", relevance=0.5)


class _StubModel:
    def predict(self, pairs):
        return [3.0, -3.0][: len(pairs)]


def test_rerank_uses_sigmoid_as_score_and_relevance():
    reranker = CrossEncoderReranker("fake")
    reranker._model = _StubModel()

    ranked = reranker.rerank("q", [_hit("a:0", "primeiro"), _hit("b:0", "segundo")], top_k=2)

    assert ranked[0].chunk.chunk_id == "a:0"
    assert ranked[0].score == sigmoid(3.0)
    assert ranked[0].relevance == ranked[0].score
    assert ranked[1].score == sigmoid(-3.0)
    assert ranked[1].relevance == ranked[1].score
    assert ranked[0].source == "rerank"
