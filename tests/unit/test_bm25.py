from datetime import date

from hotel_rag.bm25 import BM25Index, reciprocal_rank_fusion
from hotel_rag.models import Chunk, Dept, DocMetadata, DocType, RetrievedChunk


def _chunk(cid: str, text: str, dept: Dept = Dept.COPA) -> Chunk:
    meta = DocMetadata(
        doc_id=cid.split(":")[0],
        title=cid,
        dept=dept,
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


def test_bm25_finds_exact_keyword():
    index = BM25Index(
        [
            _chunk("a:0", "temperatura da camara fria do peixe"),
            _chunk("b:0", "procedimento de check-in do hospede"),
        ]
    )
    hits = index.search("camara fria", k=1)
    assert hits[0].chunk.chunk_id == "a:0"


def test_bm25_respects_department_filter():
    index = BM25Index(
        [
            _chunk("a:0", "camara fria", Dept.COPA),
            _chunk("b:0", "camara fria", Dept.RECECAO),
        ]
    )
    hits = index.search("camara fria", k=5, dept=Dept.RECECAO)
    assert [hit.chunk.metadata.dept for hit in hits] == [Dept.RECECAO]


def test_rrf_merges_two_rankings():
    c1 = _chunk("a:0", "x")
    c2 = _chunk("b:0", "y")
    first = [RetrievedChunk(chunk=c1, score=1.0, source="vector")]
    second = [
        RetrievedChunk(chunk=c2, score=1.0, source="bm25"),
        RetrievedChunk(chunk=c1, score=0.5, source="bm25"),
    ]

    fused = reciprocal_rank_fusion([first, second])

    assert [hit.chunk.chunk_id for hit in fused] == ["a:0", "b:0"]
    assert fused[0].source == "hybrid"
