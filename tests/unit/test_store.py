from datetime import date

from hotel_rag.models import Chunk, Dept, DocMetadata, DocType
from hotel_rag.store import ChunkStore


def _chunk(i: int, dept: Dept) -> Chunk:
    meta = DocMetadata(
        doc_id=f"d{i}",
        title=f"t{i}",
        dept=dept,
        doc_type=DocType.SOP,
        version="1.0",
        date=date(2026, 1, 1),
        source_path="a.md",
    )
    return Chunk(
        chunk_id=f"d{i}:0",
        doc_id=f"d{i}",
        text=f"texto {i}",
        char_start=0,
        char_end=8,
        metadata=meta,
    )


def _vec(i: int) -> list[float]:
    vector = [0.0] * 8
    vector[i % 8] = 1.0
    return vector


def test_upsert_and_count():
    store = ChunkStore(vector_size=8)
    store.upsert([_chunk(0, Dept.COPA)], [_vec(0)])
    assert store.count() == 1


def test_search_returns_most_similar_first():
    store = ChunkStore(vector_size=8)
    chunks = [_chunk(0, Dept.COPA), _chunk(1, Dept.RECECAO)]
    store.upsert(chunks, [_vec(0), _vec(1)])

    hits = store.search(_vec(0), k=2)
    assert hits[0].chunk.doc_id == "d0"
    assert hits[0].source == "vector"


def test_search_filters_by_department():
    store = ChunkStore(vector_size=8)
    store.upsert([_chunk(0, Dept.COPA), _chunk(1, Dept.RECECAO)], [_vec(0), _vec(1)])

    hits = store.search(_vec(1), k=5, dept=Dept.COPA)
    assert [hit.chunk.metadata.dept for hit in hits] == [Dept.COPA]
