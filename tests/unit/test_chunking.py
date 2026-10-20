from datetime import date

from hotel_rag.chunking import chunk_document
from hotel_rag.models import Dept, DocMetadata, DocType, Document


def _doc(text: str) -> Document:
    meta = DocMetadata(
        doc_id="d1",
        title="t",
        dept=Dept.RECECAO,
        doc_type=DocType.SOP,
        version="1.0",
        date=date(2026, 1, 1),
        source_path="a.md",
    )
    return Document(metadata=meta, text=text)


def test_short_document_yields_single_chunk():
    chunks = chunk_document(_doc("abc"), size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "abc"


def test_offsets_slice_back_to_original_text():
    text = "palavra " * 200
    chunks = chunk_document(_doc(text), size=100, overlap=20)
    assert all(text[c.char_start : c.char_end] == c.text for c in chunks)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)


def test_consecutive_chunks_overlap():
    text = "x" * 500
    chunks = chunk_document(_doc(text), size=100, overlap=20)
    assert chunks[1].char_start == chunks[0].char_end - 20


def test_no_empty_chunks():
    chunks = chunk_document(_doc("a\n\n" * 300), size=50, overlap=10)
    assert all(c.text.strip() for c in chunks)
