from datetime import date

import pytest
from pydantic import ValidationError

from hotel_rag.models import Chunk, Dept, DocMetadata, DocType


def test_dept_values_match_corpus_folders():
    assert [d.value for d in Dept] == [
        "rececao",
        "fnb",
        "copa",
        "estacionamento",
        "manutencao",
    ]


def test_metadata_rejects_unknown_dept():
    with pytest.raises(ValidationError):
        DocMetadata(
            doc_id="x",
            title="t",
            dept="spa",
            doc_type="sop",
            version="1.0",
            date=date(2026, 1, 1),
            source_path="a.md",
        )


def test_chunk_keeps_offsets_consistent():
    meta = DocMetadata(
        doc_id="x",
        title="t",
        dept=Dept.COPA,
        doc_type=DocType.SOP,
        version="1.0",
        date=date(2026, 1, 1),
        source_path="a.md",
    )
    c = Chunk(chunk_id="x:0", doc_id="x", text="abc", char_start=0, char_end=3, metadata=meta)
    assert c.char_end - c.char_start == len(c.text)
