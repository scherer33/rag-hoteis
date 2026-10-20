from pathlib import Path

from hotel_rag.corpus import load_corpus, load_document
from hotel_rag.models import Dept

FM = """---
doc_id: checkin
title: Check-in padrão
dept: rececao
doc_type: sop
version: "1.0"
date: 2026-01-10
---

# Check-in
Texto.
"""


def _write(tmp_path: Path, name: str, fm: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(fm, encoding="utf-8")
    return p


def test_load_document_reads_frontmatter(tmp_path):
    doc = load_document(_write(tmp_path, "a.md", FM))
    assert doc.metadata.dept is Dept.RECECAO
    assert doc.metadata.property_id == "algarve-1"
    assert "Check-in" in doc.text


def test_load_corpus_keeps_only_latest_version(tmp_path):
    _write(tmp_path, "v1.md", FM)
    _write(tmp_path, "v2.md", FM.replace('version: "1.0"', 'version: "2.0"'))
    docs = load_corpus(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata.version == "2.0"


def test_load_corpus_can_include_all_versions(tmp_path):
    _write(tmp_path, "v1.md", FM)
    _write(tmp_path, "v2.md", FM.replace('version: "1.0"', 'version: "2.0"'))
    assert len(load_corpus(tmp_path, latest_only=False)) == 2
