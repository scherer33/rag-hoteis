import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from build_index import build_index  # noqa: E402

from hotel_rag.store import ChunkStore  # noqa: E402
from tests.fakes import FakeEmbedder  # noqa: E402


def test_build_index_populates_store(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ndoc_id: a\ntitle: A\ndept: copa\ndoc_type: sop\nversion: '1.0'\n"
        "date: 2026-01-01\n---\n\nTexto de teste.\n",
        encoding="utf-8",
    )
    store = ChunkStore(vector_size=8)

    indexed = build_index(tmp_path, store, FakeEmbedder())

    assert indexed >= 1
    assert store.count() >= 1


def test_build_index_returns_zero_for_empty_corpus(tmp_path):
    store = ChunkStore(vector_size=8)
    assert build_index(tmp_path, store, FakeEmbedder()) == 0
