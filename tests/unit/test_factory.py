from pathlib import Path

from hotel_rag.config import Settings
from hotel_rag.factory import build_pipeline, create_llm_client
from hotel_rag.models import RetrievedChunk
from hotel_rag.store import ChunkStore
from tests.fakes import FakeEmbedder


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return "Resposta suportada [1]."


class PassThroughReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]


def _corpus(tmp_path: Path) -> Path:
    (tmp_path / "a.md").write_text(
        "---\ndoc_id: a\ntitle: SOP de teste\ndept: copa\ndoc_type: sop\n"
        "version: '1.0'\ndate: 2026-01-01\n---\n\n# SOP\n\nDesligar o equipamento.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_in_memory_pipeline_indexes_the_corpus(tmp_path):
    store = ChunkStore(vector_size=8)
    settings = Settings(qdrant_url=":memory:", abstain_min_score=0.0, llm_model="fake")

    pipeline = build_pipeline(
        settings,
        corpus_dir=_corpus(tmp_path),
        embedder=FakeEmbedder(),
        store=store,
        llm=FakeLLM(),
        reranker=PassThroughReranker(),
    )

    assert store.count() >= 1
    answer = pipeline.ask("o que fazer?")
    assert not answer.abstained
    assert answer.citations[0].doc_id == "a"


def test_existing_index_is_not_rebuilt(tmp_path):
    store = ChunkStore(vector_size=8)
    settings = Settings(qdrant_url=":memory:", abstain_min_score=0.0, llm_model="fake")
    corpus = _corpus(tmp_path)

    build_pipeline(
        settings,
        corpus_dir=corpus,
        embedder=FakeEmbedder(),
        store=store,
        llm=FakeLLM(),
        reranker=PassThroughReranker(),
    )
    count = store.count()

    build_pipeline(
        settings,
        corpus_dir=corpus,
        embedder=FakeEmbedder(),
        store=store,
        llm=FakeLLM(),
        reranker=PassThroughReranker(),
    )

    assert store.count() == count


def test_create_llm_client_passes_all_settings(monkeypatch):
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("hotel_rag.factory.OpenAICompatClient", Client)
    settings = Settings(
        _env_file=None,
        llm_model="model-x",
        llm_base_url="http://llm:1234/v1",
        llm_api_key="k",
        llm_temperature=0.2,
        llm_timeout_seconds=5,
    )

    create_llm_client(settings)

    assert captured == {
        "base_url": "http://llm:1234/v1",
        "api_key": "k",
        "model": "model-x",
        "temperature": 0.2,
        "timeout_seconds": 5,
    }
