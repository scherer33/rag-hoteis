from pathlib import Path

import pytest

from hotel_rag.bm25 import BM25Index
from hotel_rag.chunking import chunk_document
from hotel_rag.corpus import load_corpus
from hotel_rag.generation import AnswerGenerator
from hotel_rag.pipeline import HotelRAG
from hotel_rag.retrieval import Retriever
from hotel_rag.store import ChunkStore
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


class CitingLLM:
    """Deterministic stand-in for the real LLM: always cites the first chunk."""

    def complete(self, prompt: str) -> str:
        assert "[1]" in prompt
        return "Consultar o procedimento indicado [1]."


def test_pipeline_end_to_end_in_memory():
    root = Path(__file__).parents[2]
    documents = load_corpus(root / "data" / "corpus")
    chunks = [chunk for document in documents for chunk in chunk_document(document)]

    embedder = FakeEmbedder()
    store = ChunkStore(vector_size=embedder.dimension)
    store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))

    retriever = Retriever(
        embedder,
        store,
        BM25Index(chunks),
        mode="hybrid",
        k_candidates=10,
    )
    rag = HotelRAG(retriever, AnswerGenerator(CitingLLM(), min_score=0.0))

    answer = rag.ask("Qual é a temperatura da câmara de peixe?")

    assert not answer.abstained
    assert answer.citations
    assert answer.citations[0].doc_id in {document.metadata.doc_id for document in documents}
