from datetime import date

from hotel_rag.generation import AnswerGenerator
from hotel_rag.models import Chunk, Dept, DocMetadata, DocType, RetrievedChunk
from hotel_rag.pipeline import HotelRAG


class StubRetriever:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits
        self.last_query: str | None = None

    def retrieve(self, query, top_k=5, dept=None, rewrite=None):
        if rewrite:
            query = rewrite(query)
        self.last_query = query
        return self.hits


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return "Faça X [1]."


def _hit() -> RetrievedChunk:
    meta = DocMetadata(
        doc_id="a",
        title="SOP",
        dept=Dept.RECECAO,
        doc_type=DocType.SOP,
        version="1.0",
        date=date(2026, 1, 1),
        source_path="a.md",
    )
    chunk = Chunk(
        chunk_id="a:0",
        doc_id="a",
        text="Faça X.",
        char_start=0,
        char_end=7,
        metadata=meta,
    )
    return RetrievedChunk(chunk=chunk, score=0.9, source="hybrid")


def test_pipeline_returns_answer_with_citations():
    rag = HotelRAG(StubRetriever([_hit()]), AnswerGenerator(FakeLLM()))
    answer = rag.ask("como fazer X?")

    assert not answer.abstained
    assert answer.citations[0].doc_id == "a"


def test_pipeline_passes_rewriter_to_retriever():
    retriever = StubRetriever([_hit()])
    rag = HotelRAG(retriever, AnswerGenerator(FakeLLM()), rewriter=lambda q: q.upper())

    rag.ask("como fazer x?")

    assert retriever.last_query == "COMO FAZER X?"
