from datetime import date

from hotel_rag.generation import AnswerGenerator
from hotel_rag.models import (
    Chunk,
    Dept,
    DocMetadata,
    DocType,
    Document,
    RetrievedChunk,
)
from hotel_rag.pipeline import HotelRAG


class StubRetriever:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits
        self.last_query: str | None = None
        self.last_mode: str | None = None

    def retrieve(self, query, top_k=5, dept=None, rewrite=None, mode=None):
        if rewrite:
            query = rewrite(query)
        self.last_query = query
        self.last_mode = mode
        return self.hits


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return "Faça X [1]."


def _metadata(doc_id: str = "a", title: str = "SOP") -> DocMetadata:
    return DocMetadata(
        doc_id=doc_id,
        title=title,
        dept=Dept.RECECAO,
        doc_type=DocType.SOP,
        version="1.0",
        date=date(2026, 1, 1),
        source_path=f"{doc_id}.md",
    )


def _hit() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="a:0",
        doc_id="a",
        text="Faça X.",
        char_start=0,
        char_end=7,
        metadata=_metadata(),
    )
    return RetrievedChunk(chunk=chunk, score=0.9, source="hybrid", relevance=0.9)


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


def test_pipeline_passes_retrieval_mode_to_retriever():
    retriever = StubRetriever([_hit()])
    rag = HotelRAG(retriever, AnswerGenerator(FakeLLM()))

    rag.ask("como fazer x?", mode="vector")

    assert retriever.last_mode == "vector"


def test_pipeline_exposes_unique_sources():
    v1 = Document(metadata=_metadata(title="SOP v1"), text="antigo")
    v2 = Document(metadata=_metadata(title="SOP v2"), text="novo")
    other = Document(metadata=_metadata(doc_id="b", title="Manual"), text="outro")
    rag = HotelRAG(
        StubRetriever([]),
        AnswerGenerator(FakeLLM()),
        documents=[v1, v2, other],
    )

    sources = rag.sources()

    assert len(sources) == 2
    assert {source.doc_id for source in sources} == {"a", "b"}
    assert next(source for source in sources if source.doc_id == "a").title == "SOP v2"
