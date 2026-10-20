"""FastAPI service exposing the hotel RAG pipeline."""

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from hotel_rag.bm25 import BM25Index
from hotel_rag.chunking import chunk_document
from hotel_rag.config import Settings
from hotel_rag.corpus import load_corpus
from hotel_rag.embeddings import SentenceTransformerEmbedder
from hotel_rag.generation import AnswerGenerator
from hotel_rag.llm import OpenAICompatClient
from hotel_rag.models import Answer, Dept
from hotel_rag.pipeline import HotelRAG
from hotel_rag.rerank import CrossEncoderReranker
from hotel_rag.retrieval import Retriever
from hotel_rag.store import ChunkStore


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    dept: Dept | None = None


def create_app(rag: HotelRAG) -> FastAPI:
    app = FastAPI(title="Hotel RAG Algarve", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/answer", response_model=Answer)
    def answer(request: AskRequest) -> Answer:
        return rag.ask(request.question, top_k=request.top_k, dept=request.dept)

    return app


def create_default_app() -> FastAPI:
    """Build the production pipeline from environment configuration."""
    settings = Settings()

    embedder = SentenceTransformerEmbedder(settings.embed_model)
    store = ChunkStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_size=embedder.dimension,
    )

    documents = load_corpus(Path("data/corpus"))
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    bm25 = BM25Index(chunks)
    reranker = CrossEncoderReranker(settings.rerank_model)

    retriever = Retriever(
        embedder,
        store,
        bm25,
        reranker=reranker,
        mode="hybrid_rerank",
        k_candidates=20,
    )
    llm = OpenAICompatClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    generator = AnswerGenerator(llm, min_score=settings.abstain_min_score)

    return create_app(HotelRAG(retriever, generator))
