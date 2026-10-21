"""FastAPI service exposing the hotel RAG pipeline."""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hotel_rag.config import Settings
from hotel_rag.factory import build_default_pipeline
from hotel_rag.models import Answer, Dept, DocMetadata
from hotel_rag.pipeline import HotelRAG
from hotel_rag.retrieval import RetrievalMode

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    dept: Dept | None = None
    mode: RetrievalMode | None = None


def create_app(rag: HotelRAG) -> FastAPI:
    app = FastAPI(title="Hotel RAG Algarve", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/sources")
    def sources() -> list[DocMetadata]:
        return rag.sources()

    @app.post("/answer", response_model=Answer)
    def answer(request: AskRequest) -> Answer:
        try:
            return rag.ask(
                request.question,
                top_k=request.top_k,
                dept=request.dept,
                mode=request.mode,
            )
        except Exception as error:
            logger.exception("answer generation failed")
            raise HTTPException(
                status_code=503,
                detail="answer generation is temporarily unavailable",
            ) from error

    return app


def create_default_app() -> FastAPI:
    """Build the production pipeline from environment configuration."""
    return create_app(build_default_pipeline(Settings()))
