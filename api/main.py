"""FastAPI service exposing the hotel RAG pipeline."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from hotel_rag.config import Settings
from hotel_rag.factory import build_default_pipeline
from hotel_rag.models import Answer, Dept
from hotel_rag.pipeline import HotelRAG


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
    return create_app(build_default_pipeline(Settings()))
