from pathlib import Path

from hotel_rag.bm25 import BM25Index
from hotel_rag.chunking import chunk_document
from hotel_rag.config import Settings
from hotel_rag.corpus import load_corpus
from hotel_rag.embeddings import Embedder, SentenceTransformerEmbedder
from hotel_rag.generation import AnswerGenerator, LLMClient
from hotel_rag.pipeline import HotelRAG
from hotel_rag.rerank import CrossEncoderReranker, Reranker
from hotel_rag.retrieval import Retriever
from hotel_rag.store import ChunkStore


def build_pipeline(
    settings: Settings,
    corpus_dir: Path,
    embedder: Embedder,
    store: ChunkStore,
    llm: LLMClient,
    reranker: Reranker | None = None,
    mode: str = "hybrid_rerank",
    k_candidates: int = 20,
) -> HotelRAG:
    """Assemble the full pipeline, indexing the corpus when the store is empty.

    In-memory Qdrant (used by the Hugging Face Space and local demos) starts
    empty on every process, so the corpus is embedded once at startup.
    """
    documents = load_corpus(corpus_dir)
    chunks = [chunk for document in documents for chunk in chunk_document(document)]

    if store.count() == 0:
        store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))

    bm25 = BM25Index(chunks)
    effective_reranker = (
        reranker if reranker is not None else CrossEncoderReranker(settings.rerank_model)
    )

    retriever = Retriever(
        embedder,
        store,
        bm25,
        reranker=effective_reranker,
        mode=mode,
        k_candidates=k_candidates,
    )
    generator = AnswerGenerator(llm, min_score=settings.abstain_min_score)

    return HotelRAG(retriever, generator)


def build_default_pipeline(settings: Settings, corpus_dir: Path = Path("data/corpus")) -> HotelRAG:
    """Production wiring from environment settings."""
    from hotel_rag.llm import OpenAICompatClient

    embedder = SentenceTransformerEmbedder(settings.embed_model)
    store = ChunkStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_size=embedder.dimension,
    )
    llm = OpenAICompatClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    return build_pipeline(settings, corpus_dir, embedder, store, llm)
