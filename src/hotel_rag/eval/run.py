"""Compare retrieval configurations on the golden set.

Usage:
    python -m hotel_rag.eval.run --configs vector,hybrid,hybrid_rerank --out results/latest.json
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from hotel_rag.bm25 import BM25Index
from hotel_rag.chunking import chunk_document
from hotel_rag.config import Settings
from hotel_rag.corpus import load_corpus
from hotel_rag.embeddings import SentenceTransformerEmbedder
from hotel_rag.eval.golden import GoldenItem, load_golden
from hotel_rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from hotel_rag.rerank import CrossEncoderReranker
from hotel_rag.retrieval import Retriever
from hotel_rag.store import ChunkStore


def evaluate(retriever: Retriever, items: list[GoldenItem], k: int = 5) -> dict[str, float]:
    """Mean recall@k, MRR and nDCG@k over the golden set."""
    if not items:
        return {"recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0}

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for item in items:
        hits = retriever.retrieve(item.question, top_k=k)
        doc_ids = [hit.chunk.doc_id for hit in hits]
        relevant = set(item.relevant_doc_ids)
        recalls.append(recall_at_k(doc_ids, relevant, k))
        reciprocal_ranks.append(reciprocal_rank(doc_ids, relevant))
        ndcgs.append(ndcg_at_k(doc_ids, relevant, k))

    count = len(items)
    return {
        "recall@k": sum(recalls) / count,
        "mrr": sum(reciprocal_ranks) / count,
        "ndcg@k": sum(ndcgs) / count,
    }


def compare(
    configs: dict[str, Retriever], items: list[GoldenItem], k: int = 5
) -> dict[str, dict[str, float]]:
    return {name: evaluate(retriever, items, k) for name, retriever in configs.items()}


def to_markdown(results: dict[str, dict[str, float]]) -> str:
    lines = ["| config | recall@k | mrr | ndcg@k |", "|---|---|---|---|"]
    for name, scores in results.items():
        lines.append(
            f"| {name} | {scores['recall@k']:.3f} | {scores['mrr']:.3f} | {scores['ndcg@k']:.3f} |"
        )
    return "\n".join(lines)


def build_configs(
    settings: Settings,
    modes: list[str],
    corpus_dir: Path,
    k_candidates: int = 20,
) -> dict[str, Retriever]:
    embedder = SentenceTransformerEmbedder(settings.embed_model)
    store = ChunkStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_size=embedder.dimension,
    )

    documents = load_corpus(corpus_dir)
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    bm25 = BM25Index(chunks)
    reranker = CrossEncoderReranker(settings.rerank_model)

    return {
        mode: Retriever(
            embedder,
            store,
            bm25,
            reranker=reranker if mode == "hybrid_rerank" else None,
            mode=mode,
            k_candidates=k_candidates,
        )
        for mode in modes
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="vector,hybrid,hybrid_rerank")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/golden_set.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out", type=Path, default=Path("results/latest.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--k-candidates", type=int, default=20)
    args = parser.parse_args(argv)

    settings = Settings()
    modes = [mode.strip() for mode in args.configs.split(",") if mode.strip()]
    items = load_golden(args.golden)
    configs = build_configs(settings, modes, args.corpus, args.k_candidates)
    results = compare(configs, items, args.k)

    payload = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "k": args.k,
            "k_candidates": args.k_candidates,
            "questions": len(items),
            "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model,
        },
        "results": results,
        "markdown": to_markdown(results),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
