"""Build the Qdrant index from the markdown corpus.

Usage:
    python scripts/build_index.py --corpus data/corpus --qdrant-url http://localhost:6333
"""

import argparse
from pathlib import Path

from hotel_rag.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_document
from hotel_rag.corpus import load_corpus
from hotel_rag.embeddings import Embedder, SentenceTransformerEmbedder
from hotel_rag.store import ChunkStore


def build_index(
    corpus_dir: Path,
    store: ChunkStore,
    embedder: Embedder,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> int:
    documents = load_corpus(corpus_dir)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, size=chunk_size, overlap=overlap)
    ]
    if not chunks:
        return 0

    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    store.upsert(chunks, vectors)
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="hotel_docs")
    parser.add_argument("--embed-model", default="BAAI/bge-m3")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--append",
        action="store_true",
        help="keep existing points instead of recreating the collection first",
    )
    args = parser.parse_args(argv)

    if args.qdrant_url == ":memory:":
        parser.error("--qdrant-url :memory: loses the index when the process exits")

    embedder = SentenceTransformerEmbedder(args.embed_model)
    store = ChunkStore(
        url=args.qdrant_url,
        collection=args.collection,
        vector_size=embedder.dimension,
    )
    if not args.append:
        store.recreate()
    indexed = build_index(args.corpus, store, embedder, args.chunk_size, args.overlap)
    print(f"indexed {indexed} chunks from {args.corpus} into {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
