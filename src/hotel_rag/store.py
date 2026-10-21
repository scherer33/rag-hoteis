from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from hotel_rag.models import Chunk, Dept, RetrievedChunk

DEFAULT_COLLECTION = "hotel_docs"


class ChunkStore:
    """Qdrant-backed chunk store; uses in-memory mode when url is ":memory:"."""

    def __init__(
        self,
        url: str = ":memory:",
        collection: str = DEFAULT_COLLECTION,
        vector_size: int = 1024,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection = collection
        self.vector_size = vector_size
        self.client = client or (
            QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url)
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self._check_vector_size()
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def _check_vector_size(self) -> None:
        """Fail clearly when the existing collection was built with another embedder."""
        vectors = self.client.get_collection(self.collection).config.params.vectors
        size = getattr(vectors, "size", None)
        if size is not None and int(size) != self.vector_size:
            raise ValueError(
                f"collection {self.collection!r} has {size}-dim vectors but the embedder "
                f"produces {self.vector_size}; recreate the index (make index) or set EMBED_MODEL"
            )

    def recreate(self) -> None:
        """Drop and recreate the collection so removed documents leave no stale points."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._ensure_collection()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={**chunk.model_dump(mode="json"), "dept": chunk.metadata.dept.value},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self, vector: list[float], k: int = 10, dept: Dept | None = None
    ) -> list[RetrievedChunk]:
        query_filter = None
        if dept is not None:
            query_filter = Filter(
                must=[FieldCondition(key="dept", match=MatchValue(value=dept.value))]
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )

        hits: list[RetrievedChunk] = []
        for point in response.points:
            payload = dict(point.payload or {})
            payload.pop("dept", None)
            score = float(point.score)
            hits.append(
                RetrievedChunk(
                    chunk=Chunk(**payload),
                    score=score,
                    source="vector",
                    relevance=score,
                )
            )
        return hits

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection).count)
