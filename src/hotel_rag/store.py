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
    ) -> None:
        self.collection = collection
        self.vector_size = vector_size
        self.client = QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

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
            hits.append(
                RetrievedChunk(
                    chunk=Chunk(**payload),
                    score=float(point.score),
                    source="vector",
                )
            )
        return hits

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection).count)
