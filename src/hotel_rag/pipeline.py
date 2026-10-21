from collections.abc import Callable, Sequence

from hotel_rag.generation import AnswerGenerator
from hotel_rag.models import Answer, Dept, DocMetadata, Document
from hotel_rag.retrieval import RetrievalMode, Retriever


class HotelRAG:
    """End-to-end question answering: rewrite, retrieve, generate with citations."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        rewriter: Callable[[str], str] | None = None,
        documents: Sequence[Document] = (),
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.rewriter = rewriter
        self.documents = list(documents)

    def sources(self) -> list[DocMetadata]:
        """Latest metadata per document, for the /sources endpoint."""
        latest = {document.metadata.doc_id: document.metadata for document in self.documents}
        return sorted(latest.values(), key=lambda metadata: (metadata.dept.value, metadata.doc_id))

    def ask(
        self,
        question: str,
        top_k: int = 5,
        dept: Dept | None = None,
        mode: RetrievalMode | None = None,
    ) -> Answer:
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
            dept=dept,
            rewrite=self.rewriter,
            mode=mode,
        )
        return self.generator.generate(question, chunks)
