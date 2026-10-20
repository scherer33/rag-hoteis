from collections.abc import Callable

from hotel_rag.generation import AnswerGenerator
from hotel_rag.models import Answer, Dept
from hotel_rag.retrieval import Retriever


class HotelRAG:
    """End-to-end question answering: rewrite, retrieve, generate with citations."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        rewriter: Callable[[str], str] | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.rewriter = rewriter

    def ask(self, question: str, top_k: int = 5, dept: Dept | None = None) -> Answer:
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
            dept=dept,
            rewrite=self.rewriter,
        )
        return self.generator.generate(question, chunks)
