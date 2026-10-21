import re
from typing import Protocol

from hotel_rag.models import Answer, Citation, RetrievedChunk

ABSTAIN_MARKER = "ABSTAIN"
_CITATION_RE = re.compile(r"\[(\d+)\]")


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{index}] {hit.chunk.metadata.title} "
        f"({hit.chunk.metadata.dept.value}, {hit.chunk.metadata.doc_id})\n"
        f"{hit.chunk.text}"
        for index, hit in enumerate(chunks, start=1)
    )

    return (
        "És um assistente de operações hoteleiras. Responde apenas com base no contexto "
        "fornecido, em português de Portugal e de forma concisa.\n"
        "Cita as fontes usadas com [n], onde n é o número do excerto.\n"
        f"Se o contexto não contiver a resposta, responde exatamente {ABSTAIN_MARKER}.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}\n"
        "Resposta:"
    )


def _is_abstain(text: str) -> bool:
    """True only when the whole reply is the marker (not a word inside a sentence)."""
    return text.strip().strip("*_`\"' .\n\t").upper() == ABSTAIN_MARKER


def parse_citations(text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[int] = set()

    for match in _CITATION_RE.finditer(text):
        index = int(match.group(1))
        if not 1 <= index <= len(chunks) or index in seen:
            continue
        seen.add(index)

        chunk = chunks[index - 1].chunk
        citations.append(
            Citation(
                doc_id=chunk.metadata.doc_id,
                title=chunk.metadata.title,
                dept=chunk.metadata.dept,
                page=chunk.page,
                # The excerpt is the whole chunk, so it matches the cited
                # [char_start, char_end) range exactly.
                snippet=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
        )

    return citations


class AnswerGenerator:
    """Grounded answer generation: cite retrieved chunks or abstain honestly."""

    def __init__(self, llm: LLMClient, min_score: float = 0.3) -> None:
        self.llm = llm
        self.min_score = min_score

    def _abstain(self) -> Answer:
        return Answer(text=ABSTAIN_MARKER, citations=[], abstained=True, confidence=0.0)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        if not chunks:
            return self._abstain()

        # Rank scores are not comparable across modes (RRF is ~0.03, cosine is
        # a similarity, the reranker is a probability). Only `relevance` is
        # calibrated, so it is the single input to the abstention decision.
        evidence = min(1.0, max(0.0, float(chunks[0].relevance)))
        if evidence < self.min_score:
            return self._abstain()

        text = self.llm.complete(build_prompt(question, chunks))
        if _is_abstain(text):
            return self._abstain()

        citations = parse_citations(text, chunks)
        if not citations:
            # An answer without citations is not grounded; treat it as a
            # failure instead of presenting it as supported.
            return self._abstain()

        return Answer(
            text=text,
            citations=citations,
            abstained=False,
            confidence=evidence,
        )
