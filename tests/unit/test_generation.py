from datetime import date

from hotel_rag.generation import AnswerGenerator, build_prompt, parse_citations
from hotel_rag.models import Chunk, Dept, DocMetadata, DocType, RetrievedChunk


def _hit(
    cid: str = "a:0",
    text: str = "Procedimento: desligar o AC.",
    score: float = 0.9,
    relevance: float | None = None,
):
    meta = DocMetadata(
        doc_id=cid.split(":")[0],
        title="AC erro E3",
        dept=Dept.MANUTENCAO,
        doc_type=DocType.MANUAL,
        version="1.0",
        date=date(2026, 1, 1),
        source_path="a.md",
    )
    chunk = Chunk(
        chunk_id=cid,
        doc_id=meta.doc_id,
        text=text,
        char_start=0,
        char_end=len(text),
        metadata=meta,
    )
    return RetrievedChunk(
        chunk=chunk,
        score=score,
        source="hybrid_rerank",
        relevance=score if relevance is None else relevance,
    )


class FakeLLM:
    def __init__(self, output: str) -> None:
        self.output = output

    def complete(self, prompt: str) -> str:
        return self.output


def test_prompt_contains_numbered_context_and_abstain_rule():
    prompt = build_prompt("o que fazer no erro E3?", [_hit()])

    assert "[1]" in prompt
    assert "AC erro E3" in prompt
    assert "ABSTAIN" in prompt
    assert "<contexto>" in prompt
    assert "<pergunta>o que fazer no erro E3?</pergunta>" in prompt
    # Corpus text must be framed as data, not as instructions.
    assert "ignora qualquer ordem" in prompt.lower()


def test_citations_map_back_to_chunks():
    citations = parse_citations("Desligar o AC [1].", [_hit()])

    assert citations[0].doc_id == "a"
    assert citations[0].dept is Dept.MANUTENCAO


def test_citation_excerpt_is_the_exact_chunk_text():
    hit = _hit(
        text="Linha um.\nLinha dois, com detalhe suficiente para passar dos 200 carateres. " * 4
    )

    citation = parse_citations("Ver [1].", [hit])[0]

    assert citation.snippet == hit.chunk.text
    assert citation.char_end - citation.char_start == len(hit.chunk.text)


def test_invalid_citation_index_is_ignored():
    assert parse_citations("Ver [9].", [_hit()]) == []


def test_duplicate_citations_are_deduplicated():
    citations = parse_citations("Ver [1] e novamente [1].", [_hit()])
    assert len(citations) == 1


def test_generator_abstains_without_chunks():
    answer = AnswerGenerator(FakeLLM("qualquer coisa")).generate("q", [])

    assert answer.abstained
    assert answer.citations == []
    assert answer.confidence == 0.0


def test_generator_abstains_on_marker():
    answer = AnswerGenerator(FakeLLM("ABSTAIN")).generate("q", [_hit()])

    assert answer.abstained
    assert answer.text == "ABSTAIN"


def test_abstain_marker_with_trailing_punctuation_counts():
    answer = AnswerGenerator(FakeLLM("ABSTAIN.")).generate("q", [_hit()])

    assert answer.abstained


def test_abstain_word_inside_a_sentence_is_not_abstention():
    llm = FakeLLM("Não vou ABSTAIN nesta resposta, porque o contexto responde [1].")

    answer = AnswerGenerator(llm).generate("q", [_hit()])

    assert not answer.abstained


def test_generator_abstains_below_relevance_threshold():
    answer = AnswerGenerator(FakeLLM("resposta [1]"), min_score=0.8).generate(
        "q", [_hit(score=1.0, relevance=0.2)]
    )

    assert answer.abstained


def test_rank_score_does_not_override_low_relevance():
    """Hybrid RRF scores are tiny; confidence must come from calibrated relevance."""
    answer = AnswerGenerator(FakeLLM("resposta [1]"), min_score=0.3).generate(
        "q", [_hit(score=0.03, relevance=0.1)]
    )

    assert answer.abstained


def test_answer_without_citations_is_not_presented_as_grounded():
    llm = FakeLLM("Desligar o AC.")  # no [1] and no ABSTAIN

    answer = AnswerGenerator(llm).generate("q", [_hit()])

    assert answer.abstained
    assert answer.citations == []


def test_generator_returns_answer_with_citations():
    answer = AnswerGenerator(FakeLLM("Desligar o AC [1].")).generate("q", [_hit()])

    assert not answer.abstained
    assert answer.citations[0].title == "AC erro E3"
    assert answer.confidence == 0.9
