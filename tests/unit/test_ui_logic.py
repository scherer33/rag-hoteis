from hotel_rag.models import Citation, Dept
from ui.logic import (
    ABSTAIN_NOTICE,
    answer_text,
    citation_card_html,
    citation_label,
    citation_markdown,
    citations_from_payload,
    confidence_label,
    confidence_percent,
    example_questions,
    excerpt_text,
)


def test_label_includes_dept_and_page():
    citation = Citation(
        doc_id="a",
        title="AC erro E3",
        dept=Dept.MANUTENCAO,
        page=3,
        snippet="x",
        char_start=0,
        char_end=1,
    )

    assert citation_label(citation) == "[1] AC erro E3 · manutencao · pág. 3"


def test_label_omits_page_when_none():
    citation = Citation(
        doc_id="a",
        title="SOP",
        dept=Dept.COPA,
        page=None,
        snippet="x",
        char_start=0,
        char_end=1,
    )

    assert "pág." not in citation_label(citation)


def test_label_uses_given_index():
    citation = Citation(
        doc_id="a",
        title="SOP",
        dept=Dept.COPA,
        page=None,
        snippet="x",
        char_start=0,
        char_end=1,
    )

    assert citation_label(citation, index=2).startswith("[2] ")


def test_markdown_contains_snippet():
    citation = Citation(
        doc_id="a",
        title="SOP",
        dept=Dept.COPA,
        page=None,
        snippet="texto citado",
        char_start=0,
        char_end=12,
    )

    assert "texto citado" in citation_markdown(citation)


def test_citations_from_api_json_payload():
    payload = [
        {
            "doc_id": "a",
            "title": "AC erro E3",
            "dept": "manutencao",
            "page": 3,
            "snippet": "x",
            "char_start": 0,
            "char_end": 1,
        }
    ]

    citations = citations_from_payload(payload)

    assert isinstance(citations[0], Citation)
    assert citation_label(citations[0]) == "[1] AC erro E3 · manutencao · pág. 3"
    assert citation_markdown(citations[0]).startswith("**a**")


def test_answer_text_returns_grounded_answer():
    payload = {"text": "O check-out é às 11h00 [1].", "abstained": False}

    assert answer_text(payload) == "O check-out é às 11h00 [1]."


def test_answer_text_hides_the_abstain_marker():
    payload = {"text": "ABSTAIN", "abstained": True}

    assert answer_text(payload) == ABSTAIN_NOTICE
    assert "ABSTAIN" not in ABSTAIN_NOTICE


def test_confidence_label_buckets():
    assert confidence_label(0.85) == "Alta"
    assert confidence_label(0.7) == "Alta"
    assert confidence_label(0.6) == "Média"
    assert confidence_label(0.45) == "Média"
    assert confidence_label(0.2) == "Baixa"
    assert confidence_label(0.0) == "Sem evidência"


def test_confidence_percent_clamps_and_rounds():
    assert confidence_percent(0.875) == 88
    assert confidence_percent(1.2) == 100
    assert confidence_percent(-0.1) == 0
    assert confidence_percent(0.0) == 0


def test_citation_card_escapes_snippet_html():
    citation = Citation(
        doc_id="a",
        title="SOP",
        dept=Dept.COPA,
        page=None,
        snippet='<script>alert("x")</script>',
        char_start=0,
        char_end=30,
    )

    html = citation_card_html(citation, index=1)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_citation_card_includes_title_and_page():
    citation = Citation(
        doc_id="a",
        title="AC erro E3",
        dept=Dept.MANUTENCAO,
        page=3,
        snippet="trecho",
        char_start=0,
        char_end=6,
    )

    html = citation_card_html(citation, index=1)

    assert "AC erro E3" in html
    assert "pág. 3" in html


def test_example_questions_are_non_empty_and_include_abstainer():
    questions = example_questions()

    assert len(questions) >= 3
    assert all(isinstance(q, str) and q.strip() for q in questions)


def test_excerpt_text_removes_heading_hashes():
    snippet = "# Check-out e saídas tardias\n\n## Objetivo\n\nEncerrar a estadia."

    text = excerpt_text(snippet)

    assert "## Objetivo" not in text
    assert "# Check-out" not in text
    assert "Objetivo" in text
    assert "Encerrar a estadia." in text


def test_excerpt_text_truncates_long_snippets():
    snippet = "palavra " * 300

    text = excerpt_text(snippet, limit=100)

    assert text.endswith("…")
    assert len(text) <= 101


def test_excerpt_text_keeps_short_snippets_intact():
    assert excerpt_text("Trecho curto.") == "Trecho curto."


def test_citation_card_renders_plain_text_excerpt():
    citation = Citation(
        doc_id="a",
        title="SOP",
        dept=Dept.COPA,
        page=None,
        snippet="## Objetivo\n\nEncerrar a estadia.",
        char_start=0,
        char_end=30,
    )

    html = citation_card_html(citation, index=1)

    assert "## Objetivo" not in html
