from hotel_rag.models import Citation, Dept
from ui.logic import citation_label, citation_markdown


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
