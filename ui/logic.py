from html import escape

from hotel_rag.models import Citation

ABSTAIN_NOTICE = "Não encontrei informação suficiente na documentação para responder com segurança."


def answer_text(payload: dict) -> str:
    """Text to display and keep in history; never leak the internal ABSTAIN marker."""
    if payload.get("abstained", True):
        return ABSTAIN_NOTICE
    return payload.get("text", "")


def citations_from_payload(payload: list[dict]) -> list[Citation]:
    """Parse the JSON citations returned by /answer into domain objects."""
    return [Citation.model_validate(item) for item in payload]


def citation_label(citation: Citation, index: int = 1) -> str:
    """Short label for a citation, e.g. ``[1] AC erro E3 · manutencao · pág. 3``."""
    parts = [f"[{index}] {citation.title}", citation.dept.value]
    if citation.page is not None:
        parts.append(f"pág. {citation.page}")
    return " · ".join(parts)


def citation_markdown(citation: Citation) -> str:
    """Citation body: source document plus the quoted excerpt."""
    return f"**{citation.doc_id}**\n\n```text\n{citation.snippet}\n```"


def confidence_label(confidence: float) -> str:
    """Bucket the API confidence score into a PT-PT trust label."""
    if confidence >= 0.7:
        return "Alta"
    if confidence >= 0.45:
        return "Média"
    if confidence > 0:
        return "Baixa"
    return "Sem evidência"


def confidence_percent(confidence: float) -> int:
    """Clamp the confidence score to an integer percentage (0–100)."""
    return max(0, min(100, round(confidence * 100)))


def citation_card_html(citation: Citation, index: int = 1) -> str:
    """Render a citation as an editorial SOP card.

    All document-derived text is HTML-escaped and stripped of markdown
    markup: the corpus is untrusted input, and Streamlit still renders
    markdown inside HTML blocks (a ``##`` would become a giant heading).
    """
    page = f"pág. {citation.page}" if citation.page is not None else "sem nº de página"
    return (
        f'<div class="sop-card">'
        f'<div class="sop-head"><span class="sop-badge">[{index}]</span>'
        f'<span class="sop-title">{escape(citation.title)}</span></div>'
        f'<div class="sop-meta">{escape(citation.dept.value)} · '
        f"{escape(citation.doc_id)} · {page}</div>"
        f'<pre class="sop-excerpt">{escape(excerpt_text(citation.snippet))}</pre>'
        f"</div>"
    )


def excerpt_text(snippet: str, limit: int = 360) -> str:
    """Plain-text excerpt: drop markdown markers, collapse spacing, cap length."""
    lines = [line.lstrip("#").strip() for line in snippet.splitlines()]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip()
    return f"{cut}…"


def example_questions() -> list[str]:
    """Demo questions for the empty state; the last one abstains by design."""
    return [
        "A que horas é o check-out e quanto custa sair mais tarde?",
        "O que recebe um hóspede VIP A à chegada?",
        "A câmara dos lacticínios marca 7 graus. Está conforme?",
        "Quanto custa o quarto presidencial em Agosto de 2031?",
    ]
