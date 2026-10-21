from hotel_rag.models import Citation


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
