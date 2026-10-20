from pathlib import Path

import frontmatter

from hotel_rag.models import DocMetadata, Document


def _version_key(version: str) -> tuple[int, ...]:
    """Comparable numeric key: "1.10" sorts after "1.2"."""
    return tuple(int(part) if part.isdigit() else 0 for part in version.split("."))


def load_document(path: Path) -> Document:
    post = frontmatter.load(path)
    metadata = DocMetadata(**post.metadata, source_path=str(path))
    return Document(metadata=metadata, text=post.content)


def load_corpus(root: Path, latest_only: bool = True) -> list[Document]:
    """Load every markdown document under root, newest version per doc_id by default."""
    documents = [load_document(path) for path in sorted(root.rglob("*.md"))]
    if not latest_only:
        return documents

    latest: dict[str, Document] = {}
    for document in documents:
        current = latest.get(document.metadata.doc_id)
        if current is None or _version_key(document.metadata.version) > _version_key(
            current.metadata.version
        ):
            latest[document.metadata.doc_id] = document

    return sorted(latest.values(), key=lambda document: document.metadata.doc_id)
