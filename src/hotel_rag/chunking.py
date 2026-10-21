from hotel_rag.models import Chunk, Document

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 120


def chunk_document(
    doc: Document, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Split a document into overlapping character windows with exact offsets.

    Boundaries prefer the last paragraph break or sentence end in the second
    half of the window, so chunks stay readable.
    """
    text = doc.text
    if len(text) <= size:
        return [
            Chunk(
                chunk_id=f"{doc.metadata.doc_id}:0",
                doc_id=doc.metadata.doc_id,
                text=text,
                char_start=0,
                char_end=len(text),
                metadata=doc.metadata,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            window = text[start:end]
            paragraph = window.rfind("\n\n")
            sentence = window.rfind(". ")
            cut = max(paragraph, sentence)
            if cut > size // 2:
                end = start + cut + (2 if cut == paragraph else 1)

        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.metadata.doc_id}:{start}",
                    doc_id=doc.metadata.doc_id,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                    metadata=doc.metadata,
                )
            )

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks
