import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from hotel_rag.models import Dept


class GoldenItem(BaseModel):
    question: str
    answer: str
    relevant_doc_ids: list[str]
    language: Literal["pt", "en"]
    dept: Dept


def load_golden(path: Path) -> list[GoldenItem]:
    """Load a JSONL golden set (one question per line)."""
    items: list[GoldenItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(GoldenItem(**json.loads(line)))
    return items
