from datetime import date

from fastapi.testclient import TestClient

from api.main import create_app
from hotel_rag.models import Answer, Citation, Dept, DocMetadata, DocType


def _source(doc_id: str = "a") -> DocMetadata:
    return DocMetadata(
        doc_id=doc_id,
        title="SOP",
        dept=Dept.RECECAO,
        doc_type=DocType.SOP,
        version="1.0",
        date=date(2026, 1, 1),
        source_path=f"{doc_id}.md",
    )


class StubRAG:
    def __init__(self) -> None:
        self.last_mode: str | None = None

    def ask(self, question, top_k=5, dept=None, mode=None):
        self.last_mode = mode
        return Answer(
            text="Faça X [1].",
            confidence=0.9,
            abstained=False,
            citations=[
                Citation(
                    doc_id="a",
                    title="SOP",
                    dept=Dept.RECECAO,
                    page=None,
                    snippet="Faça X.",
                    char_start=0,
                    char_end=7,
                )
            ],
        )

    def sources(self):
        return [_source()]


class FailingRAG:
    def ask(self, question, top_k=5, dept=None, mode=None):
        raise RuntimeError("LLM unreachable")

    def sources(self):
        return []


def test_health():
    client = TestClient(create_app(StubRAG()))

    assert client.get("/health").json() == {"status": "ok"}


def test_answer_returns_citations():
    client = TestClient(create_app(StubRAG()))

    response = client.post("/answer", json={"question": "como fazer X?"})

    assert response.status_code == 200
    body = response.json()
    assert body["citations"][0]["doc_id"] == "a"
    assert body["abstained"] is False


def test_answer_forwards_retrieval_mode():
    rag = StubRAG()
    client = TestClient(create_app(rag))

    client.post("/answer", json={"question": "q", "mode": "vector"})

    assert rag.last_mode == "vector"


def test_answer_rejects_unknown_mode():
    client = TestClient(create_app(StubRAG()))

    assert client.post("/answer", json={"question": "q", "mode": "magic"}).status_code == 422


def test_answer_rejects_empty_question():
    client = TestClient(create_app(StubRAG()))

    assert client.post("/answer", json={"question": ""}).status_code == 422


def test_answer_rejects_unknown_department():
    client = TestClient(create_app(StubRAG()))

    response = client.post("/answer", json={"question": "q", "dept": "spa"})

    assert response.status_code == 422


def test_sources_lists_indexed_documents():
    client = TestClient(create_app(StubRAG()))

    body = client.get("/sources").json()

    assert body[0]["doc_id"] == "a"
    assert body[0]["dept"] == "rececao"


def test_answer_returns_503_when_generation_fails():
    client = TestClient(create_app(FailingRAG()), raise_server_exceptions=False)

    response = client.post("/answer", json={"question": "q"})

    assert response.status_code == 503
