from fastapi.testclient import TestClient

from api.main import create_app
from hotel_rag.models import Answer, Citation, Dept


class StubRAG:
    def ask(self, question, top_k=5, dept=None):
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


def test_answer_rejects_empty_question():
    client = TestClient(create_app(StubRAG()))

    assert client.post("/answer", json={"question": ""}).status_code == 422


def test_answer_rejects_unknown_department():
    client = TestClient(create_app(StubRAG()))

    response = client.post("/answer", json={"question": "q", "dept": "spa"})

    assert response.status_code == 422
