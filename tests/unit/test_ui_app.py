import ui.app


def test_ui_module_exposes_main():
    assert callable(ui.app.main)


def test_ask_posts_to_api(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "ok", "citations": [], "abstained": False, "confidence": 0.9}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        return FakeResponse()

    monkeypatch.setattr(ui.app.requests, "post", fake_post)

    body = ui.app.ask("a que horas é o check-out?", top_k=3, dept="rececao", mode="vector")

    assert body["text"] == "ok"
    assert calls["json"] == {
        "question": "a que horas é o check-out?",
        "top_k": 3,
        "dept": "rececao",
        "mode": "vector",
    }
    assert calls["url"].endswith("/answer")
