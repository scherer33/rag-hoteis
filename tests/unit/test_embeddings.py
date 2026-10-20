from tests.fakes import FakeEmbedder


def test_fake_embedder_is_deterministic():
    embedder = FakeEmbedder()
    assert embedder.embed_documents(["a"]) == embedder.embed_documents(["a"])
    assert embedder.embed_query("a") == embedder.embed_documents(["a"])[0]


def test_fake_embedder_vectors_have_fixed_dimension():
    embedder = FakeEmbedder()
    assert len(embedder.embed_query("texto")) == 8
    assert all(len(vector) == 8 for vector in embedder.embed_documents(["a", "b"]))


def test_fake_embedder_distinguishes_texts():
    embedder = FakeEmbedder()
    assert embedder.embed_query("camara fria") != embedder.embed_query("check-in")
