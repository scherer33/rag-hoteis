from hotel_rag.embeddings import SentenceTransformerEmbedder
from tests.fakes import FakeEmbedder


class _NewApiModel:
    def get_embedding_dimension(self) -> int:
        return 1024


class _LegacyApiModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 768


def test_dimension_uses_current_sentence_transformers_api():
    embedder = SentenceTransformerEmbedder("fake")
    embedder._model = _NewApiModel()

    assert embedder.dimension == 1024


def test_dimension_falls_back_to_legacy_api():
    embedder = SentenceTransformerEmbedder("fake")
    embedder._model = _LegacyApiModel()

    assert embedder.dimension == 768


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
