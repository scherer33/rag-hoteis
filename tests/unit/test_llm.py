import pytest

from hotel_rag.llm import OpenAICompatClient


def test_client_requires_a_configured_model():
    with pytest.raises(ValueError):
        OpenAICompatClient(base_url="http://localhost:20128/v1", api_key="", model="")
