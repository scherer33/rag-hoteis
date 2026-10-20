from hotel_rag.rerank import sigmoid


def test_sigmoid_normalizes_logits_to_probabilities():
    assert sigmoid(0.0) == 0.5
    assert 0.0 < sigmoid(-5.0) < 0.5 < sigmoid(5.0) < 1.0
