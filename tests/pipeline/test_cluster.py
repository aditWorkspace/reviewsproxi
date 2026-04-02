import numpy as np
import pytest


def test_signals_to_text_combines_fields():
    from pipeline.cluster import _signals_to_text
    signals = {
        "pain_points": [{"signal": "breaks fast", "frequency": 3, "emotional_intensity": 0.8}],
        "desired_outcomes": [{"outcome": "lasts a year", "priority": 0.9}],
        "deal_breakers": ["no warranty"],
        "decision_factors_ranked": ["durability"],
    }
    text = _signals_to_text(signals)
    assert "breaks fast" in text
    assert "lasts a year" in text
    assert "no warranty" in text
    assert "durability" in text


def test_signals_to_text_empty_signals():
    from pipeline.cluster import _signals_to_text
    assert _signals_to_text({}) == ""


def test_score_clustering_returns_three_metrics():
    from pipeline.cluster import score_clustering
    rng = np.random.default_rng(42)
    embeddings = rng.random((20, 8))
    labels = np.array([0] * 10 + [1] * 10)
    metrics = score_clustering(embeddings, labels)
    assert set(metrics.keys()) == {"silhouette", "intra_similarity", "inter_distance"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_combined_score_uses_weights():
    from pipeline.cluster import combined_score
    metrics = {"silhouette": 1.0, "intra_similarity": 1.0, "inter_distance": 1.0}
    weights = {"silhouette": 0.5, "intra_similarity": 0.3, "inter_distance": 0.2}
    assert combined_score(metrics, weights) == pytest.approx(1.0)
