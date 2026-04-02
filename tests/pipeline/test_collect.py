import json
from pathlib import Path
import pytest


def test_review_hash_is_deterministic():
    from pipeline.collect import _review_hash
    h1 = _review_hash("B001234567", "user42")
    h2 = _review_hash("B001234567", "user42")
    assert h1 == h2


def test_review_hash_differs_for_different_inputs():
    from pipeline.collect import _review_hash
    h1 = _review_hash("B001234567", "user42")
    h2 = _review_hash("B001234567", "user99")
    assert h1 != h2


def test_load_existing_hashes_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.collect.REVIEWS_PATH", tmp_path / "reviews.jsonl")
    from pipeline.collect import load_existing_hashes
    assert load_existing_hashes() == set()


def test_load_existing_hashes_reads_written_reviews(tmp_path, monkeypatch):
    reviews_path = tmp_path / "reviews.jsonl"
    reviews_path.write_text(
        json.dumps({"asin": "B001", "reviewer_id": "u1", "text": "good"}) + "\n" +
        json.dumps({"asin": "B002", "reviewer_id": "u2", "text": "bad"}) + "\n"
    )
    monkeypatch.setattr("pipeline.collect.REVIEWS_PATH", reviews_path)
    from pipeline.collect import load_existing_hashes, _review_hash
    hashes = load_existing_hashes()
    assert _review_hash("B001", "u1") in hashes
    assert _review_hash("B002", "u2") in hashes
    assert len(hashes) == 2
