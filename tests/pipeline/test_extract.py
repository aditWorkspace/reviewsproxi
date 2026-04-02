import json
from pathlib import Path
import pytest


def test_count_completed_batches_zero_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.extract.SIGNALS_PATH", tmp_path / "signals.jsonl")
    from pipeline.extract import _count_completed_batches
    assert _count_completed_batches() == 0


def test_count_completed_batches_counts_lines(tmp_path, monkeypatch):
    signals_path = tmp_path / "signals.jsonl"
    signals_path.write_text(
        json.dumps({"pain_points": []}) + "\n" +
        json.dumps({"pain_points": []}) + "\n"
    )
    monkeypatch.setattr("pipeline.extract.SIGNALS_PATH", signals_path)
    from pipeline.extract import _count_completed_batches
    assert _count_completed_batches() == 2


def test_build_batches():
    from pipeline.extract import _build_batches
    reviews = [{"text": f"review {i}"} for i in range(75)]
    batches = _build_batches(reviews, batch_size=30)
    assert len(batches) == 3
    assert len(batches[0]) == 30
    assert len(batches[2]) == 15
