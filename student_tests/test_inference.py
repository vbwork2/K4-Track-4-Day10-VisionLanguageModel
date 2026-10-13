import json

import pytest

from student.inference import read_existing_predictions, run_incremental_inference, score_predictions


def test_resume_skips_completed_predictions_and_rewrites_canonical_order(tmp_path, monkeypatch):
    output = tmp_path / "pred.jsonl"
    output.write_text('{"image_id":"b","raw":"old b"}\n', encoding="utf-8")
    rows = [{"image_id": "a"}, {"image_id": "b"}, {"image_id": "c"}]
    generated = []

    def fake_generate(record, **kwargs):
        generated.append(record["image_id"])
        return f"raw {record['image_id']}"

    monkeypatch.setattr("student.inference.generate_one", fake_generate)
    run_incremental_inference(rows, tmp_path, object(), object(), output)
    assert generated == ["a", "c"]
    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["image_id"] for row in saved] == ["a", "b", "c"]
    assert saved[1]["raw"] == "old b"


def test_invalid_partial_jsonl_line_is_ignored(tmp_path):
    path = tmp_path / "pred.jsonl"
    path.write_text('{"image_id":"a","raw":"ok"}\n{"broken"', encoding="utf-8")
    with pytest.warns(UserWarning, match="invalid/incomplete"):
        assert read_existing_predictions(path) == {"a": "ok"}


def test_scorer_refuses_incomplete_predictions_before_subprocess(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / "scripts").mkdir(parents=True)
    (data / "splits").mkdir(parents=True)
    (data / "splits" / "test.jsonl").write_text('{"image_id":"a"}\n', encoding="utf-8")
    pred = tmp_path / "pred.jsonl"
    pred.write_text("", encoding="utf-8")

    def should_not_run(*args, **kwargs):
        raise AssertionError("score.py must not run for incomplete predictions")

    monkeypatch.setattr("student.inference.subprocess.run", should_not_run)
    with pytest.raises(ValueError, match="incomplete predictions"):
        score_predictions(repo, data, pred, tmp_path / "evidence.json", "E0")

