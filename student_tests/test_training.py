import json

import pytest

from student.training import (
    TrainingOptions,
    append_metric,
    last_logged_step,
    metric_summary,
    rebuild_train_log,
    sample_indices,
)


def test_step_sampling_is_deterministic_and_step_specific():
    first = sample_indices(1000, step=1, micro_step=0, batch_size=4, seed=42)
    assert first == sample_indices(1000, step=1, micro_step=0, batch_size=4, seed=42)
    assert first != sample_indices(1000, step=2, micro_step=0, batch_size=4, seed=42)


def test_metrics_are_fsynced_and_optimizer_steps_are_monotonic(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_metric(path, {"event": "train_step", "step": 1, "loss": 1.25}, expected_previous_step=0)
    append_metric(path, {"event": "train_step", "step": 2, "loss": 0.75}, expected_previous_step=1)
    assert last_logged_step(path) == 2
    with pytest.raises(ValueError, match="non-monotonic"):
        append_metric(path, {"event": "train_step", "step": 2, "loss": 0.5}, expected_previous_step=2)


def test_existing_duplicate_metric_steps_are_detected(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"event": "train_step", "step": 3}) + "\n"
        + json.dumps({"event": "train_step", "step": 3}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="increase strictly"):
        last_logged_step(path)


def test_metric_summary_uses_cumulative_elapsed_and_peak_values(tmp_path):
    path = tmp_path / "metrics.jsonl"
    events = [
        {"event": "train_step", "step": 1, "elapsed_seconds": 12, "peak_gpu_gb": 6.5},
        {"event": "train_step", "step": 2, "elapsed_seconds": 25, "peak_gpu_gb": 7.0},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    assert metric_summary(path) == {"global_step": 2, "peak_gpu_gb": 7.0, "elapsed_seconds": 25.0}


def test_rebuild_text_log_discards_steps_beyond_restored_metrics(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps({
            "event": "train_step",
            "step": 2,
            "loss": 0.75,
            "learning_rate": 1e-4,
            "grad_norm": 0.8,
            "peak_gpu_gb": 7.0,
        }) + "\n",
        encoding="utf-8",
    )
    log = tmp_path / "train.log"
    log.write_text("stale future step", encoding="utf-8")
    rebuild_train_log(metrics, log)
    assert log.read_text(encoding="utf-8").startswith("step=2 ")
    assert "stale" not in log.read_text(encoding="utf-8")


def test_kaggle_training_defaults_keep_effective_batch_at_eight():
    options = TrainingOptions(setup="E1")
    assert options.micro_batch_size * options.grad_accum == 8
    assert (options.max_steps, options.learning_rate, options.checkpoint_every) == (300, 2e-4, 25)

