import json

import pytest

from student.experiments import (
    config_fingerprint,
    compare_rank_evidence,
    load_experiment_config,
    run_id_for,
    run_rank_ablation,
)


def test_experiment_configs_match_setup_files():
    config = load_experiment_config(".", "E1")
    assert config["name"] == "E1"
    assert config["lora"]["r"] == 16


def test_e4_ids_separate_both_rank_values_and_baseline():
    assert run_id_for("E4", "E1", 4) == "e4_rank_4_base_e1"
    assert run_id_for("E4", "E1", 64) == "e4_rank_64_base_e1"
    with pytest.raises(ValueError):
        run_id_for("E4", "E1", 16)


def test_fingerprint_is_stable_and_tracks_single_variable_change():
    first = {"setup": "E1", "rank": 4, "seed": 42}
    same = {"seed": 42, "rank": 4, "setup": "E1"}
    changed_rank = {"setup": "E1", "rank": 64, "seed": 42}
    assert config_fingerprint(first) == config_fingerprint(same)
    assert config_fingerprint(first) != config_fingerprint(changed_rank)


def test_rank_ablation_runs_both_values_with_identical_controls(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return {"run_id": run_id_for("E4", kwargs["e4_base"], kwargs["e4_rank"]), "bundle": object()}

    monkeypatch.setattr("student.experiments.run_experiment", fake_run_experiment)
    results = run_rank_ablation(
        repo_root=tmp_path,
        data_dir=tmp_path,
        run_root=tmp_path,
        base_setup="E1",
        micro_batch_size=1,
        grad_accum=8,
        max_steps=300,
        checkpoint_every=25,
        remote_repo_id="owner/private-checkpoints",
        token="test-token",
    )
    assert [call["e4_rank"] for call in calls] == [4, 64]
    assert {call["micro_batch_size"] for call in calls} == {1}
    assert {call["grad_accum"] for call in calls} == {8}
    assert {call["max_steps"] for call in calls} == {300}
    assert [result["run_id"] for result in results] == [
        "e4_rank_4_base_e1",
        "e4_rank_64_base_e1",
    ]


def test_rank_comparison_requires_complete_same_dataset_evidence(tmp_path):
    def write(path, rank, f1):
        path.write_text(json.dumps({
            "split": "test",
            "dataset_version": "d10-testfixture",
            "missing_predictions": 0,
            "micro": {"f1": f1},
            "f1_ci95": [f1 - 0.1, f1 + 0.1],
            "meta": {"trainable_params": rank * 100, "peak_vram_gb": 5.0, "train_minutes": 20.0},
        }), encoding="utf-8")

    rank4 = tmp_path / "rank4.json"
    rank64 = tmp_path / "rank64.json"
    write(rank4, 4, 0.2)
    write(rank64, 64, 0.3)
    result = compare_rank_evidence(rank4, rank64)
    assert result["rank_4"]["f1"] == 0.2
    assert result["rank_64"]["trainable_params"] == 6400
    rank64.write_text(rank64.read_text().replace("d10-testfixture", "d10-other"), encoding="utf-8")
    with pytest.raises(ValueError, match="different dataset versions"):
        compare_rank_evidence(rank4, rank64)

