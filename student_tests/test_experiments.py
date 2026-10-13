import pytest

from student.experiments import config_fingerprint, load_experiment_config, run_id_for


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

