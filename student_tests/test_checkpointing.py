import json

import pytest

from student.checkpointing import (
    CheckpointError,
    CheckpointMismatchError,
    RemoteBackupError,
    apply_training_state,
    atomic_write_json,
    build_manifest,
    local_checkpoints,
    require_remote_credentials,
    restore_metrics,
    validate_checkpoint,
)


def _checkpoint(root, setup="E1", step=5):
    (root / "adapter").mkdir(parents=True)
    files = {
        "adapter/adapter_config.json": b"{}",
        "adapter/adapter_model.safetensors": b"adapter state",
        "trainer_state.pt": b"trainer state",
        "metrics.jsonl": b'{"step":5}\n',
        "run.json": b"{}",
    }
    if setup in ("E2", "E3"):
        files["projector.pt"] = b"projector"
    for name, content in files.items():
        (root / name).write_bytes(content)
    manifest = build_manifest(root, step, "cfg-a", "data-a", setup, 123)
    atomic_write_json(root / "manifest.json", manifest)
    return root


def test_valid_checkpoint_and_corrupt_file_detection(tmp_path):
    checkpoint = _checkpoint(tmp_path / "step_000005")
    assert validate_checkpoint(checkpoint, "cfg-a", "data-a", "E1")["global_step"] == 5
    (checkpoint / "metrics.jsonl").write_text("corrupt", encoding="utf-8")
    assert validate_checkpoint(checkpoint) is None


def test_incomplete_checkpoint_is_not_resumable(tmp_path):
    checkpoint = tmp_path / "step_000010"
    checkpoint.mkdir()
    assert validate_checkpoint(checkpoint) is None


def test_dataset_setup_and_fingerprint_mismatch_fail_closed(tmp_path):
    checkpoint = _checkpoint(tmp_path / "step_000005", setup="E2")
    with pytest.raises(CheckpointMismatchError, match="dataset_version mismatch"):
        validate_checkpoint(checkpoint, dataset_version="other")
    with pytest.raises(CheckpointMismatchError, match="setup mismatch"):
        validate_checkpoint(checkpoint, setup="E1")
    with pytest.raises(CheckpointMismatchError, match="config_fingerprint mismatch"):
        validate_checkpoint(checkpoint, config_fingerprint="other")


def test_projector_checkpoint_requires_independent_projector_state(tmp_path):
    checkpoint = tmp_path / "step_000005"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer")
    (checkpoint / "metrics.jsonl").write_text("", encoding="utf-8")
    (checkpoint / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CheckpointError, match="projector.pt"):
        build_manifest(checkpoint, 5, "cfg-a", "data-a", "E2", 123)


def test_training_state_restores_all_components_and_returns_step():
    class Holder:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, value):
            self.loaded = value

    optimizer, scheduler, scaler = Holder(), Holder(), Holder()
    restored_rng = []
    state = {
        "global_step": 17,
        "elapsed_seconds": 123.5,
        "optimizer": {"m": 1},
        "scheduler": {"step": 17},
        "scaler": {"scale": 128},
        "rng": {"torch_cpu": "state"},
    }
    step, elapsed = apply_training_state(
        state, optimizer, scheduler, scaler, restore_rng=restored_rng.append
    )
    assert (step, elapsed) == (17, 123.5)
    assert optimizer.loaded == {"m": 1}
    assert scheduler.loaded == {"step": 17}
    assert scaler.loaded == {"scale": 128}
    assert restored_rng == [state["rng"]]


def test_resume_requires_remote_credentials_before_training():
    with pytest.raises(RemoteBackupError, match="HF_TOKEN and HF_REPO_ID"):
        require_remote_credentials(None, None, required=True)
    require_remote_credentials(None, None, required=False)


def test_restore_metrics_discards_uncheckpointed_tail(tmp_path):
    checkpoint = _checkpoint(tmp_path / "step_000005")
    active = tmp_path / "metrics.jsonl"
    active.write_text('{"step":5}\n{"step":6}\n', encoding="utf-8")
    restore_metrics(checkpoint, active)
    assert active.read_text(encoding="utf-8") == '{"step":5}\n'


def test_local_checkpoint_listing_ignores_temporary_directory(tmp_path):
    root = tmp_path / "run"
    (root / "checkpoints").mkdir(parents=True)
    _checkpoint(root / "checkpoints" / "step_000005", step=5)
    _checkpoint(root / "checkpoints" / "step_000025", step=25)
    (root / "checkpoints" / "step_000050.tmp").mkdir()
    assert [path.name for path in local_checkpoints(root)] == ["step_000005", "step_000025"]

