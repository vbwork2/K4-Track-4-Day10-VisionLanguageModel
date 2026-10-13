"""Atomic training checkpoints and private Hugging Face recovery support."""

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time


class CheckpointError(RuntimeError):
    """Raised when a training checkpoint cannot be safely resumed."""


class CheckpointMismatchError(CheckpointError):
    """Raised when a checkpoint belongs to another setup or dataset."""


class RemoteBackupError(CheckpointError):
    """Raised when required durable remote backup did not complete."""


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_files(root):
    root = Path(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp")
    )


def build_manifest(checkpoint_dir, global_step, config_fingerprint, dataset_version, setup,
                   trainable_params, extra=None):
    root = Path(checkpoint_dir)
    integrity = {
        path.relative_to(root).as_posix(): {"size": path.stat().st_size, "sha256": _sha256(path)}
        for path in _checkpoint_files(root)
    }
    required = {
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "trainer_state.pt",
        "metrics.jsonl",
        "run.json",
    }
    if setup in ("E2", "E3"):
        required.add("projector.pt")
    missing = required - set(integrity)
    if missing:
        raise CheckpointError(f"Checkpoint is missing required files: {sorted(missing)}")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "global_step": int(global_step),
        "config_fingerprint": config_fingerprint,
        "dataset_version": dataset_version,
        "setup": setup,
        "trainable_params": int(trainable_params),
        "file_integrity": integrity,
    }
    if extra:
        payload.update(extra)
    return payload


def validate_checkpoint(checkpoint_dir, config_fingerprint=None, dataset_version=None, setup=None):
    """Return complete manifest or None for incomplete/corrupt checkpoint contents."""
    root = Path(checkpoint_dir)
    manifest_path = root / "manifest.json"
    try:
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or int(manifest.get("global_step", -1)) < 0:
            return None
        for expected, key in (
            (config_fingerprint, "config_fingerprint"),
            (dataset_version, "dataset_version"),
            (setup, "setup"),
        ):
            if expected is not None and manifest.get(key) != expected:
                raise CheckpointMismatchError(
                    f"Checkpoint {key} mismatch: found {manifest.get(key)!r}, expected {expected!r}"
                )
        integrity = manifest.get("file_integrity")
        if not isinstance(integrity, dict) or not integrity:
            return None
        for relative, attributes in integrity.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return None
            file_path = root / relative_path
            if not file_path.is_file():
                return None
            if file_path.stat().st_size != attributes.get("size"):
                return None
            if _sha256(file_path) != attributes.get("sha256"):
                return None
        if "adapter/adapter_config.json" not in integrity or "adapter/adapter_model.safetensors" not in integrity:
            return None
        if not {"trainer_state.pt", "metrics.jsonl", "run.json"}.issubset(integrity):
            return None
        if manifest.get("setup") in ("E2", "E3") and "projector.pt" not in integrity:
            return None
        return manifest
    except CheckpointMismatchError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _remove_owned_path(path, parent):
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    if parent not in path.parents:
        raise CheckpointError(f"Refusing to remove a path outside its run directory: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def capture_rng_state():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def apply_training_state(state, optimizer, scheduler, scaler, restore_rng=restore_rng_state):
    """Restore optimizer, LR scheduler, gradient scaler, step, elapsed time, and RNG."""
    required = {"global_step", "optimizer", "scheduler", "scaler", "rng", "elapsed_seconds"}
    missing = required - set(state)
    if missing:
        raise CheckpointError(f"Training state is missing fields: {sorted(missing)}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return int(state["global_step"]), float(state["elapsed_seconds"])


def restore_metrics(checkpoint_dir, active_metrics_path):
    """Replace an interrupted run log with the last fully committed checkpoint log."""
    source = Path(checkpoint_dir) / "metrics.jsonl"
    target = Path(active_metrics_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def save_checkpoint(
    run_dir,
    global_step,
    model,
    optimizer,
    scheduler,
    scaler,
    setup,
    config_fingerprint,
    dataset_version,
    trainable_params,
    elapsed_seconds,
    metrics_path,
    run_manifest,
    projector_modules=None,
    remote_repo_id=None,
    token=None,
    remote_required=True,
    checkpoint_every=None,
    max_retries=3,
):
    """Write, validate, atomically publish, and optionally upload a complete checkpoint."""
    import torch

    run_dir = Path(run_dir).resolve()
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final_dir = checkpoint_root / f"step_{int(global_step):06d}"
    temporary_dir = checkpoint_root / f"step_{int(global_step):06d}.tmp"
    if final_dir.exists():
        current = validate_checkpoint(final_dir, config_fingerprint, dataset_version, setup)
        if current and int(current["global_step"]) == int(global_step):
            return current
        raise CheckpointError(f"Checkpoint target already exists and is not reusable: {final_dir}")
    if temporary_dir.exists():
        _remove_owned_path(temporary_dir, checkpoint_root)
    temporary_dir.mkdir()
    (temporary_dir / "adapter").mkdir()
    model.save_pretrained(temporary_dir / "adapter", safe_serialization=True)
    if setup in ("E2", "E3"):
        if not projector_modules:
            raise CheckpointError(f"{setup} requires a separate full projector state")
        from student.model import projector_state_dict

        torch.save(projector_state_dict(projector_modules), temporary_dir / "projector.pt")
    state = {
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(),
        "elapsed_seconds": float(elapsed_seconds),
    }
    torch.save(state, temporary_dir / "trainer_state.pt")
    if metrics_path and Path(metrics_path).is_file():
        shutil.copyfile(metrics_path, temporary_dir / "metrics.jsonl")
    else:
        (temporary_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    atomic_write_json(temporary_dir / "run.json", run_manifest)
    manifest = build_manifest(
        temporary_dir,
        global_step=global_step,
        config_fingerprint=config_fingerprint,
        dataset_version=dataset_version,
        setup=setup,
        trainable_params=trainable_params,
        extra={"elapsed_seconds": float(elapsed_seconds), "checkpoint_every": checkpoint_every},
    )
    atomic_write_json(temporary_dir / "manifest.json", manifest)
    if validate_checkpoint(temporary_dir, config_fingerprint, dataset_version, setup) is None:
        raise CheckpointError("New checkpoint failed integrity verification")
    os.replace(temporary_dir, final_dir)
    if remote_repo_id or token:
        upload_checkpoint(
            final_dir,
            remote_repo_id=remote_repo_id,
            token=token,
            run_id=run_dir.name,
            required=remote_required,
            max_retries=max_retries,
        )
    elif remote_required:
        raise RemoteBackupError("HF_TOKEN and HF_REPO_ID are required before training can continue")
    return manifest


def require_remote_credentials(token, repo_id, required=True):
    if required and (not token or not repo_id):
        raise RemoteBackupError("Set Kaggle Secrets HF_TOKEN and HF_REPO_ID before training")


def ensure_private_repo(repo_id, token):
    require_remote_credentials(token, repo_id, required=True)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not getattr(info, "private", False):
        raise RemoteBackupError("Checkpoint repository must be private")
    return api


def upload_checkpoint(checkpoint_dir, remote_repo_id, token, run_id, required=True, max_retries=3):
    require_remote_credentials(token, remote_repo_id, required=required)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    checkpoint_dir = Path(checkpoint_dir)
    path_in_repo = f"{run_id}/checkpoints/{checkpoint_dir.name}"
    for attempt in range(max_retries):
        try:
            api.upload_folder(
                folder_path=str(checkpoint_dir),
                repo_id=remote_repo_id,
                repo_type="model",
                path_in_repo=path_in_repo,
                commit_message=f"checkpoint {run_id} step {checkpoint_dir.name.removeprefix('step_')}",
            )
            return {"status": "BACKUP_OK", "path": path_in_repo}
        except Exception as error:
            if attempt + 1 == max_retries:
                message = f"FAIL_BACKUP: remote upload failed after {max_retries} attempts ({type(error).__name__})"
                if required:
                    raise RemoteBackupError(message) from error
                return {"status": "FAIL_BACKUP", "message": message}
            time.sleep(2 ** attempt)


def local_checkpoints(run_dir):
    root = Path(run_dir) / "checkpoints"
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.glob("step_*") if path.is_dir() and not path.name.endswith(".tmp")),
        key=lambda path: int(path.name.removeprefix("step_")),
    )


def find_latest_remote_checkpoint(repo_id, token, run_id, config_fingerprint, dataset_version,
                                  setup, cache_dir=None):
    """Download the newest complete and compatible checkpoint from a private HF repo."""
    require_remote_credentials(token, repo_id, required=True)
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not getattr(info, "private", False):
        raise RemoteBackupError("Checkpoint repository is not private")
    prefix = f"{run_id}/checkpoints/"
    manifests = [
        name for name in api.list_repo_files(repo_id=repo_id, repo_type="model")
        if name.startswith(prefix) and name.endswith("/manifest.json")
    ]
    manifests.sort(
        key=lambda name: int(name.split("/")[-2].removeprefix("step_")), reverse=True
    )
    for manifest_name in manifests:
        try:
            path = hf_hub_download(
                repo_id=repo_id, repo_type="model", filename=manifest_name, token=token, cache_dir=cache_dir
            )
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                continue
            validate_checkpoint_identity(manifest, config_fingerprint, dataset_version, setup)
            folder = manifest_name.rpartition("/")[0]
            snapshot = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                token=token,
                cache_dir=cache_dir,
                allow_patterns=[f"{folder}/**"],
            )
            candidate = Path(snapshot) / folder
            verified = validate_checkpoint(candidate, config_fingerprint, dataset_version, setup)
            if verified is not None:
                return candidate
        except CheckpointMismatchError:
            raise
        except Exception:
            continue
    return None


def validate_checkpoint_identity(manifest, config_fingerprint, dataset_version, setup):
    for key, expected in (
        ("config_fingerprint", config_fingerprint),
        ("dataset_version", dataset_version),
        ("setup", setup),
    ):
        if manifest.get(key) != expected:
            raise CheckpointMismatchError(
                f"Remote checkpoint {key} mismatch: found {manifest.get(key)!r}, expected {expected!r}"
            )


def resolve_resume_checkpoint(run_dir, run_id, config_fingerprint, dataset_version, setup,
                              remote_repo_id=None, token=None, remote_required=True, cache_dir=None):
    """Choose the newest compatible local or remote checkpoint for the same run identity."""
    require_remote_credentials(token, remote_repo_id, required=remote_required)
    local = []
    for path in local_checkpoints(run_dir):
        manifest = validate_checkpoint(path)
        if manifest is None:
            continue
        validate_checkpoint_identity(manifest, config_fingerprint, dataset_version, setup)
        local.append((int(manifest["global_step"]), path))
    remote_path = None
    if token and remote_repo_id:
        remote_path = find_latest_remote_checkpoint(
            remote_repo_id, token, run_id, config_fingerprint, dataset_version, setup, cache_dir=cache_dir
        )
    candidates = list(local)
    if remote_path is not None:
        remote_manifest = validate_checkpoint(remote_path, config_fingerprint, dataset_version, setup)
        if remote_manifest is not None:
            candidates.append((int(remote_manifest["global_step"]), remote_path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def restore_checkpoint(checkpoint_dir, optimizer, scheduler, scaler, projector_modules=None,
                       active_metrics_path=None, config_fingerprint=None, dataset_version=None, setup=None):
    """Verify and restore every saved training state before resuming the next step."""
    import torch

    manifest = validate_checkpoint(checkpoint_dir, config_fingerprint, dataset_version, setup)
    if manifest is None:
        raise CheckpointError(f"Checkpoint is incomplete or corrupt: {checkpoint_dir}")
    state = torch.load(Path(checkpoint_dir) / "trainer_state.pt", map_location="cpu", weights_only=False)
    if int(state.get("global_step", -1)) != int(manifest["global_step"]):
        raise CheckpointError("trainer_state global_step differs from checkpoint manifest")
    if setup in ("E2", "E3"):
        if not projector_modules:
            raise CheckpointError(f"{setup} resume requires projector modules")
        from student.model import load_projector_state_dict

        projector_state = torch.load(Path(checkpoint_dir) / "projector.pt", map_location="cpu", weights_only=True)
        load_projector_state_dict(projector_modules, projector_state)
    step, elapsed = apply_training_state(state, optimizer, scheduler, scaler)
    if active_metrics_path is not None:
        restore_metrics(checkpoint_dir, active_metrics_path)
    return {"global_step": step, "elapsed_seconds": elapsed, "manifest": manifest}

