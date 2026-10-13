"""Shared E0-E4 experiment orchestration used by both lab notebooks."""

import hashlib
import json
from pathlib import Path

from student.checkpointing import (
    ensure_private_repo,
    find_latest_remote_checkpoint,
    resolve_resume_checkpoint,
    upload_checkpoint,
    validate_checkpoint,
)
from student.data import validate_splits
from student.dataset import PROMPT
from student.inference import read_existing_predictions, run_incremental_inference, score_predictions
from student.model import MODEL_ID, load_model_bundle
from student.training import TrainingOptions, git_revision, metric_summary, train_model
from student.visualization import render_prediction_examples


def load_experiment_config(repo_root, setup):
    import yaml

    path = Path(repo_root) / "configs" / f"{setup.lower()}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("name") != setup:
        raise ValueError(f"Config name mismatch in {path}: {config.get('name')!r}")
    return config


def config_fingerprint(config):
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_id_for(experiment, base_setup=None, rank=None):
    if experiment == "E4":
        if base_setup not in ("E1", "E2", "E3") or rank not in (4, 64):
            raise ValueError("E4 requires base_setup E1/E2/E3 and rank 4 or 64")
        return f"e4_rank_{rank}_base_{base_setup.lower()}"
    if experiment not in ("E0", "E1", "E2", "E3"):
        raise ValueError(f"Unsupported experiment: {experiment}")
    return experiment.lower()


def _relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _prediction_output(run_dir, split):
    return Path(run_dir) / f"predictions_{split}.jsonl"


def _training_meta(train_result, metrics_path):
    manifest = train_result["run_manifest"]
    summary = metric_summary(metrics_path)
    values = {
        "trainable_params": int(manifest.get("trainable_params", 0)),
        "peak_vram_gb": float(manifest.get("peak_gpu_gb", summary["peak_gpu_gb"])),
        "train_minutes": float(manifest.get("train_minutes", summary["elapsed_seconds"] / 60)),
    }
    if any(value <= 0 for value in values.values()):
        raise RuntimeError(f"Training metadata must be measured and positive before scoring: {values}")
    return values


def run_experiment(
    experiment,
    repo_root,
    data_dir,
    run_root,
    split="test",
    e4_base="E1",
    e4_rank=4,
    micro_batch_size=None,
    grad_accum=None,
    max_steps=None,
    checkpoint_every=25,
    remote_repo_id=None,
    token=None,
    remote_required=True,
    evaluate=True,
    on_step=None,
):
    """Load data/model, optionally train, infer, and call the canonical scorer."""
    repo_root = Path(repo_root).resolve()
    data_dir = Path(data_dir).resolve()
    run_root = Path(run_root).resolve()
    dataset = validate_splits(data_dir, check_expected=True)
    rows = dataset["rows"]
    dataset_version = dataset["meta"]["dataset_version"]
    if split not in ("val", "test"):
        raise ValueError(f"Unsupported evaluation split: {split}")

    if experiment == "E4":
        base_setup = e4_base
        rank = e4_rank
    else:
        base_setup = experiment
        rank = None
    run_id = run_id_for(experiment, base_setup=base_setup if experiment == "E4" else None, rank=rank)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config(repo_root, base_setup)

    if experiment == "E0":
        training_options = None
        lora_rank = None
        fingerprint_payload = {
            "model_id": MODEL_ID,
            "experiment": experiment,
            "dataset_version": dataset_version,
            "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "max_new_tokens": int(config["max_new_tokens"]),
            "coordinate_scale": "norm1000",
        }
    else:
        lora_config = dict(config.get("lora", {}))
        lora_rank = rank if experiment == "E4" else int(lora_config.get("r", 16))
        lora_alpha = lora_rank * 2 if experiment == "E4" else int(lora_config.get("alpha", lora_rank * 2))
        selected_micro_batch = int(micro_batch_size or config.get("batch_size", 2))
        selected_grad_accum = int(grad_accum or config.get("grad_accum", 4))
        selected_steps = int(max_steps or config.get("max_steps", 300))
        training_options = TrainingOptions(
            setup=base_setup,
            max_steps=selected_steps,
            learning_rate=float(config.get("lr", 2e-4)),
            micro_batch_size=selected_micro_batch,
            grad_accum=selected_grad_accum,
            checkpoint_every=int(checkpoint_every),
            seed=int(config.get("seed", 42)),
            coordinate_scale="norm1000",
            require_remote_backup=remote_required,
        )
        fingerprint_payload = {
            "model_id": MODEL_ID,
            "experiment": experiment,
            "base_setup": base_setup,
            "dataset_version": dataset_version,
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": float(lora_config.get("dropout", 0.05)),
            "training_options": training_options.__dict__,
            "lora_targets": lora_config.get("targets"),
            "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "coordinate_scale": "norm1000",
            "code_revision": git_revision(repo_root),
            "model_config": config,
        }
    fingerprint = config_fingerprint(fingerprint_payload)
    resume_checkpoint = None
    if experiment != "E0":
        if remote_required:
            ensure_private_repo(remote_repo_id, token)
        resume_checkpoint = resolve_resume_checkpoint(
            run_dir,
            run_id,
            fingerprint,
            dataset_version,
            base_setup,
            remote_repo_id=remote_repo_id,
            token=token,
            remote_required=remote_required,
        )
        if resume_checkpoint is not None and _relative_to(resume_checkpoint, run_dir):
            local_manifest = validate_checkpoint(
                resume_checkpoint, fingerprint, dataset_version, base_setup
            )
            remote_checkpoint = None
            if token and remote_repo_id:
                remote_checkpoint = find_latest_remote_checkpoint(
                    remote_repo_id,
                    token,
                    run_id,
                    fingerprint,
                    dataset_version,
                    base_setup,
                )
            remote_step = -1
            if remote_checkpoint is not None:
                remote_manifest = validate_checkpoint(
                    remote_checkpoint, fingerprint, dataset_version, base_setup
                )
                remote_step = int(remote_manifest["global_step"]) if remote_manifest else -1
            if local_manifest and int(local_manifest["global_step"]) > remote_step and remote_required:
                upload_checkpoint(
                    resume_checkpoint,
                    remote_repo_id=remote_repo_id,
                    token=token,
                    run_id=run_id,
                    required=True,
                )

    bundle = load_model_bundle(
        setup=base_setup,
        model_id=MODEL_ID,
        rank=lora_rank or 16,
        alpha=(2 * lora_rank if experiment == "E4" else None),
        dropout=float(config.get("lora", {}).get("dropout", 0.05)),
        adapter_path=Path(resume_checkpoint) / "adapter" if resume_checkpoint else None,
        trainable=experiment != "E0",
        device_map={"": 0},
    )
    training_result = None
    metadata = {}
    if experiment != "E0":
        training_result = train_model(
            bundle=bundle,
            train_rows=rows["train"],
            data_dir=data_dir,
            repo_root=repo_root,
            run_dir=run_dir,
            options=training_options,
            config_fingerprint=fingerprint,
            dataset_version=dataset_version,
            remote_repo_id=remote_repo_id,
            token=token,
            resume_checkpoint=resume_checkpoint,
            run_metadata=fingerprint_payload,
            on_step=on_step,
        )
        metadata = _training_meta(training_result, run_dir / "metrics.jsonl")
    if not evaluate:
        return {
            "experiment": experiment,
            "run_id": run_id,
            "run_dir": run_dir,
            "dataset_version": dataset_version,
            "training": training_result,
            "metadata": metadata,
            "bundle": bundle,
        }

    bundle.model.eval()
    if hasattr(bundle.model, "config"):
        bundle.model.config.use_cache = True
    prediction_path = _prediction_output(run_dir, split)
    run_incremental_inference(
        rows=rows[split],
        image_root=data_dir,
        processor=bundle.processor,
        model=bundle.model,
        output_path=prediction_path,
        device="cuda:0",
    )
    evidence_name = (
        f"e4_rank_{rank}_base_{base_setup.lower()}"
        if experiment == "E4"
        else experiment.lower()
    )
    if split == "test":
        evidence_path = repo_root / "report" / "evidence" / f"{evidence_name}.json"
    else:
        evidence_path = run_dir / "evidence" / f"{split}.json"
    evidence = score_predictions(
        repo_root=repo_root,
        data_dir=data_dir,
        prediction_path=prediction_path,
        evidence_path=evidence_path,
        name=(f"E4-rank-{rank}-{base_setup}" if experiment == "E4" else experiment),
        split=split,
        coord="norm1000",
        metadata=metadata,
    )
    existing = read_existing_predictions(prediction_path)
    examples = []
    if split == "test":
        examples = render_prediction_examples(
            rows=rows[split],
            prediction_path=prediction_path,
            image_root=data_dir,
            output_dir=evidence_path.parent / "figures" / evidence_name,
            limit=6,
            coord="norm1000",
        )
    evidence["prediction_path"] = str(prediction_path)
    evidence["evidence_path"] = str(evidence_path)
    evidence["raw_examples"] = list(existing.items())[:5]
    evidence["annotated_examples"] = [str(path) for path in examples]
    return {
        "experiment": experiment,
        "run_id": run_id,
        "run_dir": run_dir,
        "dataset_version": dataset_version,
        "training": training_result,
        "metadata": metadata,
        "predictions": prediction_path,
        "evidence": evidence,
        "bundle": bundle,
    }

