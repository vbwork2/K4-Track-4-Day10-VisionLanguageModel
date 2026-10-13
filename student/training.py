"""Resumable fp16 QLoRA training loop for E1, E2, and E3."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from student.checkpointing import (
    atomic_write_json,
    require_remote_credentials,
    restore_checkpoint,
    save_checkpoint,
)
from student.dataset import assert_image_token_count, collate_examples, encode_example
from student.labels import objects_to_answer


@dataclass(frozen=True)
class TrainingOptions:
    setup: str
    max_steps: int = 300
    learning_rate: float = 2e-4
    micro_batch_size: int = 1
    grad_accum: int = 8
    checkpoint_every: int = 25
    seed: int = 42
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.05
    coordinate_scale: str = "norm1000"
    require_remote_backup: bool = True


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_revision(repo_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def package_versions():
    from importlib.metadata import PackageNotFoundError, version

    output = {}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "trl", "huggingface_hub"):
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output


def sample_indices(row_count, step, micro_step, batch_size, seed):
    """Select deterministic training rows without depending on a dataloader iterator."""
    if row_count <= 0 or batch_size <= 0:
        raise ValueError("row_count and batch_size must be positive")
    sampler = random.Random(seed + step * 8191 + micro_step)
    return [sampler.randrange(row_count) for _ in range(batch_size)]


def last_logged_step(metrics_path):
    path = Path(metrics_path)
    if not path.is_file():
        return 0
    last = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "train_step":
                continue
            step = int(event["step"])
            if step <= last:
                raise ValueError(f"Training metric steps must increase strictly; line {line_number} has {step}")
            last = step
    return last


def rebuild_train_log(metrics_path, train_log_path):
    """Recreate the optional text log from the checkpointed canonical metrics."""
    source = Path(metrics_path)
    target = Path(train_log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        if source.is_file():
            with source.open(encoding="utf-8") as events:
                for line in events:
                    event = json.loads(line)
                    if event.get("event") != "train_step":
                        continue
                    output.write(
                        f"step={event['step']} loss={event['loss']:.6f} lr={event['learning_rate']:.9g} "
                        f"grad_norm={event['grad_norm']:.5f} peak_gpu_gb={event['peak_gpu_gb']:.3f}\n"
                    )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)


def append_metric(metrics_path, event, expected_previous_step):
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step = int(event["step"])
    if step != expected_previous_step + 1:
        raise ValueError(f"Refusing non-monotonic metric step {step} after {expected_previous_step}")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_train_log(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            f"step={event['step']} loss={event['loss']:.6f} lr={event['learning_rate']:.9g} "
            f"grad_norm={event['grad_norm']:.5f} peak_gpu_gb={event['peak_gpu_gb']:.3f}\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _encode_record(record, data_dir, processor, coordinate_scale):
    from PIL import Image

    with Image.open(Path(data_dir) / record["image_path"]) as source:
        image = source.convert("RGB")
    answer = objects_to_answer(
        record["objects"],
        width=record["width"],
        height=record["height"],
        coord=coordinate_scale,
    )
    return encode_example(processor, image, answer)


def _autocast(torch):
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except AttributeError:
        return torch.cuda.amp.autocast(dtype=torch.float16)


def _grad_scaler(torch):
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def _run_manifest(repo_root, dataset_version, options, config_fingerprint, trainable_params,
                  run_id, run_metadata=None):
    manifest = {
        "run_id": run_id,
        "git_sha": git_revision(repo_root),
        "dataset_version": dataset_version,
        "setup": options.setup,
        "config_fingerprint": config_fingerprint,
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "seed": options.seed,
        "training_options": asdict(options),
        "trainable_params": int(trainable_params),
        "package_versions": package_versions(),
        "started_at_utc": utc_now(),
        "status": "initialized",
    }
    if run_metadata:
        manifest["config"] = run_metadata
    return manifest


def train_model(
    bundle,
    train_rows,
    data_dir,
    repo_root,
    run_dir,
    options,
    config_fingerprint,
    dataset_version,
    remote_repo_id=None,
    token=None,
    resume_checkpoint=None,
    run_metadata=None,
    on_step=None,
):
    """Train one supervised setup and persist a verified checkpoint every configured interval."""
    import numpy as np
    import torch
    from transformers import get_cosine_schedule_with_warmup

    if options.setup not in ("E1", "E2", "E3"):
        raise ValueError("train_model accepts E1, E2, or E3")
    if options.max_steps <= 0 or options.grad_accum <= 0 or options.micro_batch_size <= 0:
        raise ValueError("Training steps and batch sizes must be positive")
    if options.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    require_remote_credentials(token, remote_repo_id, required=options.require_remote_backup)
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")
    if not train_rows:
        raise ValueError("Training split is empty")
    if bundle.setup != options.setup:
        raise ValueError(f"Loaded model setup {bundle.setup} does not match {options.setup}")

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    metrics_path = run_dir / "metrics.jsonl"
    train_log_path = run_dir / "train.log"
    train_params = [parameter for parameter in bundle.model.parameters() if parameter.requires_grad]
    if not train_params or sum(parameter.numel() for parameter in train_params) != bundle.trainable_params:
        raise RuntimeError("Loaded model trainable parameter list changed before training")

    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    optimizer = torch.optim.AdamW(train_params, lr=options.learning_rate, weight_decay=0.0)
    warmup_steps = max(1, round(options.max_steps * options.warmup_fraction))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=options.max_steps,
    )
    scaler = _grad_scaler(torch)
    manifest = _run_manifest(
        repo_root,
        dataset_version,
        options,
        config_fingerprint,
        bundle.trainable_params,
        run_id,
        run_metadata=run_metadata,
    )
    active_run_json = run_dir / "run.json"
    atomic_write_json(active_run_json, manifest)

    starting_step = 0
    elapsed_before = 0.0
    if resume_checkpoint is not None:
        restored = restore_checkpoint(
            resume_checkpoint,
            optimizer,
            scheduler,
            scaler,
            projector_modules=bundle.projector_modules,
            active_metrics_path=metrics_path,
            config_fingerprint=config_fingerprint,
            dataset_version=dataset_version,
            setup=options.setup,
        )
        starting_step = restored["global_step"]
        elapsed_before = restored["elapsed_seconds"]
        rebuild_train_log(metrics_path, train_log_path)
        if starting_step >= options.max_steps:
            manifest.update({
                "status": "trained",
                "completed_steps": starting_step,
                "train_minutes": elapsed_before / 60,
                "finished_at_utc": utc_now(),
            })
            atomic_write_json(active_run_json, manifest)
            return {"status": "already_complete", "global_step": starting_step, "run_manifest": manifest}
    else:
        metrics_path.write_text("", encoding="utf-8")
        initial_manifest = dict(manifest, status="training", start_step=0)
        save_checkpoint(
            run_dir=run_dir,
            global_step=0,
            model=bundle.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            setup=options.setup,
            config_fingerprint=config_fingerprint,
            dataset_version=dataset_version,
            trainable_params=bundle.trainable_params,
            elapsed_seconds=0.0,
            metrics_path=metrics_path,
            run_manifest=initial_manifest,
            projector_modules=bundle.projector_modules,
            remote_repo_id=remote_repo_id,
            token=token,
            remote_required=options.require_remote_backup,
            checkpoint_every=options.checkpoint_every,
        )

    previous_step = last_logged_step(metrics_path)
    if previous_step != starting_step:
        raise RuntimeError(
            f"Checkpoint step {starting_step} does not match metrics log step {previous_step}; refusing duplicate metrics"
        )
    probe = _encode_record(train_rows[0], data_dir, bundle.processor, options.coordinate_scale)
    assert_image_token_count(bundle.processor, probe, expected=220)
    del probe

    bundle.model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    manifest.update({"status": "training", "start_step": starting_step, "resumed_from": str(resume_checkpoint) if resume_checkpoint else None})
    atomic_write_json(active_run_json, manifest)
    try:
        for step in range(starting_step + 1, options.max_steps + 1):
            loss_total = 0.0
            for micro_step in range(options.grad_accum):
                indices = sample_indices(
                    len(train_rows), step, micro_step, options.micro_batch_size, options.seed
                )
                examples = [
                    _encode_record(train_rows[index], data_dir, bundle.processor, options.coordinate_scale)
                    for index in indices
                ]
                batch = collate_examples(examples, bundle.processor, device="cuda:0")
                with _autocast(torch):
                    output = bundle.model(**batch)
                    loss = output.loss / options.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at optimizer step {step}: {loss.detach().item()}")
                scaler.scale(loss).backward()
                loss_total += float(loss.detach().item()) * options.grad_accum
                del batch, examples, output, loss

            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(train_params, options.max_grad_norm)
            grad_norm = float(grad_norm_tensor.detach().item())
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at optimizer step {step}: {grad_norm}")
            scale_before = scaler.get_scale()
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"GradScaler skipped the optimizer update at step {step}")
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            elapsed = elapsed_before + time.monotonic() - started
            duration = max(elapsed - float(manifest.get("last_elapsed_seconds", elapsed_before)), 1e-9)
            event = {
                "event": "train_step",
                "run_id": run_id,
                "setup": options.setup,
                "step": step,
                "loss": loss_total / options.grad_accum,
                "learning_rate": learning_rate_used,
                "grad_norm": grad_norm,
                "grad_scale": float(scaler.get_scale()),
                "elapsed_seconds": elapsed,
                "peak_gpu_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
                "images_per_second": float(options.micro_batch_size * options.grad_accum / duration),
                "timestamp_utc": utc_now(),
            }
            append_metric(metrics_path, event, previous_step)
            _write_train_log(train_log_path, event)
            previous_step = step
            manifest["last_elapsed_seconds"] = elapsed
            if on_step is not None:
                on_step(event)
            if step % options.checkpoint_every == 0 or step == options.max_steps:
                manifest.update({"status": "training", "last_checkpoint_step": step})
                atomic_write_json(active_run_json, manifest)
                save_checkpoint(
                    run_dir=run_dir,
                    global_step=step,
                    model=bundle.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    setup=options.setup,
                    config_fingerprint=config_fingerprint,
                    dataset_version=dataset_version,
                    trainable_params=bundle.trainable_params,
                    elapsed_seconds=elapsed,
                    metrics_path=metrics_path,
                    run_manifest=manifest,
                    projector_modules=bundle.projector_modules,
                    remote_repo_id=remote_repo_id,
                    token=token,
                    remote_required=options.require_remote_backup,
                    checkpoint_every=options.checkpoint_every,
                )

        elapsed_total = elapsed_before + time.monotonic() - started
        manifest.update({
            "status": "trained",
            "completed_steps": options.max_steps,
            "train_minutes": elapsed_total / 60,
            "peak_gpu_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "finished_at_utc": utc_now(),
        })
        atomic_write_json(active_run_json, manifest)
        return {"status": "trained", "global_step": options.max_steps, "run_manifest": manifest}
    except Exception as error:
        error_text = str(error)
        if token:
            error_text = error_text.replace(token, "[REDACTED]")
        manifest.update({
            "status": "interrupted",
            "error_type": type(error).__name__,
            "error_message": error_text[:1000],
            "last_checkpoint_step": max(
                [0] + [int(path.name.removeprefix("step_")) for path in (run_dir / "checkpoints").glob("step_*") if path.is_dir() and not path.name.endswith(".tmp")]
            ),
            "finished_at_utc": utc_now(),
        })
        atomic_write_json(active_run_json, manifest)
        raise

