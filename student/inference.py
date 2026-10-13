"""Incremental VLM inference and canonical scorer integration."""

import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

from student.dataset import PROMPT, user_message


def read_existing_predictions(path):
    """Read valid JSONL predictions, keeping the last complete record per image."""
    path = Path(path)
    found = {}
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row.get("image_id"), str) or not isinstance(row.get("raw"), str):
                    raise ValueError("invalid prediction fields")
                found[row["image_id"]] = row["raw"]
            except (json.JSONDecodeError, AttributeError, ValueError):
                warnings.warn(f"Skipping invalid/incomplete prediction line {line_number} in {path}")
    return found


def _write_prediction_map(path, rows, found):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            image_id = row["image_id"]
            if image_id in found:
                stream.write(json.dumps({"image_id": image_id, "raw": found[image_id]}, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def generate_one(record, image_root, processor, model, device=None, prompt=PROMPT, max_new_tokens=256):
    """Generate one raw answer without applying any ground-truth information to the prompt."""
    import torch
    from PIL import Image

    with Image.open(Path(image_root) / record["image_path"]) as source:
        image = source.convert("RGB")
    inputs = processor.apply_chat_template(
        [user_message(image, prompt=prompt)],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_device = device or getattr(model, "device", torch.device("cuda:0"))
    inputs = {key: value.to(model_device) for key, value in inputs.items() if torch.is_tensor(value)}
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        output[0, prompt_length:], skip_special_tokens=True
    ).strip()


def run_incremental_inference(
    rows,
    image_root,
    processor,
    model,
    output_path,
    device=None,
    prompt=PROMPT,
    max_new_tokens=256,
    on_prediction=None,
):
    """Resume an interrupted JSONL run and flush each new image prediction."""
    output_path = Path(output_path)
    found = read_existing_predictions(output_path)
    valid_ids = {row["image_id"] for row in rows}
    found = {image_id: raw for image_id, raw in found.items() if image_id in valid_ids}
    _write_prediction_map(output_path, rows, found)
    for index, record in enumerate(rows, 1):
        image_id = record["image_id"]
        if image_id in found:
            continue
        raw = generate_one(
            record,
            image_root=image_root,
            processor=processor,
            model=model,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        found[image_id] = raw
        with output_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"image_id": image_id, "raw": raw}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if on_prediction is not None:
            on_prediction(index, record, raw)
    if set(found) != valid_ids:
        missing = valid_ids - set(found)
        raise RuntimeError(f"Inference did not produce one prediction for every row: {len(missing)} missing")
    _write_prediction_map(output_path, rows, found)
    return output_path


def score_predictions(
    repo_root,
    data_dir,
    prediction_path,
    evidence_path,
    name,
    split="test",
    coord="norm1000",
    metadata=None,
):
    """Run the original score.py only after verifying a complete prediction set."""
    repo_root = Path(repo_root).resolve()
    data_dir = Path(data_dir).resolve()
    prediction_path = Path(prediction_path).resolve()
    evidence_path = Path(evidence_path).resolve()
    rows_path = data_dir / "splits" / f"{split}.jsonl"
    expected = [json.loads(line)["image_id"] for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    predictions = read_existing_predictions(prediction_path)
    if len(predictions) != len(expected) or set(predictions) != set(expected):
        raise ValueError(f"Refusing to score incomplete predictions: have {len(predictions)}, expected {len(expected)}")
    command = [
        sys.executable,
        str(repo_root / "scripts" / "score.py"),
        "--pred", str(prediction_path),
        "--splits-dir", str(data_dir / "splits"),
        "--split", split,
        "--coord", coord,
        "--name", name,
        "--out", str(evidence_path),
    ]
    for key, value in (metadata or {}).items():
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError(f"Scorer metadata {key} must be positive")
        command.extend(["--meta", f"{key}={value}"])
    subprocess.run(command, cwd=repo_root, check=True)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("missing_predictions") != 0:
        raise RuntimeError("Canonical scorer reported missing predictions")
    return evidence

