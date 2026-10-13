"""Dataset split loading and integrity checks for the student pipeline."""

import json
import math
import subprocess
import sys
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.splitting import SPLITS, dataset_version


EXPECTED_COUNTS = {
    "train": {"images": 1422, "boxes": {"pothole": 1074, "crack": 1750, "manhole": 691}},
    "val": {"images": 276, "boxes": {"pothole": 66, "crack": 470, "manhole": 103}},
    "test": {"images": 311, "boxes": {"pothole": 121, "crack": 299, "manhole": 163}},
}


def prepare_public_dataset(repo_dir, data_dir, work_dir, source_dir=None, zip_path=None):
    """Prepare Colab data through the unchanged instructor script, then validate it."""
    if source_dir and zip_path:
        raise ValueError("Choose either an extracted dataset directory or a ZIP file")
    data_dir = Path(data_dir)
    if not (data_dir / "splits" / "meta.json").is_file():
        if zip_path:
            source_args = ["--zip", str(zip_path)]
        else:
            if not source_dir:
                import kagglehub
                from scripts.prepare_data import KAGGLE_SLUG

                source_dir = kagglehub.dataset_download(KAGGLE_SLUG)
            source_args = ["--source", str(source_dir)]
        subprocess.run(
            [sys.executable, str(Path(repo_dir) / "scripts" / "prepare_data.py"),
             *source_args, "--out", str(data_dir), "--work", str(work_dir)],
            cwd=repo_dir, check=True,
        )
    return validate_splits(data_dir, check_expected=True)


def validate_splits(data_dir, check_expected=True, require_images=True):
    """Load the instructor split files and assert identity, counts, and no leakage."""
    root = Path(data_dir).resolve()
    split_root = root / "splits"
    rows = {split: read_jsonl(split_root / f"{split}.jsonl") for split in SPLITS}
    meta = json.loads((split_root / "meta.json").read_text(encoding="utf-8"))
    sessions = {}
    all_ids = set()
    summaries = {}
    for split, records in rows.items():
        boxes = {label: 0 for label in CLASSES}
        for record in records:
            image_id = record["image_id"]
            session = record["session"]
            if image_id in all_ids:
                raise ValueError(f"Duplicate image_id across splits: {image_id}")
            all_ids.add(image_id)
            prior_split = sessions.setdefault(session, split)
            if prior_split != split:
                raise ValueError(f"Session leakage: {session} appears in {prior_split} and {split}")
            if record["width"] <= 0 or record["height"] <= 0:
                raise ValueError(f"Invalid image dimensions for {image_id}")
            for obj in record["objects"]:
                if obj["label"] not in boxes:
                    raise ValueError(f"Unknown class in {image_id}: {obj['label']}")
                values = [float(value) for value in obj["yolo"]]
                if len(values) != 4 or not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                    raise ValueError(f"Invalid YOLO label in {image_id}: {obj['yolo']}")
                boxes[obj["label"]] += 1
            if require_images:
                image_path = (root / record["image_path"]).resolve()
                if root not in image_path.parents or not image_path.is_file():
                    raise FileNotFoundError(f"Missing or out-of-root image: {image_path}")
        summaries[split] = {"images": len(records), "boxes": boxes}
        if check_expected and summaries[split] != EXPECTED_COUNTS[split]:
            raise ValueError(f"Unexpected {split} counts: {summaries[split]}")

    if meta.get("classes") != list(CLASSES):
        raise ValueError(f"Unexpected class order in split metadata: {meta.get('classes')}")
    if meta.get("counts") != summaries:
        raise ValueError("Split counts in metadata do not match the JSONL rows")
    computed_version = dataset_version(
        [(record["image_id"], split) for split, records in rows.items() for record in records]
    )
    if meta.get("dataset_version") != computed_version:
        raise ValueError("dataset_version does not match split rows")
    if set(meta.get("groups", {})) != set(sessions):
        raise ValueError("Session groups in metadata do not match split rows")
    for session, split in sessions.items():
        if meta["groups"].get(session) != split:
            raise ValueError(f"Metadata assignment differs for session {session}")
    return {"rows": rows, "meta": meta, "summary": summaries, "sessions": sessions}

