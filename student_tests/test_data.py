import json

from scripts.labcommon import CLASSES
from scripts.splitting import SPLITS, dataset_version
from student.data import EXPECTED_COUNTS, validate_splits


def _write_dataset(root):
    (root / "splits").mkdir(parents=True)
    (root / "images").mkdir()
    rows_by_split = {split: [] for split in SPLITS}
    groups = {}
    for split, count in EXPECTED_COUNTS.items():
        # Keep the test fixture small by checking metadata/count logic without README expectations.
        rows_by_split[split] = [{
            "image_id": split,
            "image_path": "images/example.jpg",
            "width": 640,
            "height": 360,
            "session": f"session-{split}",
            "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}],
        }]
        groups[f"session-{split}"] = split
        (root / "splits" / f"{split}.jsonl").write_text(
            json.dumps(rows_by_split[split][0]) + "\n", encoding="utf-8"
        )
    (root / "images" / "example.jpg").write_bytes(b"fixture")
    image_splits = [(split, split) for split in SPLITS]
    meta = {
        "dataset_version": dataset_version(image_splits),
        "classes": list(CLASSES),
        "groups": groups,
    }
    (root / "splits" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_validate_splits_checks_sessions_and_metadata(tmp_path):
    _write_dataset(tmp_path)
    result = validate_splits(tmp_path, check_expected=False)
    assert result["meta"]["dataset_version"].startswith("d10-")
    assert len(result["sessions"]) == 3

