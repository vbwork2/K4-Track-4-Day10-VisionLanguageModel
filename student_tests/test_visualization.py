import json

from PIL import Image

from student.visualization import render_ground_truth_overlays, render_prediction_examples


def test_render_examples_saves_gt_and_prediction_overlays(tmp_path):
    image_root = tmp_path / "data"
    (image_root / "images").mkdir(parents=True)
    image_path = image_root / "images" / "frame.jpg"
    Image.new("RGB", (64, 36), (80, 80, 80)).save(image_path)
    row = {
        "image_id": "frame",
        "image_path": "images/frame.jpg",
        "width": 64,
        "height": 36,
        "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.25, 0.25]}],
    }
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        json.dumps({
            "image_id": "frame",
            "raw": json.dumps([{"label": "pothole", "bbox_2d": [375, 375, 625, 625]}]),
        }) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "figures"
    paths = render_prediction_examples([row], prediction_path, image_root, output_dir, limit=1)
    assert len(paths) == 1
    assert paths[0].is_file()
    with Image.open(paths[0]) as saved:
        assert saved.size == (64, 36)
    labels = render_ground_truth_overlays([row], image_root, tmp_path / "ground-truth", limit=1)
    assert len(labels) == 1 and labels[0].is_file()

