"""Small annotated ground-truth and prediction examples for lab evidence."""

from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import read_jsonl
from scripts.score import parse_output
from student.labels import grounding_box_to_pixels, yolo_to_box


COLORS = {
    "pothole": (235, 65, 65),
    "crack": (55, 205, 105),
    "manhole": (60, 140, 245),
    "other": (235, 195, 50),
}


def render_ground_truth_overlays(rows, image_root, output_dir, limit=6):
    """Save local label-only overlays for the dataset preparation checklist."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, row in enumerate(rows[:limit], 1):
        with Image.open(Path(image_root) / row["image_path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in row["objects"]:
            box = yolo_to_box(obj, row["width"], row["height"], coord="norm1000")
            xyxy = grounding_box_to_pixels(box, row["width"], row["height"], coord="norm1000")
            color = COLORS.get(obj["label"], COLORS["other"])
            draw.rectangle(xyxy, outline=color, width=3)
            draw.text((xyxy[0], max(0, xyxy[1] - 14)), obj["label"], fill=color)
        path = output_dir / f"ground_truth_{index:02d}_{row['image_id']}.jpg"
        image.save(path, quality=90)
        saved.append(path)
    if len(saved) < min(limit, len(rows)):
        raise ValueError(f"Only saved {len(saved)} ground-truth overlays")
    return saved


def render_prediction_examples(rows, prediction_path, image_root, output_dir, limit=6, coord="norm1000"):
    """Save up to limit real image overlays with GT and parsed model boxes."""
    predictions = {}
    for row in read_jsonl(prediction_path):
        if row["image_id"] in predictions:
            raise ValueError(f"Duplicate prediction for {row['image_id']}")
        predictions[row["image_id"]] = row["raw"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    candidates = [row for row in rows if row["image_id"] in predictions][:limit]
    for index, row in enumerate(candidates, 1):
        with Image.open(Path(image_root) / row["image_path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in row["objects"]:
            box = yolo_to_box(obj, row["width"], row["height"], coord="norm1000")
            xyxy = grounding_box_to_pixels(box, row["width"], row["height"], coord="norm1000")
            color = COLORS.get(obj["label"], COLORS["other"])
            draw.rectangle(xyxy, outline=color, width=3)
            draw.text((xyxy[0], max(0, xyxy[1] - 14)), f"GT {obj['label']}", fill=color)
        parsed = parse_output(predictions[row["image_id"]])
        if parsed:
            for obj in parsed:
                box = {"label": obj["label"], "bbox_2d": list(obj["bbox"])}
                xyxy = grounding_box_to_pixels(box, row["width"], row["height"], coord=coord)
                color = COLORS.get(obj["label"], COLORS["other"])
                draw.rectangle(xyxy, outline=color, width=2)
                draw.text((xyxy[0], min(row["height"] - 14, xyxy[3] + 2)), f"Pred {obj['label']}", fill=color)
        path = output_dir / f"example_{index:02d}_{row['image_id']}.jpg"
        image.save(path, quality=90)
        saved.append(path)
    if len(saved) < min(limit, len(rows)):
        raise ValueError(f"Only saved {len(saved)} annotated examples")
    return saved

