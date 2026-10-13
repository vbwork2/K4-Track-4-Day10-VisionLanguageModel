"""Conversions between normalized YOLO labels and model grounding JSON."""

import json
import math

from scripts.labcommon import CLASSES


COORDINATE_SCALES = ("norm1000", "pixel")


def _scale_dimensions(width, height, coord):
    if coord == "norm1000":
        return 1000, 1000
    if coord == "pixel":
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        return int(width), int(height)
    raise ValueError(f"Unsupported coordinate scale: {coord}")


def yolo_to_box(obj, width=640, height=360, coord="norm1000"):
    """Convert one normalized YOLO object to a clipped xyxy grounding box."""
    label = obj["label"]
    if label not in CLASSES:
        raise ValueError(f"Unsupported label: {label}")
    values = tuple(float(value) for value in obj["yolo"])
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("YOLO coordinates must contain four finite numbers")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("YOLO coordinates must be in [0, 1]")
    scale_x, scale_y = _scale_dimensions(width, height, coord)
    xc, yc, box_width, box_height = values
    raw = (
        (xc - box_width / 2) * scale_x,
        (yc - box_height / 2) * scale_y,
        (xc + box_width / 2) * scale_x,
        (yc + box_height / 2) * scale_y,
    )
    x1 = max(0, min(scale_x, round(raw[0])))
    y1 = max(0, min(scale_y, round(raw[1])))
    x2 = max(0, min(scale_x, round(raw[2])))
    y2 = max(0, min(scale_y, round(raw[3])))
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects, width=640, height=360, coord="norm1000"):
    """Serialize boxes in stable top-to-bottom, left-to-right order."""
    boxes = [yolo_to_box(obj, width=width, height=height, coord=coord) for obj in objects]
    boxes.sort(key=lambda item: (item["bbox_2d"][1], item["bbox_2d"][0]))
    return json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))


def grounding_box_to_pixels(box, width, height, coord="norm1000"):
    """Convert a grounding box to clipped pixel coordinates for drawing."""
    scale_x, scale_y = _scale_dimensions(width, height, coord)
    x1, y1, x2, y2 = box["bbox_2d"]
    if coord == "norm1000":
        x1, x2 = x1 * width / scale_x, x2 * width / scale_x
        y1, y2 = y1 * height / scale_y, y2 * height / scale_y
    values = (round(x1), round(y1), round(x2), round(y2))
    return (
        max(0, min(width, values[0])),
        max(0, min(height, values[1])),
        max(0, min(width, values[2])),
        max(0, min(height, values[3])),
    )


def boxes_to_yolo(boxes, width, height, coord="norm1000"):
    """Convert grounding boxes to normalized YOLO tuples for CPU checks."""
    scale_x, scale_y = _scale_dimensions(width, height, coord)
    converted = []
    for box in boxes:
        if box["label"] not in CLASSES:
            raise ValueError(f"Unsupported label: {box['label']}")
        x1, y1, x2, y2 = box["bbox_2d"]
        if coord == "norm1000":
            x1, x2 = x1 * width / scale_x, x2 * width / scale_x
            y1, y2 = y1 * height / scale_y, y2 * height / scale_y
        converted.append({
            "label": box["label"],
            "yolo": [
                ((x1 + x2) / 2) / width,
                ((y1 + y2) / 2) / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            ],
        })
    return converted

