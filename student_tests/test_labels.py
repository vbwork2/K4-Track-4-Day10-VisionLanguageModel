import json

import pytest

from student.labels import boxes_to_yolo, grounding_box_to_pixels, objects_to_answer, yolo_to_box


def test_norm1000_round_trip_returns_original_pixel_edges():
    objects = [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]
    encoded = objects_to_answer(objects)
    boxes = json.loads(encoded)
    assert boxes == [{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}]
    assert grounding_box_to_pixels(boxes[0], 640, 360) == (256, 144, 384, 216)
    assert boxes_to_yolo(boxes, 640, 360)[0]["yolo"] == pytest.approx(objects[0]["yolo"])


def test_pixel_coordinate_scale_uses_image_dimensions():
    obj = {"label": "crack", "yolo": [0.5, 0.5, 0.25, 0.5]}
    assert yolo_to_box(obj, width=640, height=360, coord="pixel")["bbox_2d"] == [240, 90, 400, 270]


def test_yolo_box_clips_to_coordinate_range():
    obj = {"label": "manhole", "yolo": [0.05, 0.1, 0.3, 0.4]}
    assert yolo_to_box(obj)["bbox_2d"] == [0, 0, 200, 300]


def test_empty_objects_are_json_empty_list():
    assert objects_to_answer([]) == "[]"


def test_boxes_are_ordered_by_top_then_left():
    objects = [
        {"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]},
        {"label": "pothole", "yolo": [0.7, 0.2, 0.1, 0.1]},
        {"label": "manhole", "yolo": [0.2, 0.2, 0.1, 0.1]},
    ]
    boxes = json.loads(objects_to_answer(objects))
    assert [box["label"] for box in boxes] == ["manhole", "pothole", "crack"]


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        yolo_to_box({"label": "pothole", "yolo": [0.5, 0.5, float("nan"), 0.2]})
    with pytest.raises(ValueError):
        yolo_to_box({"label": "other", "yolo": [0.5, 0.5, 0.2, 0.2]})

