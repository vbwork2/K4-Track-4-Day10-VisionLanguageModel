import json

import pytest

from student.reward import ground_truth_from_record, iou_reward, reward_for_record


def _completion(label="pothole", box=(100, 100, 300, 300)):
    return json.dumps([{"label": label, "bbox_2d": list(box)}])


def test_reward_is_one_for_perfect_match():
    assert iou_reward(
        _completion(), [{"label": "pothole", "bbox": (100, 100, 300, 300)}]
    ) == pytest.approx(1.0)


def test_malformed_json_has_zero_reward():
    assert iou_reward("not json", [{"label": "pothole", "bbox": (100, 100, 300, 300)}]) == 0.0


def test_wrong_class_receives_valid_json_baseline_only():
    assert iou_reward(
        _completion(label="crack"), [{"label": "pothole", "bbox": (100, 100, 300, 300)}]
    ) == pytest.approx(0.2)


def test_hallucinated_box_is_penalized_by_denominator():
    assert iou_reward(_completion(), []) == pytest.approx(0.2)


def test_both_empty_has_full_reward():
    assert iou_reward("[]", []) == 1.0


def test_record_ground_truth_conversion_matches_norm1000_reward():
    record = {
        "width": 640,
        "height": 360,
        "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}],
    }
    truth = ground_truth_from_record(record)
    assert truth == [{"label": "pothole", "bbox": (400, 400, 600, 600)}]
    assert reward_for_record(_completion(box=(400, 400, 600, 600)), record) == pytest.approx(1.0)

