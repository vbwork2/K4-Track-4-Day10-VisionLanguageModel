"""Pure IoU-based reward helpers for the optional E5 GRPO exercise."""

from scripts.score import iou_xyxy, parse_output
from student.labels import yolo_to_box


def ground_truth_from_record(record, coord="norm1000"):
    """Convert one lab split record to the bbox shape used by the reward function."""
    return [
        {
            "label": obj["label"],
            "bbox": tuple(yolo_to_box(
                obj,
                width=record["width"],
                height=record["height"],
                coord=coord,
            )["bbox_2d"]),
        }
        for obj in record["objects"]
    ]


def reward_for_record(completion, record, coord="norm1000", iou_threshold=0.5):
    return iou_reward(
        completion,
        ground_truth_from_record(record, coord=coord),
        iou_threshold=iou_threshold,
    )


def iou_reward(completion, ground_truth, iou_threshold=0.5):
    """Score valid JSON and greedy same-class box matches using the lab definition."""
    predictions = parse_output(completion)
    if predictions is None:
        return 0.0
    if not ground_truth and not predictions:
        return 1.0
    pairs = []
    for gt_index, target in enumerate(ground_truth):
        target_label = target["label"]
        target_box = target.get("bbox", target.get("bbox_2d"))
        if target_box is None:
            raise ValueError("Ground-truth boxes require bbox or bbox_2d")
        for pred_index, prediction in enumerate(predictions):
            if prediction["label"] != target_label:
                continue
            overlap = iou_xyxy(target_box, prediction["bbox"])
            pairs.append((overlap, gt_index, pred_index))
    pairs.sort(key=lambda item: item[0], reverse=True)
    matched_gt = set()
    matched_predictions = set()
    total_iou = 0.0
    for overlap, gt_index, pred_index in pairs:
        if overlap < iou_threshold:
            break
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
        total_iou += overlap
    denominator = max(len(ground_truth), len(predictions), 1)
    return 0.2 + 0.8 * total_iou / denominator

