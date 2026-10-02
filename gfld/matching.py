"""Score-ordered greedy one-to-one matching, using inclusive VOC coordinates."""
import numpy as np


def box_iou(boxes, ground_truth):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    ground_truth = np.asarray(ground_truth, dtype=float).reshape(-1, 4)
    low = np.maximum(boxes[:, None, :2], ground_truth[None, :, :2])
    high = np.minimum(boxes[:, None, 2:], ground_truth[None, :, 2:])
    intersection = np.maximum(high - low + 1.0, 0).prod(axis=2)
    a = np.maximum(boxes[:, 2:] - boxes[:, :2] + 1.0, 0).prod(axis=1)
    b = np.maximum(ground_truth[:, 2:] - ground_truth[:, :2] + 1.0, 0).prod(axis=1)
    union = a[:, None] + b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_boxes(predictions, annotation, iou_threshold=0.5, mode="spatial", label_sets=None):
    """Return GT indices in original prediction order; -1 denotes no match.

    Modes: spatial (CP calibration), label (top-1 baseline), set (CF).
    Equal-confidence ties retain input order; equal-IoU ties use annotation order.
    """
    if mode not in {"spatial", "label", "set"} or not 0 < iou_threshold <= 1:
        raise ValueError("Invalid matching mode or IoU threshold")
    if ((annotation.labels < 0) | (annotation.labels >= predictions.class_scores.shape[1])).any():
        raise ValueError("Ground-truth class outside the prediction class order")
    if mode == "set":
        if label_sets is None or np.shape(label_sets) != predictions.class_scores.shape:
            raise ValueError("Set matching requires a boolean (N, C) label-set matrix")
        label_sets = np.asarray(label_sets, dtype=bool)
    overlaps = box_iou(predictions.boxes, annotation.boxes)
    matches = np.full(len(predictions.scores), -1, dtype=int)
    used = np.zeros(len(annotation.labels), dtype=bool)
    top1 = predictions.class_scores.argmax(axis=1)
    for j in np.argsort(-predictions.scores, kind="stable"):
        eligible = ~used
        if mode == "label":
            eligible = eligible & (annotation.labels == top1[j])
        elif mode == "set":
            eligible = eligible & label_sets[j, annotation.labels]
        indices = np.flatnonzero(eligible)
        if not len(indices):
            continue
        best = indices[np.argmax(overlaps[j, indices])]
        if overlaps[j, best] >= iou_threshold:
            matches[j] = best
            used[best] = True
    return matches
