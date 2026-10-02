"""Label-set calibration using true-class scores from matched objects."""
from dataclasses import dataclass
import numpy as np
from .matching import match_boxes


def collect_scores(predictions, annotations, iou_threshold=0.5):
    scores, image_ids = [], []
    for p in predictions:
        ann = annotations[p.image_id]
        matches = match_boxes(p, ann, iou_threshold)
        for j in np.argsort(-p.scores, kind="stable"):
            if matches[j] >= 0:
                scores.append(p.class_scores[j, ann.labels[matches[j]]])
                image_ids.append(p.image_id)
    return np.asarray(scores, dtype=float), image_ids


def score_quantile(scores, alpha=0.01, weights=None):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a finite vector")
    if not len(scores):
        return 0.0
    q = min(np.ceil((len(scores) + 1) * alpha) / len(scores), 1.0)
    if weights is None:
        return float(np.quantile(scores, q, method="lower"))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != scores.shape or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, aligned, and have positive sum")
    order = np.argsort(scores)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    index = min(np.searchsorted(cumulative, q, side="left"), len(scores) - 1)
    return float(scores[order[index]])


@dataclass(frozen=True)
class LabelSets:
    threshold: float
    alpha: float
    matched_pairs: int
    weighted: bool = False

    def predict(self, class_scores, empty_policy="keep"):
        """Include classes whose scores meet the calibrated threshold.

        Empty sets are retained by default. Optional policies expand them to
        all classes ('all') or the highest-scoring class ('top1').
        """
        if empty_policy not in {"keep", "all", "top1"}:
            raise ValueError("Unknown empty-set policy")
        values = np.asarray(class_scores)
        if values.ndim != 2:
            raise ValueError("class_scores must be a matrix")
        sets = values >= self.threshold
        empty = ~sets.any(axis=1)
        if empty_policy == "all":
            sets[empty] = True
        elif empty_policy == "top1":
            rows = np.flatnonzero(empty)
            sets[rows, values[rows].argmax(axis=1)] = True
        return sets


def calibrate_cp(predictions, annotations, alpha=0.01, iou_threshold=0.5, image_weights=None):
    scores, image_ids = collect_scores(predictions, annotations, iou_threshold)
    weights = None if image_weights is None else np.array([image_weights[i] for i in image_ids])
    return LabelSets(score_quantile(scores, alpha, weights), alpha, len(scores), weights is not None)
