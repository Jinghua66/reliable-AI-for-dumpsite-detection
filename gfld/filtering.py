"""Recall-first filtering with threshold calibration and region assignment."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta
from .matching import match_boxes


@dataclass(frozen=True)
class CalibrationCounts:
    scores: np.ndarray
    is_tp: np.ndarray
    image_index: np.ndarray
    gt_counts: np.ndarray
    image_ids: tuple


def detection_counts(predictions, annotations, cp, iou_threshold=0.5):
    scores, flags, image_index, gt_counts, ids = [], [], [], [], []
    for i, p in enumerate(predictions):
        ann = annotations[p.image_id]
        matched = match_boxes(p, ann, iou_threshold, "set", cp.predict(p.class_scores))
        order = np.argsort(-p.scores, kind="stable")
        scores.extend(p.scores[order])
        flags.extend(matched[order] >= 0)
        image_index.extend([i] * len(p.scores))
        gt_counts.append(len(ann.labels))
        ids.append(p.image_id)
    return CalibrationCounts(np.asarray(scores, dtype=float), np.asarray(flags, dtype=bool),
                             np.asarray(image_index, dtype=int), np.asarray(gt_counts), tuple(ids))


def binomial_lower(k, n, delta):
    """One-sided exact binomial lower confidence bound."""
    if not 0 < delta < 1 or k < 0 or n < k:
        raise ValueError("Invalid binomial count or delta")
    return float(beta.ppf(delta, k, n - k + 1)) if k and n else 0.0


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    target: float
    achieved_bound: float | None
    feasible: bool
    method: str
    candidates: int
    relaxation: float = 0.0
    fallback: bool = False


@dataclass(frozen=True)
class Filtering:
    cp: object
    recall: ThresholdResult
    precision: ThresholdResult

    def apply(self, predictions):
        """Return region labels in input order, without requiring annotations."""
        output = {}
        for p in predictions:
            region = np.full(len(p.scores), "declined", dtype="<U9")
            candidate = p.scores >= self.recall.threshold
            region[candidate] = "uncertain"
            if not self.precision.fallback:
                region[candidate & (p.scores >= self.precision.threshold)] = "confident"
            output[p.image_id] = region
        return output


def select_threshold(counts, target, kind, delta=0.5, method="greedy", skip_highest=20):
    """Select a threshold using binomial bounds or empirical rates.

    'greedy' scans all pointwise bounds; 'fixed_sequence' stops on first failure.
    If no threshold passes, recall retains all candidates and precision abstains.
    """
    if kind not in {"recall", "precision"} or method not in {"greedy", "fixed_sequence", "empirical"}:
        raise ValueError("Invalid threshold search")
    if not 0 < target < 1 or not 0 < delta < 1 or skip_highest < 0:
        raise ValueError("Invalid target, delta, or skip_highest")
    grid = np.unique(counts.scores)[::-1][skip_highest:]
    if kind == "recall":
        grid = grid[::-1]
    passing = []
    for threshold in grid:
        keep = counts.scores >= threshold
        k = int(counts.is_tp[keep].sum())
        n = int(counts.gt_counts.sum()) if kind == "recall" else int(keep.sum())
        bound = (k / n if n else 0.0) if method == "empirical" else binomial_lower(k, n, delta)
        if bound >= target:
            passing.append((float(threshold), bound))
        elif method == "fixed_sequence":
            break
    if not passing:
        fallback = 0.0 if kind == "recall" else 1.0
        return ThresholdResult(fallback, target, None, False, method, len(grid), fallback=True)
    threshold, bound = (max(passing) if kind == "recall" else min(passing))
    return ThresholdResult(threshold, target, bound, True, method, len(grid))


def calibrate_filtering(predictions, annotations, cp, target_recall=0.80, target_precision=0.85,
                        delta=0.5, method="greedy", iou_threshold=0.5, skip_highest=20):
    if not predictions:
        raise ValueError("Calibration requires images")
    counts = detection_counts(predictions, annotations, cp, iou_threshold)
    recall = select_threshold(counts, target_recall, "recall", delta, method, skip_highest)
    candidates = [p.select(p.scores >= recall.threshold) for p in predictions]
    counts_p = detection_counts(candidates, annotations, cp, iou_threshold)
    precision = select_threshold(counts_p, target_precision, "precision", delta, method, skip_highest)
    return Filtering(cp, recall, precision)
