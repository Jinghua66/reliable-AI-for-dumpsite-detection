"""Domain weighting, weighted label-set calibration, and weighted filtering."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .data import feature_matrix
from .conformal import collect_scores, calibrate_cp
from .filtering import Filtering, ThresholdResult, detection_counts


class DensityRatio:
    def __init__(self, weight_cap=10.0, C=1.0):
        if not np.isfinite(weight_cap) or weight_cap <= 0:
            raise ValueError("weight_cap must be positive and finite")
        self.weight_cap = weight_cap
        self.C = C
        self.model = None

    def fit(self, source, target):
        source, target = np.asarray(source), np.asarray(target)
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1] or not len(source) or not len(target):
            raise ValueError("Source and target need nonempty, aligned feature matrices")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("Non-finite features")
        balanced = min(len(source), len(target)) / max(len(source), len(target)) < 0.1
        self.model = LogisticRegression(C=self.C, max_iter=2000 if balanced else 1000,
                                        class_weight="balanced" if balanced else None)
        self.model.fit(np.vstack([source, target]), np.r_[np.zeros(len(source)), np.ones(len(target))])
        return self

    def weights(self, features):
        if self.model is None:
            raise RuntimeError("Fit the density-ratio model first")
        p = self.model.predict_proba(features)[:, 1]
        return np.minimum(p / np.maximum(1 - p, 1e-9), self.weight_cap)


def estimate_weights(source_ids, source_features, target_features, weight_cap=10.0, sort_ids=False):
    """Only features are accepted for the target; target labels cannot enter fitting."""
    source_ids = list(dict.fromkeys(source_ids))
    target_ids = list(target_features)
    if sort_ids:
        source_ids, target_ids = sorted(source_ids), sorted(target_ids)
    if not source_ids or not target_ids:
        raise ValueError("Density estimation needs both source and target images")
    model = DensityRatio(weight_cap).fit(feature_matrix(source_features, source_ids),
                                        feature_matrix(target_features, target_ids))
    return model, dict(zip(source_ids, model.weights(feature_matrix(source_features, source_ids))))


def calibrate_shift_cp(predictions, annotations, source_features, target_features,
                       alpha=0.01, iou_threshold=0.5, weight_cap=10.0):
    """Fit image weights and calibrate weighted label sets.

    Returns (label-set model, image weights, domain model). CF separately fits
    on all source calibration images; CP uses images contributing matched pairs.
    """
    _, matched_ids = collect_scores(predictions, annotations, iou_threshold)
    if not matched_ids:
        return calibrate_cp(predictions, annotations, alpha, iou_threshold, {}), {}, None
    domain, weights = estimate_weights(matched_ids, source_features, target_features, weight_cap, sort_ids=True)
    return calibrate_cp(predictions, annotations, alpha, iou_threshold, weights), weights, domain


def weighted_risk(counts, image_weights, threshold, kind="recall", delta=0.5):
    """Return empirical risk and an image-level Bernstein-style margin."""
    if kind not in {"recall", "precision"} or not 0 < delta < 1:
        raise ValueError("Invalid risk kind or delta")
    w = np.asarray(image_weights, dtype=float)
    if w.shape != counts.gt_counts.shape or not np.isfinite(w).all() or (w < 0).any() or not w.sum() > 0:
        raise ValueError("Invalid image weights")
    keep = counts.scores >= threshold
    n_images = len(w)
    tp = np.bincount(counts.image_index[keep & counts.is_tp], minlength=n_images)
    kept = np.bincount(counts.image_index[keep], minlength=n_images)
    denominator_counts = counts.gt_counts if kind == "recall" else kept
    errors = denominator_counts - tp
    risks = np.divide(errors, denominator_counts, out=np.zeros(n_images, dtype=float), where=denominator_counts > 0)
    effective = w * denominator_counts
    total = effective.sum()
    if total == 0:
        # A zero denominator cannot support threshold selection.
        return float("nan"), float("nan")
    risk = float(np.dot(w, errors) / total)
    margin = (np.std(risks) * np.linalg.norm(effective) / total * np.sqrt(2 * np.log(1 / delta))
              + effective.max() / (3 * total) * np.log(1 / delta))
    return risk, float(margin)


def _weighted_search(counts, weights, risk_limit, kind, delta, method):
    grid = np.unique(counts.scores)
    if not len(grid):
        return None
    lower = float(grid[0] - 1e-9)

    def value(t):
        risk, margin = weighted_risk(counts, weights, t, kind, delta)
        return risk if method == "empirical" else risk + margin

    first = value(lower)
    if kind == "recall":
        first_passes = np.isfinite(first) and first <= risk_limit
        if not first_passes and method != "greedy":
            return None
        best = (lower, first) if first_passes else None
        for threshold in grid:
            bound = value(threshold)
            if np.isfinite(bound) and bound <= risk_limit:
                best = (float(threshold), bound)
            elif method != "greedy":
                break
        return best
    if method != "fixed_sequence" and np.isfinite(first) and first <= risk_limit:
        return lower, first
    if method in {"greedy", "empirical"}:
        for threshold in grid:
            bound = value(threshold)
            if np.isfinite(bound) and bound <= risk_limit:
                return float(threshold), bound
        return None
    best = None
    for threshold in grid[::-1]:
        bound = value(threshold)
        if np.isfinite(bound) and bound <= risk_limit:
            best = (float(threshold), bound)
        else:
            break
    return best


def select_weighted_threshold(counts, image_weights, target, kind, delta=0.5,
                              method="greedy", relaxation_step=0.03, max_retries=10):
    if not 0 < target < 1 or not 0 < delta < 1 or method not in {"greedy", "empirical", "fixed_sequence"}:
        raise ValueError("Invalid target, delta, or method")
    if kind not in {"recall", "precision"} or relaxation_step < 0 or max_retries < 0:
        raise ValueError("Invalid kind or relaxation configuration")
    for attempt in range(max_retries + 1):
        relaxation = attempt * relaxation_step
        result = _weighted_search(counts, image_weights, 1 - target + relaxation, kind, delta, method)
        if result is not None:
            threshold, upper_risk = result
            return ThresholdResult(threshold, target, 1 - upper_risk, attempt == 0,
                                   method, len(np.unique(counts.scores)), relaxation)
    return ThresholdResult(0.0 if kind == "recall" else 1.0, target, None, False,
                           method, len(np.unique(counts.scores)), max_retries * relaxation_step, True)


def calibrate_shift_filtering(predictions, annotations, cp, source_features, target_features,
                              target_recall=0.80, target_precision=0.85, delta=0.5,
                              method="greedy", iou_threshold=0.5, weight_cap=10.0,
                              relaxation_step=0.03, max_retries=10):
    _, weight_map = estimate_weights([p.image_id for p in predictions], source_features, target_features, weight_cap)
    weights = np.array([weight_map[p.image_id] for p in predictions])
    counts = detection_counts(predictions, annotations, cp, iou_threshold)
    recall = select_weighted_threshold(counts, weights, target_recall, "recall", delta,
                                       method, relaxation_step, max_retries)
    candidates = [p.select(p.scores >= recall.threshold) for p in predictions]
    counts_p = detection_counts(candidates, annotations, cp, iou_threshold)
    precision = select_weighted_threshold(counts_p, weights, target_precision, "precision", delta,
                                          method, relaxation_step, max_retries)
    return Filtering(cp, recall, precision)
