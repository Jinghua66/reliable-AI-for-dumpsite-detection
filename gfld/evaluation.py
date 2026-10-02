"""Evaluation is separate from calibration; metrics use explicit denominators."""
import numpy as np
from .matching import match_boxes
from .conformal import calibrate_cp
from .filtering import calibrate_filtering
from .data import calibration_splits


def evaluate_cp(predictions, annotations, cp, iou_threshold=0.5, empty_policy="keep"):
    covered = total = set_size = 0
    empty_count = 0
    for p in predictions:
        ann = annotations[p.image_id]
        matches = match_boxes(p, ann, iou_threshold)
        sets = cp.predict(p.class_scores, empty_policy)
        for j in np.flatnonzero(matches >= 0):
            total += 1
            covered += int(sets[j, ann.labels[matches[j]]])
            set_size += int(sets[j].sum())
            empty_count += int(not sets[j].any())
    return {"covered": covered, "matched_pairs": total, "set_size_sum": set_size,
            "coverage": covered / total if total else None,
            "mean_set_size": set_size / total if total else None, "empty_sets": empty_count,
            "empty_policy": empty_policy}


def evaluate_detections(predictions, annotations, cp, threshold=0.0, iou_threshold=0.5):
    tp = retained = total_gt = 0
    for p in predictions:
        ann = annotations[p.image_id]
        total_gt += len(ann.labels)
        kept = p.select(p.scores >= threshold)
        matched = match_boxes(kept, ann, iou_threshold, "set", cp.predict(kept.class_scores))
        tp += int((matched >= 0).sum())
        retained += len(kept.scores)
    return {"tp": tp, "fp": retained - tp, "fn": total_gt - tp, "retained": retained,
            "total_gt": total_gt, "precision": tp / retained if retained else None,
            "recall": tp / total_gt if total_gt else None}


def evaluate_filtering(predictions, annotations, filtering, iou_threshold=0.5):
    regions = filtering.apply(predictions)
    candidate = [p.select(regions[p.image_id] != "declined") for p in predictions]
    confident = [p.select(regions[p.image_id] == "confident") for p in predictions]
    return {
        "baseline": evaluate_detections(predictions, annotations, filtering.cp, iou_threshold=iou_threshold),
        "candidate": evaluate_detections(candidate, annotations, filtering.cp, iou_threshold=iou_threshold),
        "confident": evaluate_detections(confident, annotations, filtering.cp, iou_threshold=iou_threshold),
        "regions": {name: sum(int((r == name).sum()) for r in regions.values())
                    for name in ("confident", "uncertain", "declined")},
        "recall_feasible": filtering.recall.feasible,
        "precision_feasible": filtering.precision.feasible,
    }


def cross_validate(predictions, annotations, folds=5, permutations=4, seed=42,
                   cp_alpha=0.01, delta=0.5, target_recall=0.80, target_precision=0.85,
                   iou_threshold=0.5):
    """Pool held-out counts across folds, then average rates across permutations.

    Returns thresholds per fold and pooled metrics per permutation. Thresholds
    are never averaged to produce a deployment model.
    """
    fold_results = []
    totals = {}
    for run, cal, test in calibration_splits(len(predictions), folds, permutations, seed):
        calibration = [predictions[i] for i in cal]
        evaluation = [predictions[i] for i in test]
        cp = calibrate_cp(calibration, annotations, cp_alpha, iou_threshold)
        cf = calibrate_filtering(calibration, annotations, cp, target_recall, target_precision, delta,
                                 iou_threshold=iou_threshold)
        metrics = evaluate_filtering(evaluation, annotations, cf, iou_threshold)
        label_metrics = evaluate_cp(evaluation, annotations, cp, iou_threshold)
        fold_results.append({"permutation": run, "calibration_ids": [p.image_id for p in calibration],
                             "evaluation_ids": [p.image_id for p in evaluation],
                             "tau": cp.threshold, "lambda_r": cf.recall.threshold,
                             "lambda_p": cf.precision.threshold, "metrics": metrics})
        pooled = totals.setdefault(run, dict(covered=0, matched=0, size=0, candidate_tp=0,
                                            confident_tp=0, confident_count=0, gt=0))
        for key, value in {
            "covered": label_metrics["covered"], "matched": label_metrics["matched_pairs"],
            "size": label_metrics["set_size_sum"], "candidate_tp": metrics["candidate"]["tp"],
            "confident_tp": metrics["confident"]["tp"], "confident_count": metrics["confident"]["retained"],
            "gt": metrics["candidate"]["total_gt"],
        }.items():
            pooled[key] += value
    per_run = []
    for run, t in totals.items():
        per_run.append({"permutation": run, "coverage": t["covered"] / t["matched"] if t["matched"] else None,
                        "mean_set_size": t["size"] / t["matched"] if t["matched"] else None,
                        "candidate_recall": t["candidate_tp"] / t["gt"] if t["gt"] else None,
                        "confident_precision": t["confident_tp"] / t["confident_count"] if t["confident_count"] else None})
    mean = {k: float(np.mean([r[k] for r in per_run if r[k] is not None]))
            if any(r[k] is not None for r in per_run) else None
            for k in ("coverage", "mean_set_size", "candidate_recall", "confident_precision")}
    return {"folds": fold_results, "permutations": per_run, "mean": mean}
