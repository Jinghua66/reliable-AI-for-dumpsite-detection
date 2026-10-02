"""Post-processing for saved object detector predictions."""
from .data import (Annotation, Detections, CLASSES, load_predictions, load_annotations,
                   load_features, load_example, calibration_splits, load_voc_annotations,
                   example_directory, verify_manifest)
from .matching import box_iou, match_boxes
from .conformal import LabelSets, calibrate_cp
from .filtering import Filtering, calibrate_filtering
from .covariate_shift import DensityRatio, calibrate_shift_cp, calibrate_shift_filtering
from .evaluation import evaluate_cp, evaluate_detections, evaluate_filtering, cross_validate

__version__ = "0.2.0"
__all__ = ["Annotation", "Detections", "CLASSES", "load_predictions", "load_annotations",
           "load_features", "load_example", "calibration_splits", "box_iou", "match_boxes",
           "LabelSets", "calibrate_cp", "Filtering", "calibrate_filtering", "DensityRatio",
           "calibrate_shift_cp", "calibrate_shift_filtering", "evaluate_cp", "evaluate_detections",
           "evaluate_filtering", "cross_validate", "load_voc_annotations", "example_directory", "verify_manifest"]
