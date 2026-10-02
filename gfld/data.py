"""Portable, NumPy-only inputs for post-processing saved predictions."""
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import os
import hashlib
import xml.etree.ElementTree as ET
import numpy as np

CLASSES = ("domestic garbage", "construction waste", "agriculture forestry")


@dataclass(frozen=True)
class Detections:
    image_id: str
    boxes: np.ndarray
    scores: np.ndarray
    class_scores: np.ndarray
    labels: np.ndarray

    def __post_init__(self):
        n = len(self.scores)
        if self.boxes.shape != (n, 4) or self.labels.shape != (n,):
            raise ValueError(f"{self.image_id}: inconsistent prediction shapes")
        if self.class_scores.ndim != 2 or self.class_scores.shape[0] != n:
            raise ValueError(f"{self.image_id}: class_scores must have shape (N, C)")
        if self.class_scores.shape[1] < 1:
            raise ValueError("At least one class is required")
        for x in (self.boxes, self.scores, self.class_scores):
            if not np.isfinite(x).all():
                raise ValueError(f"{self.image_id}: non-finite prediction values")
        if (self.boxes[:, 2:] < self.boxes[:, :2]).any():
            raise ValueError(f"{self.image_id}: inverted boxes")
        if ((self.scores < 0) | (self.scores > 1)).any():
            raise ValueError("Detection confidence must be in [0, 1]")
        if ((self.class_scores < 0) | (self.class_scores > 1)).any():
            raise ValueError("Class scores must be in [0, 1]")
        if ((self.labels < 0) | (self.labels >= self.class_scores.shape[1])).any():
            raise ValueError("Prediction label outside class order")

    def select(self, mask):
        return Detections(self.image_id, self.boxes[mask], self.scores[mask],
                          self.class_scores[mask], self.labels[mask])


@dataclass(frozen=True)
class Annotation:
    boxes: np.ndarray
    labels: np.ndarray

    def __post_init__(self):
        if self.boxes.shape != (len(self.labels), 4):
            raise ValueError("Annotation boxes and labels are misaligned")
        if not np.isfinite(self.boxes).all() or (self.boxes[:, 2:] < self.boxes[:, :2]).any():
            raise ValueError("Invalid annotation coordinates")


def load_predictions(path, num_classes=3):
    """Read a trusted NumPy-backed PKL; raw tensor PKLs require prior export.

    Preserve sigmoid class scores. Remove and renormalize the last column only
    when it is an explicit extra background class.
    """
    with Path(path).open("rb") as f:
        records = pickle.load(f)
    result = []
    seen = set()
    for entry in records:
        image_id = Path(str(entry.get("img_id", entry.get("img_path", "")))).stem
        if not image_id or image_id in seen:
            raise ValueError(f"Missing or duplicate image ID: {image_id}")
        seen.add(image_id)
        p = entry["pred_instances"]
        probs = np.asarray(p["cls_probs"], dtype=float).copy()
        if probs.ndim != 2:
            raise ValueError("cls_probs must be a matrix")
        if probs.shape[1] == num_classes + 1:
            probs = probs[:, :-1]
            sums = probs.sum(axis=1, keepdims=True)
            probs /= np.where(sums == 0, 1, sums)
        elif probs.shape[1] != num_classes:
            raise ValueError("Class score columns do not match the class order")
        result.append(Detections(image_id, np.asarray(p["bboxes"], dtype=float).reshape(-1, 4),
                                 np.asarray(p["scores"], dtype=float), probs,
                                 np.asarray(p["labels"], dtype=int)))
    return result


def load_annotations(path, num_classes=3):
    data = json.loads(Path(path).read_text())
    result = {}
    for image_id, d in data.items():
        labels = np.asarray(d["labels"], dtype=int)
        if ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError(f"{image_id}: annotation label outside class order")
        result[image_id] = Annotation(np.asarray(d["boxes"], dtype=float).reshape(-1, 4), labels)
    return result


def load_features(path):
    with np.load(path, allow_pickle=False) as f:
        ids, features = f["image_ids"].astype(str), f["features"]
        if features.ndim != 2 or len(ids) != len(features) or len(set(ids)) != len(ids):
            raise ValueError("Feature IDs and vectors are misaligned or duplicated")
        if not np.isfinite(features).all():
            raise ValueError("Non-finite features")
        return dict(zip(ids, features))


def feature_matrix(features, image_ids):
    missing = [i for i in image_ids if i not in features]
    if missing:
        raise ValueError(f"Missing features for {missing[:5]}")
    return np.stack([features[i] for i in image_ids])


def load_voc_annotations(directory, image_ids, classes=CLASSES):
    """Read VOC XMLs without depending on a detector framework.

    Convert coordinates to integers and exclude difficult objects.
    Missing XMLs are reported as errors.
    """
    mapping = {name: i for i, name in enumerate(classes)}
    result = {}
    for image_id in image_ids:
        root = ET.parse(Path(directory) / f"{image_id}.xml").getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            if obj.findtext("difficult") == "1":
                continue
            label = " ".join(obj.findtext("name").lower().split())
            if label not in mapping:
                raise ValueError(f"{image_id}: unknown annotation class {label!r}")
            labels.append(mapping[label])
            box = obj.find("bndbox")
            boxes.append([int(float(box.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")])
        result[image_id] = Annotation(np.array(boxes, dtype=float).reshape(-1, 4), np.array(labels, dtype=int))
    return result


def example_directory(repository_root, model="retinanet_x101"):
    """GFLD_DATA_DIR points to a dataset root containing the model and XML tree."""
    data_root = Path(os.environ.get("GFLD_DATA_DIR", Path(repository_root) / "data")).expanduser()
    directory = data_root / model
    if not (directory / "metadata.json").is_file():
        raise FileNotFoundError(
            f"Example data not found at {directory}. Set GFLD_DATA_DIR to the prepared "
            "dataset folder, or place the downloaded files in the repository's data/ folder.")
    return directory


def verify_manifest(directory):
    """Verify all dataset file hashes before reading trusted prediction PKLs."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = (directory / name).resolve()
        if directory not in path.parents or not path.is_file():
            raise ValueError(f"Invalid or missing manifest file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(manifest["sha256"])


def load_example(directory):
    """Load a local model folder; external annotations remain evaluation-only.

    Schema v2 uses shared VOC XMLs beside the model directory.
    JSON annotation bundles are also supported.
    """
    directory = Path(directory)
    result = {"metadata": json.loads((directory / "metadata.json").read_text())}
    for split in ("internal", "external"):
        classes = result["metadata"].get("classes", CLASSES)
        result[split] = load_predictions(directory / f"{split}.pkl", len(classes))
        ids = [p.image_id for p in result[split]]
        split_meta = result["metadata"]["splits"][split]
        if "annotation_dir" in split_meta:
            result[f"{split}_annotations"] = load_voc_annotations(directory / split_meta["annotation_dir"], ids, classes)
        else:
            result[f"{split}_annotations"] = load_annotations(directory / f"{split}_annotations.json", len(classes))
        result[f"{split}_features"] = load_features(directory / f"{split}_features.npz")
        if set(ids) != set(result[f"{split}_annotations"]) or set(ids) != set(result[f"{split}_features"]):
            raise ValueError(f"{split}: predictions, annotations, and feature IDs must agree")
        if len(ids) != split_meta["images"]:
            raise ValueError(f"{split}: image count differs from metadata")
        counts = {"predictions": sum(len(p.scores) for p in result[split]),
                  "ground_truth": sum(len(a.labels) for a in result[f"{split}_annotations"].values())}
        for name, count in counts.items():
            if count != split_meta[name]:
                raise ValueError(f"{split}: {name} count differs from metadata")
        if any(len(v) != split_meta["feature_dimension"] for v in result[f"{split}_features"].values()):
            raise ValueError(f"{split}: feature dimension differs from metadata")
    return result


def calibration_splits(n, folds=5, permutations=4, seed=42, calibration_ratio=0.5):
    """Yield (permutation, calibration indices, evaluation indices) by image."""
    if not 1 <= folds <= n or permutations < 1 or not 0 < calibration_ratio < 1:
        raise ValueError("Invalid split configuration")
    for run in range(permutations):
        order = np.random.default_rng(seed + run).permutation(n)
        if folds == 1:
            cut = int(round(n * calibration_ratio))
            if cut == 0 or cut == n:
                raise ValueError("Calibration and evaluation must both be nonempty")
            yield run, order[:cut], order[cut:]
        else:
            parts = np.array_split(order, folds)
            for k in range(folds):
                yield run, np.concatenate([p for j, p in enumerate(parts) if j != k]), parts[k]
