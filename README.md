# GFLD

## From automated detection to trustworthy decision support: a reliable AI framework for monitoring solid waste dumping sites using remote sensing

[Installation](#installation) | [Getting started](#getting-started) | [Demo notebooks](#demo-notebooks) | 


## Installation

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -c constraints.txt -e '.[notebooks]'
```

## Getting started

Place the downloaded files in `data/`, with `retinanet_x101/` and `xml_labels/` as subfolders. See the paper for dataset details. For another location, set `GFLD_DATA_DIR` to the data folder.

```python
from pathlib import Path
from gfld import (
    example_directory, load_example, calibration_splits,
    calibrate_cp, calibrate_filtering, evaluate_filtering,
)

data = load_example(example_directory(Path.cwd()))
_, calibration_idx, evaluation_idx = next(
    calibration_splits(len(data["internal"]), folds=5, permutations=1, seed=42)
)
calibration = [data["internal"][i] for i in calibration_idx]
evaluation = [data["internal"][i] for i in evaluation_idx]
annotations = data["internal_annotations"]

cp = calibrate_cp(calibration, annotations, alpha=0.01)
filtering = calibrate_filtering(
    calibration, annotations, cp,
    target_recall=0.80, target_precision=0.85, delta=0.5,
)
regions = filtering.apply(evaluation)
metrics = evaluate_filtering(evaluation, annotations, filtering)
print(metrics["candidate"]["recall"], metrics["confident"]["precision"])
```

## Demo notebooks

Run `jupyter lab` and open a notebook below. Each tutorial runs independently.

| Notebook | Demonstration |
| --- | --- |
| [01 Bounding-box matching](notebooks/01_bbox_matching.ipynb) | Match predictions to annotations and visualize assignments |
| [02 Conformal prediction](notebooks/02_conformal_prediction.ipynb) | Calibrate label sets and evaluate coverage |
| [03 Conformal filtering](notebooks/03_conformal_filtering.ipynb) | Partition detections into confident, uncertain, and declined regions |
| [04 Covariate-shift CP](notebooks/04_covariate_shift_cp.ipynb) | Estimate image weights and calibrate weighted label sets |
| [05 Covariate-shift filtering](notebooks/05_covariate_shift_filtering.ipynb) | Apply weighted precision and recall calibration |

## Methods and reproducibility

Calibration and evaluation use separate image splits. Filtering calibrates recall on the candidate set and precision on the confident set. Shift-aware calibration uses internal annotations and external features; external annotations are used for evaluation.

Notebook parameters are explicit, with seed `42` and IoU threshold `0.5`. Notebook 03 includes five-fold cross-calibration over four permutations. Calibration results report thresholds and target feasibility alongside the evaluation metrics.

```bash
pip install -c constraints.txt -e '.[test]'
pytest -q
python scripts/check_notebooks.py
```

## Acknowledgments

The selective-calibration approach builds on [cascaded-selective-evaluation](https://github.com/jaehunjung1/cascaded-selective-evaluation). Repository presentation is inspired by [TITAN](https://github.com/mahmoodlab/TITAN), selective calibration(https://github.com/jaehunjung1/cascaded-selective-evaluation)
