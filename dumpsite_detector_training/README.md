# Generative augmentation for dumping-site detection

This directory provides code for  data augmentation, object-detector training and evaluation on remote-sensing images of agricultural, construction and domestic waste dumping sites.

## 1. Installation

Run all commands below from the `detection/` directory. Use Python 3.10 and separate environments for SD/LoRA and the different detector backends.

From the repository root:

```bash
cd dumpsite_detector_training
python -m pip install -r requirements-demo.txt
```

For SD/LoRA, install a compatible PyTorch build and `environments/augmentation.txt`. For MMDetection 3:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
python -m pip install -r environments/mmdet3.txt
```

See `environments/` for the other backends and `configs/models.json` for the 18 detector configurations.

## 2. Data preparation

Organize images and annotations in PASCAL VOC format:

```text
VOC2007/
  JPEGImages/
  Annotations/
  ImageSets/Main/train.txt
  ImageSets/Main/val.txt
  ImageSets/Main/test.txt
```

Each split file contains one image ID per line, without the extension. Class names are `Agricultural`, `Construction` and `Domestic`. Keep training, validation and test sources separate. Use only training images for crop extraction, LoRA fine-tuning and augmentation; keep all derivatives of a source image in the same partition.

```bash
python -m src.augment prepare --voc data/VOC2007 --split data/VOC2007/ImageSets/Main/train.txt --output outputs/crops
```

## 3. Data augmentation

The four groups are `baseline` (original data), `balanced` (class-balanced resampling), `color_jitter` and `sd_lora`. The augmentation groups use a shared sampling manifest. The supplied manifest is `manifests/augmentation_plan.csv`; for another dataset, create a matching manifest using `python -m src.augment plan --help`.

To fine-tune SD with training crops:

```bash
BASE_MODEL=/models/stable-diffusion-v1-5 PATCH_DATA=outputs/crops LORA_OUTPUT=outputs/lora bash augmentation/train_lora.sh
```

To generate class-specific patches, configure the LoRA adapter in `src.generate` and run:

```bash
python -m src.generate --base-model /models/stable-diffusion-v1-5 --manifest manifests/augmentation_plan.csv --output outputs/patches
```

For a remote base model, also specify its commit with `--revision`. Build the datasets using the interface employed by `demo.py`:

```python
from src.augment import build

voc = "data/VOC2007"
for group in ("baseline", "balanced", "color_jitter", "sd_lora"):
    build(voc, f"{voc}/ImageSets/Main/train.txt",
          "manifests/augmentation_plan.csv", "outputs/patches",
          f"outputs/{group}/VOC2007", group,
          [(s, f"{voc}/ImageSets/Main/{s}.txt") for s in ("val", "test")])
```

Synthetic samples are combined with the original training data. Retain annotations for co-occurring objects and count images and object instances separately.

## 4. Detector training

Select a model from `configs/models.json` and provide the prepared dataset:

```bash
python train.py --model retinanet_x101 --group sd_lora --voc outputs/sd_lora/VOC2007 --work-dir outputs/train-retinanet --fresh --dry-run
```

Inspect the exported configuration, then remove `--dry-run` to train. Use `--fresh` for a new run or `--checkpoint PATH` for detector-weight initialization. Training settings are defined in the model configurations; `--epochs`, `--batch` and `--workers` provide overrides. `--group` records the dataset group and does not change online augmentation.

For Ultralytics, first convert the prepared VOC dataset with `src.convert_yolo.convert`, as shown in `demo.py`, and pass the converted directory to `--voc`.

## 5. Evaluation

Run the trained detector on the test split and export predictions with columns `image_id,class,score,xmin,ymin,xmax,ymax`. Use the same class names and coordinate convention as the annotations.

```bash
python -m src.evaluate --voc data/VOC2007 --split data/VOC2007/ImageSets/Main/test.txt --predictions outputs/predictions.csv --output outputs/metrics.json
```

Evaluation retains detections with confidence ≥0.1 and applies score-ordered, same-class, one-to-one matching at IoU ≥0.5. The output reports macro and micro precision and recall.

## Small example

```bash
python verify.py --examples
python demo.py --output outputs/demo
```

The example uses pre-generated patches, builds all four groups and exports detector configurations without GPU training. Use a new output directory.

See `THIRD_PARTY_NOTICES.md` for attribution and licensing information.
