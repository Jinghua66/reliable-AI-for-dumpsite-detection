# Provenance and licensing

`third_party/train_text_to_image_lora.py` is the author's archived Hugging Face
Diffusers training script. Its Apache-2.0 copyright/license header is retained.
See `third_party/APACHE-2.0.txt` and https://github.com/huggingface/diffusers.

Detector configurations derive from the author's local MMDetection experiment
files (MMDetection is Apache-2.0; https://github.com/open-mmlab/mmdetection).
`custom/bca_backbone.py` is the recovered local BCA implementation; only two
relative imports were changed to absolute imports. Its origin/hash is recorded
in `configs/provenance.json`. No claim of a new blanket license over third-party
code, images or weights is made here.

Ultralytics is an external dependency with its own license terms:
https://github.com/ultralytics/ultralytics/tree/v8.3.4.
No Ultralytics framework implementation is vendored in this repository.

Example image redistribution was explicitly authorized by the author. That
authorization is not a declaration that the imagery is public domain or that
all imagery in the source dataset may be redistributed. Retain source attribution
when reusing examples. The original full dataset is described in the manuscript.

The authors should select their desired license for author-owned additions before
advertising this repository as generally licensed open-source software. Until
then, no additional broad redistribution rights are granted by this notice.
