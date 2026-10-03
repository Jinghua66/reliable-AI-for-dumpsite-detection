#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. BASE_MODEL must identify the exact local SD v1.5 snapshot.
: "${BASE_MODEL:?Set BASE_MODEL to the local SD v1.5 model directory}"
: "${PATCH_DATA:?Set PATCH_DATA to the crop directory containing metadata.jsonl}"
: "${LORA_OUTPUT:?Set LORA_OUTPUT to a new output directory}"
# Default: 8 processes, per-device batch 1, accumulation 4, BF16.
# The provided adapter uses rank 4; BF16 requires compatible GPU hardware.
accelerate launch --num_processes "${NPROC:-8}" --mixed_precision bf16 \
  third_party/train_text_to_image_lora.py \
  --pretrained_model_name_or_path "$BASE_MODEL" \
  --train_data_dir "$PATCH_DATA" --caption_column text \
  --dataloader_num_workers 8 --resolution 128 --center_crop --random_flip \
  --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_steps 15000 --learning_rate 1e-5 --max_grad_norm 1 \
  --lr_scheduler cosine --lr_warmup_steps 0 --rank 4 \
  --output_dir "$LORA_OUTPUT" --checkpointing_steps 5000 \
  --validation_prompt "domestic garbage dumpsite, municipal solid waste, remote sensing image, complicated component" \
  --seed 42
