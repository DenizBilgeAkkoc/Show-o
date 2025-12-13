#!/bin/bash

# Training script for Show-o2 7B with MathCanvas dataset
# Optimized for 2× A40 (48GB each) GPUs
# Usage: bash train_mathcanvas_7b.sh

set -e  # Exit on error

# ===== CONFIGURATION =====
# Path to MathCanvas data - can be:
#   - A single parquet file: /path/to/all_data.parquet
#   - A directory containing multiple parquet files: /path/to/data/train/
#   - The dataset will automatically discover and load all .parquet files in a directory
MATHCANVAS_PARQUET="/path/to/MathCanvas-Instruct/data/train"

# Output directory
OUTPUT_DIR="./outputs/showo2-7b-mathcanvas-lora"

# Number of GPUs
NUM_GPUS=2

# LoRA settings (higher for 7B model)
LORA_R=32
LORA_ALPHA=64

# Training settings for 2× A40
BATCH_SIZE=1
GRAD_ACCUM=8  # Effective batch size = 1 * 2 GPUs * 8 = 16

# Max training steps (~3 epochs for 218K samples)
MAX_STEPS=82000

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== TRAINING =====
echo "Starting Show-o2 7B training on ${NUM_GPUS} GPUs..."
echo "Dataset: ${MATHCANVAS_PARQUET}"
echo "Output: ${OUTPUT_DIR}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --use_deepspeed \
    --deepspeed_config_file=ds_config_7b_zero2.json \
    train_mathcanvas.py \
    config=configs/showo2_7b_mathcanvas.yaml \
    dataset.mathcanvas.parquet_path="${MATHCANVAS_PARQUET}" \
    experiment.output_dir="${OUTPUT_DIR}" \
    model.showo.lora.enabled=True \
    model.showo.lora.r=${LORA_R} \
    model.showo.lora.alpha=${LORA_ALPHA} \
    model.gradient_checkpointing=True \
    training.batch_size_mixed_modal=${BATCH_SIZE} \
    training.gradient_accumulation_steps=${GRAD_ACCUM} \
    training.max_train_steps=${MAX_STEPS} \
    experiment.save_every=10000 \
    experiment.log_every=50

echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
