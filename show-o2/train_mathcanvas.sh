#!/bin/bash

# Training script for MathCanvas dataset with Show-o2 + LoRA
# Usage: bash train_mathcanvas.sh

# Memory optimization for CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set the path to your MathCanvas parquet file
MATHCANVAS_PARQUET="/content/calculus_and_vector.parquet"

# Set the output directory
OUTPUT_DIR="./outputs/showo2-mathcanvas-lora"

# Number of GPUs (adjust as needed)
NUM_GPUS=1

# LoRA settings (can override config here)
LORA_R=16
LORA_ALPHA=32

# Training settings for 40GB A100
BATCH_SIZE=1
GRAD_ACCUM=8  # Effective batch size = BATCH_SIZE * GRAD_ACCUM = 8

# Run training with LoRA
accelerate launch --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    train_mathcanvas.py \
    config=configs/showo2_1.5b_mathcanvas.yaml \
    dataset.mathcanvas.parquet_path="${MATHCANVAS_PARQUET}" \
    experiment.output_dir="${OUTPUT_DIR}" \
    model.showo.lora.enabled=True \
    model.showo.lora.r=${LORA_R} \
    model.showo.lora.alpha=${LORA_ALPHA} \
    model.gradient_checkpointing=True \
    training.batch_size_mixed_modal=${BATCH_SIZE} \
    training.gradient_accumulation_steps=${GRAD_ACCUM} \
    training.max_train_steps=50000 \
    dataset.preprocessing.max_mixed_modal_seq_length=2048 \
    experiment.save_every=1000 \
    experiment.log_every=10

echo "Training completed!"
