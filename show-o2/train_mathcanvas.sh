#!/bin/bash

# Training script for MathCanvas dataset with Show-o2
# Usage: bash train_mathcanvas.sh

# Set the path to your MathCanvas parquet file
MATHCANVAS_PARQUET="/Users/denizakkoc/Desktop/master donem1/research/MathCanvas-Instruct-PlaneGeometry/plane_geometry.parquet"

# Set the output directory
OUTPUT_DIR="./outputs/showo2-mathcanvas"

# Number of GPUs (adjust as needed)
NUM_GPUS=1

# Run training
accelerate launch --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    train_mathcanvas.py \
    config=configs/showo2_1.5b_mathcanvas.yaml \
    dataset.mathcanvas.parquet_path="${MATHCANVAS_PARQUET}" \
    experiment.output_dir="${OUTPUT_DIR}" \
    training.batch_size_mixed_modal=2 \
    training.gradient_accumulation_steps=4 \
    training.max_train_steps=50000

echo "Training completed!"

