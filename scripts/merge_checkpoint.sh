#!/bin/bash
set -e

EXP_DIR="${1:?Please provide experiment directory}"
SHOWUI_DIR="third_party/ShowUI"

CKPT_DIR="${EXP_DIR}/ckpt_model"
MERGE_DIR="${CKPT_DIR}/merged_model"

echo "=== Merging checkpoint ==="
echo "Experiment dir: $EXP_DIR"

cd "$CKPT_DIR"
python zero_to_fp32.py . pytorch_model.bin

mkdir -p merged_model

cd "$SHOWUI_DIR"
python merge_weight.py --exp_dir="$EXP_DIR"

echo "=== Merged model saved to: $MERGE_DIR ==="
