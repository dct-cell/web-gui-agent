#!/bin/bash
set -e

SHOWUI_DIR="third_party/ShowUI"
DATA_DIR="${DATA_DIR:-./datasets}"
SAVE_DIR="${SAVE_DIR:-./checkpoints}"
WANDB_KEY="${WANDB_KEY:-}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "=== Starting Grounding Training ==="
echo "Data dir: $DATA_DIR"
echo "Save dir: $SAVE_DIR"
echo "GPUs: $NUM_GPUS"

cd "$SHOWUI_DIR"

deepspeed --include "localhost:$(seq -s, 0 $((NUM_GPUS-1)))" --master_port 5678 train.py \
  --wandb_key="$WANDB_KEY" \
  --model_id='Qwen/Qwen2-VL-2B-Instruct' \
  --version='Qwen/Qwen2-VL-2B-Instruct' \
  --dataset_dir="$DATA_DIR" \
  --log_base_dir="$SAVE_DIR" \
  --epochs=50 \
  --steps_per_epoch=100 \
  --batch_size=1 \
  --grad_accumulation_steps=2 \
  --model_max_length=8192 \
  --exp_id="grounding-web" \
  --train_ratio="1" \
  --train_dataset="showui-web" \
  --train_json="hf_train" \
  --val_dataset="screenspot" \
  --precision="bf16" \
  --attn_imple="sdpa" \
  --workers=0 \
  --lora_r=64 \
  --lora_alpha=128 \
  --min_visual_tokens=256 \
  --max_visual_tokens=1344 \
  --num_turn=100 \
  --crop_min=0.5 \
  --crop_max=1.5 \
  --random_sample \
  --record_sample \
  --lr=0.0001 \
  --uniform_prompt \
  --ds_zero="zero2" \
  --gradient_checkpointing \
  --lm_skip_ratio=0.5 \
  --lm_skip_layer='[1,28,0]'

echo "=== Grounding Training Complete ==="
