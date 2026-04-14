#!/bin/bash
set -e

SHOWUI_DIR="third_party/ShowUI"
DATA_DIR="${DATA_DIR:-./datasets}"
SAVE_DIR="${SAVE_DIR:-./checkpoints}"
WANDB_KEY="${WANDB_KEY:-}"
NUM_GPUS="${NUM_GPUS:-1}"
GROUNDING_CKPT="${GROUNDING_CKPT:-Qwen/Qwen2-VL-2B-Instruct}"

echo "=== Starting Navigation Training ==="
echo "Data dir: $DATA_DIR"
echo "Save dir: $SAVE_DIR"
echo "Base model/ckpt: $GROUNDING_CKPT"

cd "$SHOWUI_DIR"

deepspeed --include "localhost:$(seq -s, 0 $((NUM_GPUS-1)))" --master_port 5678 train.py \
  --wandb_key="$WANDB_KEY" \
  --model_id="$GROUNDING_CKPT" \
  --version="$GROUNDING_CKPT" \
  --dataset_dir="$DATA_DIR" \
  --log_base_dir="$SAVE_DIR" \
  --epochs=50 \
  --steps_per_epoch=100 \
  --batch_size=1 \
  --grad_accumulation_steps=2 \
  --model_max_length=8192 \
  --exp_id="navigation-web" \
  --train_ratio="1" \
  --train_dataset="mind2web" \
  --train_json="hf_train" \
  --val_dataset="mind2web" \
  --val_json="hf_test_full" \
  --precision="bf16" \
  --attn_imple="sdpa" \
  --workers=0 \
  --lora_r=64 \
  --lora_alpha=128 \
  --min_visual_tokens=1344 \
  --max_visual_tokens=1680 \
  --num_turn=100 \
  --random_sample \
  --record_sample \
  --lr=0.0001 \
  --uniform_prompt \
  --ds_zero="zero2" \
  --gradient_checkpointing \
  --lm_skip_ratio=0.5 \
  --lm_skip_layer='[1,28,0]' \
  --num_history=2 \
  --interleaved_history='tttt'

echo "=== Navigation Training Complete ==="
