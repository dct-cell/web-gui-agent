# Web GUI Agent — GPU 训练指南

## 前置条件

- GPU 服务器（推荐 A100 80G 或 4090 24G+）
- CUDA 11.8+ 和 Python 3.10+
- 能访问 HuggingFace（下载模型和数据集）

---

## Step 1: 将项目拷贝到 GPU 服务器

```bash
# 方式一：直接拷贝
scp -r web-gui-agent/ user@gpu-server:/path/to/workspace/

# 方式二：如果已推送到 git
git clone <your-repo-url>
cd web-gui-agent
```

---

## Step 2: 安装环境

```bash
cd web-gui-agent

# 创建 conda 环境（推荐）
conda create -n webagent python=3.10
conda activate webagent

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8:
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1+:
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 运行 setup 脚本（安装依赖 + 克隆 ShowUI）
bash scripts/setup_showui.sh
```

---

## Step 3: 下载数据集

```bash
# 下载 Mind2Web、ShowUI-web、ScreenSpot
python data/download.py --data_dir ./datasets

# 预计下载大小：
#   Mind2Web: ~6.7 GB
#   ShowUI-web: ~2-3 GB
#   ScreenSpot: ~500 MB
```

---

## Step 4: 预处理数据

```bash
python data/preprocess.py --data_dir ./datasets --output_dir ./datasets/processed

# 产出文件：
#   ./datasets/processed/mind2web_navigation.jsonl
#   ./datasets/processed/showui_web_grounding.jsonl
```

---

## Step 5: 准备 ShowUI 数据目录结构

ShowUI 的训练代码期望特定的目录结构。需要做以下处理：

```bash
# 用 ShowUI 提供的脚本处理 Mind2Web 数据
cd third_party/ShowUI
python prepare/hf_mind2web.py --data_dir ../../datasets

# 确保目录结构如下：
# datasets/
#   ├── ShowUI-web/
#   │   ├── images/
#   │   └── metadata/
#   ├── Mind2Web/
#   │   ├── images/
#   │   └── metadata/
#   └── ScreenSpot/
#       ├── images/
#       └── metadata/

cd ../..  # 回到 web-gui-agent 根目录
```

如果 ShowUI 的 prepare 脚本报错，参考：https://github.com/showlab/ShowUI/blob/main/TRAIN.md

---

## Step 6: 训练阶段一 — GUI Grounding 预训练

```bash
# 设置环境变量
export DATA_DIR=./datasets
export SAVE_DIR=./checkpoints
export WANDB_KEY=你的wandb_key    # 可选，用于监控训练
export NUM_GPUS=1                  # 根据你的 GPU 数量调整

# 启动训练
bash scripts/train_grounding.sh

# 预计时间：
#   单卡 A100: ~12-16h
#   单卡 4090: ~24-36h（可能需要改用 QLoRA）
#   8卡 H20:  ~1-1.5h (设 NUM_GPUS=8, batch_size 可调大到 4)
#
# 训练过程中观察：
#   - WandB 面板上的 loss 曲线应该稳定下降
#   - 检查点保存在 ./checkpoints/grounding-web/<日期时间>/
```

**如果显存不足 (OOM)**，在 `scripts/train_grounding.sh` 中调整：
- 减小 `--max_visual_tokens` (比如 1024)
- 减小 `--lora_r` (比如 32)
- 添加 `--gradient_checkpointing`（已默认开启）

---

## Step 7: 评测 Grounding 效果

```bash
cd third_party/ShowUI

python main/eval_screenspot.py \
  --model_id Qwen/Qwen2-VL-2B-Instruct \
  --lora_path ../../checkpoints/grounding-web/<日期时间>/ckpt_model \
  --dataset_dir ../../datasets

# 目标：Web 子集准确率 60%+
# 如果低于预期，考虑增加训练 epoch 或数据量

cd ../..
```

---

## Step 8: 合并 Grounding Checkpoint

```bash
# 将 <日期时间> 替换为实际的目录名，如 2026-04-20_10-30-00
bash scripts/merge_checkpoint.sh ./checkpoints/grounding-web/<日期时间>

# 合并后的模型在：
# ./checkpoints/grounding-web/<日期时间>/ckpt_model/merged_model/
```

---

## Step 9: 训练阶段二 — Web Navigation 微调

```bash
# 指向 grounding 合并后的 checkpoint
export GROUNDING_CKPT=./checkpoints/grounding-web/<日期时间>/ckpt_model/merged_model
export DATA_DIR=./datasets
export SAVE_DIR=./checkpoints
export NUM_GPUS=1

bash scripts/train_navigation.sh

# 预计时间：
#   单卡 A100: ~8-12h
#   单卡 4090: ~16-24h
#   8卡 H20:  ~0.5-1h (设 NUM_GPUS=8)
```

---

## Step 10: 合并 Navigation Checkpoint

```bash
bash scripts/merge_checkpoint.sh ./checkpoints/navigation-web/<日期时间>

# 最终模型路径：
# ./checkpoints/navigation-web/<日期时间>/ckpt_model/merged_model/
```

---

## Step 11: 离线评测

```bash
PYTHONPATH=. python eval/offline_eval.py \
  --model_path ./checkpoints/navigation-web/<日期时间>/ckpt_model/merged_model \
  --eval_data ./datasets/processed/mind2web_navigation.jsonl \
  --max_samples 100

# 输出示例：
# === Offline Evaluation Results ===
# Total samples:      100
# Element Accuracy:   0.xxxx
# Operation F1:       0.xxxx
# Step Success Rate:  0.xxxx
```

---

## Step 12: 在线端到端评测

```bash
PYTHONPATH=. python eval/online_eval.py \
  --model_path ./checkpoints/navigation-web/<日期时间>/ckpt_model/merged_model \
  --output_dir ./eval_results

# 会运行 5 个预定义任务（Wikipedia、Google、百度等）
# 每步截图保存在 ./eval_results/<task_id>/
# 结果 JSON 保存在 ./eval_results/results_<时间戳>.json
```

---

## Step 13: 启动 Gradio Demo

```bash
PYTHONPATH=. python demo/app.py \
  --model_path ./checkpoints/navigation-web/<日期时间>/ckpt_model/merged_model \
  --port 7860 \
  --share  # 加上 --share 生成公网链接，方便展示

# 打开 http://localhost:7860 或公网链接
# 输入 URL + 指令，观看 Agent 自动操作浏览器
```

---

## 消融实验（可选，用于课程报告）

在训练时调整参数，对比效果：

```bash
# 实验 1: 不同 LoRA rank
# 修改 scripts/train_navigation.sh 中的 --lora_r 和 --lora_alpha
# rank=16, alpha=32
# rank=32, alpha=64
# rank=64, alpha=128（默认）

# 实验 2: 有无 Grounding 预训练
# 直接跳过 Step 6-8，用原始 Qwen2-VL-2B 跑 Step 9
# 对比有无 grounding 预训练的 navigation 效果

# 实验 3: 有无历史截图
# 修改 --num_history=0 vs --num_history=2

# 实验 4: 不同数据量
# 修改 --steps_per_epoch 和 --epochs 控制训练量
```

---

## 常见问题

**Q: OOM (显存不足)**
A: 用 QLoRA：修改训练脚本，添加 `--bits 4`。或减小 `--max_visual_tokens`。

**Q: ShowUI 的 prepare 脚本报错**
A: 参考 ShowUI 的 TRAIN.md，确保数据目录结构正确。也可以直接用我们 preprocess.py 的输出。

**Q: 训练 loss 不降**
A: 检查学习率（1e-4 是默认值，可以尝试 5e-5）和数据格式是否正确。

**Q: Demo 中 Agent 动作不准**
A: 先用离线评测确认模型效果。如果离线指标 OK 但在线不行，可能是 prompt 格式不匹配，检查 prompt_builder.py 的格式是否和训练数据一致。
