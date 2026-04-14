# CLAUDE.md

## 项目概述

**Web GUI Agent** — 基于视觉语言模型 (VLM) 的网页操作智能体。Deep Learning 课程 Project。

核心能力：给定网页截图 + 自然语言指令（如"在 Wikipedia 搜索 Transformer 并导航到 Architecture 章节"），Agent 自动操控浏览器完成任务。

## 技术方案

- **基座模型**: Qwen2-VL-2B + LoRA 微调
- **训练两阶段**: GUI Grounding 预训练（学定位元素）→ Web Navigation 微调（学多步操作）
- **推理循环**: 截图 → VLM 预测动作 → Playwright 执行 → 截新图 → 循环
- **参考论文**: ShowUI (CVPR 2025), SeeClick, Mind2Web (NeurIPS 2023)

## 项目结构

```
web-gui-agent/
├── agent/                  # Agent 执行引擎
│   ├── action_parser.py    # 模型输出 → 结构化动作 (click/type/scroll/done)
│   ├── prompt_builder.py   # 截图+历史+指令 → Qwen2-VL 消息格式
│   ├── browser.py          # Playwright 浏览器封装 (归一化坐标 0-1)
│   └── executor.py         # 截图→推理→执行 主循环 (TaskExecutor)
├── model/
│   ├── config.py           # TrainConfig + InferenceConfig 数据类
│   └── inference.py        # Qwen2-VL 加载 + LoRA 合并 + predict()
├── data/
│   ├── download.py         # 从 HuggingFace 下载 Mind2Web / ShowUI-web / ScreenSpot
│   └── preprocess.py       # 转换为统一训练格式 (JSONL)
├── eval/
│   ├── offline_eval.py     # Mind2Web 离线评测 (Element Acc / Op F1 / Step SR)
│   └── online_eval.py      # 5 个预定义任务的端到端评测
├── demo/
│   └── app.py              # Gradio Demo (实时截图流 + 动作标注 + 执行日志)
├── configs/
│   ├── train_grounding.yaml
│   └── train_navigation.yaml
├── scripts/
│   ├── setup_showui.sh     # 环境搭建 (克隆 ShowUI + 安装依赖)
│   ├── train_grounding.sh  # 阶段一训练启动
│   ├── train_navigation.sh # 阶段二训练启动
│   └── merge_checkpoint.sh # 合并 LoRA 权重
├── tests/                  # 18 个测试 (action_parser, preprocess, executor, prompt_builder)
├── GPU_TRAINING_GUIDE.md   # GPU 训练完整步骤指南
├── requirements.txt
└── docs/superpowers/
    ├── specs/              # 设计文档
    └── plans/              # 实现计划
```

## 当前进度

- [x] 代码框架全部完成 (Tasks 1-11)
- [x] 18/18 测试通过
- [ ] **下一步: 在 GPU 服务器上训练** — 详见 `GPU_TRAINING_GUIDE.md`
- [ ] 训练后运行评测 (offline + online)
- [ ] 启动 Gradio Demo 验证端到端效果

## 关键命令

```bash
# 环境搭建
bash scripts/setup_showui.sh

# 下载数据
python data/download.py --data_dir ./datasets

# 预处理
python data/preprocess.py --data_dir ./datasets --output_dir ./datasets/processed

# 训练 (需要 GPU)
NUM_GPUS=8 bash scripts/train_grounding.sh       # 阶段一
bash scripts/merge_checkpoint.sh ./checkpoints/grounding-web/<date>
GROUNDING_CKPT=<merged_path> NUM_GPUS=8 bash scripts/train_navigation.sh  # 阶段二
bash scripts/merge_checkpoint.sh ./checkpoints/navigation-web/<date>

# 评测
PYTHONPATH=. python eval/offline_eval.py --model_path <model> --eval_data ./datasets/processed/mind2web_navigation.jsonl
PYTHONPATH=. python eval/online_eval.py --model_path <model>

# Demo
PYTHONPATH=. python demo/app.py --model_path <model> --port 7860 --share

# 测试
PYTHONPATH=. python -m pytest tests/ -v
```

## 动作空间

| 动作 | 格式 | 说明 |
|------|------|------|
| click | `{"action": "click", "position": [x, y]}` | 归一化坐标 (0-1) |
| type | `{"action": "type", "value": "text", "position": [x, y]}` | 点击后输入 |
| scroll | `{"action": "scroll", "value": "up"/"down"}` | 滚动 |
| done | `{"action": "done", "answer": "..."}` | 任务完成 |

## 注意事项

- 训练脚本中 `NUM_GPUS` 默认为 1，多卡时通过环境变量设置
- 训练使用 ShowUI 的训练框架 (`third_party/ShowUI/`)，我们自己的代码负责数据处理、推理引擎和 Demo
- 截图分辨率固定 1344×756，坐标归一化到 [0, 1]
