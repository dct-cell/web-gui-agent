"""Configuration for model training and inference."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    # Model
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    model_version: str = "Qwen/Qwen2-VL-2B-Instruct"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128

    # Training
    epochs: int = 50
    steps_per_epoch: int = 100
    batch_size: int = 1
    grad_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    precision: str = "bf16"
    model_max_length: int = 8192

    # Data
    dataset_dir: str = "./datasets"
    train_dataset: str = "showui-web"
    train_json: str = "hf_train"
    val_dataset: str = "screenspot"
    val_json: str = "hf_test"
    train_ratio: str = "1"

    # Visual tokens
    min_visual_tokens: int = 256
    max_visual_tokens: int = 1344

    # Navigation
    num_history: int = 2
    interleaved_history: str = "tttt"

    # Output
    log_base_dir: str = "./checkpoints"
    exp_id: str = "web-gui-agent"

    # DeepSpeed
    ds_zero: str = "zero2"

    # ShowUI specific
    lm_skip_ratio: float = 0.5
    lm_skip_layer: str = "[1,28,0]"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InferenceConfig:
    model_path: str = ""  # path to merged checkpoint or base model
    lora_path: str = ""  # path to LoRA weights (if not merged)
    device: str = "cuda"
    max_new_tokens: int = 256
    screenshot_width: int = 1344
    screenshot_height: int = 756
