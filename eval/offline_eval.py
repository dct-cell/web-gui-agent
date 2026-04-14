"""Offline evaluation on Mind2Web test set.

Metrics:
- Element Accuracy: did the model predict the correct element position?
- Operation F1: is the action type correct?
- Step Success Rate: is the full step correct (element + op + value)?
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

from model.config import InferenceConfig
from model.inference import WebGUIModel
from agent.action_parser import parse_action


def compute_position_match(pred_pos: list[float], gold_pos: list[float], threshold: float = 0.05) -> bool:
    """Check if predicted position is within threshold of gold position."""
    if pred_pos is None or gold_pos is None:
        return False
    dx = abs(pred_pos[0] - gold_pos[0])
    dy = abs(pred_pos[1] - gold_pos[1])
    return dx <= threshold and dy <= threshold


def evaluate(
    model: WebGUIModel,
    eval_data_path: str,
    max_samples: int = 0,
) -> dict:
    """Run offline evaluation and return metrics."""
    counters = defaultdict(int)

    with open(eval_data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break

            record = json.loads(line)
            conv = record["conversations"]

            gold_text = conv[1]["content"]
            try:
                gold_action = json.loads(gold_text)
            except json.JSONDecodeError:
                continue

            messages = [{"role": "user", "content": [{"type": "text", "text": conv[0]["content"]}]}]
            pred_action = model.predict(messages)

            counters["total"] += 1

            if pred_action is None:
                continue

            op_match = pred_action.get("action") == gold_action.get("action")
            if op_match:
                counters["op_correct"] += 1

            pos_match = compute_position_match(
                pred_action.get("position"),
                gold_action.get("position"),
            )
            if pos_match:
                counters["pos_correct"] += 1

            value_match = pred_action.get("value", "") == gold_action.get("value", "")
            if op_match and pos_match and value_match:
                counters["step_correct"] += 1

    total = max(counters["total"], 1)
    return {
        "total_samples": counters["total"],
        "element_accuracy": counters["pos_correct"] / total,
        "operation_f1": counters["op_correct"] / total,
        "step_success_rate": counters["step_correct"] / total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = InferenceConfig(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
    )
    model = WebGUIModel(config)
    metrics = evaluate(model, args.eval_data, args.max_samples)

    print("\n=== Offline Evaluation Results ===")
    print(f"Total samples:      {metrics['total_samples']}")
    print(f"Element Accuracy:   {metrics['element_accuracy']:.4f}")
    print(f"Operation F1:       {metrics['operation_f1']:.4f}")
    print(f"Step Success Rate:  {metrics['step_success_rate']:.4f}")
