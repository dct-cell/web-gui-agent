"""Convert raw datasets to unified training format for Qwen2-VL."""
import json
import argparse
from pathlib import Path
from typing import Optional


def parse_bbox(attributes_str: str) -> Optional[list[float]]:
    """Extract bounding_box_rect from Mind2Web attributes JSON string.

    Returns [left, top, right, bottom] as floats, or None if not found.
    """
    try:
        attrs = json.loads(attributes_str)
        bbox_str = attrs.get("bounding_box_rect", "")
        if not bbox_str:
            return None
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) == 4:
            return parts
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def normalize_position(bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """Convert pixel bbox [left, top, right, bottom] to normalized center [x, y]."""
    cx = (bbox[0] + bbox[2]) / 2.0 / img_width
    cy = (bbox[1] + bbox[3]) / 2.0 / img_height
    return [round(cx, 4), round(cy, 4)]


def convert_mind2web_sample(
    sample: dict, img_width: int = 1344, img_height: int = 756
) -> list[dict]:
    """Convert a Mind2Web sample to list of step dicts.

    Each step: {"task": str, "action": {"action": str, "value": str, "position": [x,y]}, "annotation_id": str, "action_uid": str}
    """
    task = sample["confirmed_task"]
    steps = []

    for action in sample["actions"]:
        op = action["operation"]["op"].upper()
        value = action["operation"].get("value", "")

        position = None
        for cand in action.get("pos_candidates", []):
            if cand.get("is_top_level_target"):
                bbox = parse_bbox(cand.get("attributes", "{}"))
                if bbox:
                    position = normalize_position(bbox, img_width, img_height)
                break

        if position is None:
            continue

        action_type = {"CLICK": "click", "TYPE": "type", "SELECT": "click"}.get(op)
        if action_type is None:
            continue

        step = {
            "task": task,
            "action": {
                "action": action_type,
                "position": position,
            },
            "annotation_id": sample["annotation_id"],
            "action_uid": action["action_uid"],
        }
        if action_type == "type" and value:
            step["action"]["value"] = value

        steps.append(step)

    return steps


def convert_showui_web_sample(sample: dict) -> list[dict]:
    """Convert a ShowUI-web grounding sample to list of element dicts.

    Each element: {"instruction": str, "position": [x, y], "bbox": [l, t, r, b], "data_type": str}
    """
    elements = []
    for elem in sample.get("element", []):
        point = elem.get("point")
        if point and len(point) == 2:
            elements.append({
                "instruction": elem["instruction"],
                "position": [round(point[0], 4), round(point[1], 4)],
                "bbox": elem.get("bbox"),
                "data_type": elem.get("data_type", "text"),
            })
    return elements


def build_grounding_conversation(
    instruction: str, position: list[float]
) -> list[dict]:
    """Build a grounding training conversation pair."""
    return [
        {"role": "user", "content": f"<image>In this web page screenshot, click on: {instruction}"},
        {"role": "assistant", "content": f"({position[0]}, {position[1]})"},
    ]


def build_navigation_conversation(
    task: str,
    history_actions: list[str],
    current_action: dict,
) -> list[dict]:
    """Build a navigation training conversation pair."""
    history_str = "\n".join(f"  Step {i+1}: {a}" for i, a in enumerate(history_actions))
    if not history_str:
        history_str = "  (none)"

    user_content = (
        f"<image>Task: {task}\n"
        f"Previous actions:\n{history_str}\n"
        f"What action should be performed next?"
    )
    assistant_content = json.dumps(current_action, ensure_ascii=False)

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def preprocess_mind2web(data_dir: str, output_dir: str):
    """Process full Mind2Web dataset and save as JSONL."""
    from datasets import load_from_disk

    ds = load_from_disk(str(Path(data_dir) / "mind2web" / "train"))
    output_path = Path(output_dir) / "mind2web_navigation.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in ds:
            steps = convert_mind2web_sample(sample)
            history_actions = []
            for step in steps:
                conv = build_navigation_conversation(
                    task=step["task"],
                    history_actions=history_actions[-2:],
                    current_action=step["action"],
                )
                record = {
                    "conversations": conv,
                    "annotation_id": step["annotation_id"],
                    "action_uid": step["action_uid"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                a = step["action"]
                pos_str = f"({a['position'][0]}, {a['position'][1]})"
                val_str = f', "{a["value"]}"' if "value" in a else ""
                history_actions.append(f"{a['action']}{pos_str}{val_str}")
                count += 1

    print(f"Wrote {count} navigation samples to {output_path}")


def preprocess_showui_web(data_dir: str, output_dir: str):
    """Process ShowUI-web dataset and save as grounding JSONL."""
    metadata_dir = Path(data_dir) / "showui-web" / "metadata"
    output_path = Path(output_dir) / "showui_web_grounding.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for json_file in sorted(metadata_dir.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as jf:
                samples = json.load(jf) if json_file.stat().st_size > 0 else []
                if isinstance(samples, dict):
                    samples = [samples]
                for sample in samples:
                    elements = convert_showui_web_sample(sample)
                    for elem in elements:
                        conv = build_grounding_conversation(
                            instruction=elem["instruction"],
                            position=elem["position"],
                        )
                        record = {
                            "conversations": conv,
                            "image": sample.get("img_url", ""),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1

    print(f"Wrote {count} grounding samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--output_dir", type=str, default="./datasets/processed")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mind2web", "showui-web"],
        choices=["mind2web", "showui-web"],
    )
    args = parser.parse_args()

    if "mind2web" in args.datasets:
        preprocess_mind2web(args.data_dir, args.output_dir)
    if "showui-web" in args.datasets:
        preprocess_showui_web(args.data_dir, args.output_dir)
