"""Parse model text output into structured action dicts.

Handles multiple output formats:
- JSON: {"action": "click", "position": [x, y]}
- Coordinate tuple: (0.85, 0.03) → interpreted as click
- JSON embedded in surrounding text
"""
import json
import re
from typing import Optional


def parse_action(text: str) -> Optional[dict]:
    """Parse model output text into an action dict.

    Returns dict with keys: action, position (optional), value (optional)
    Returns None if parsing fails.
    """
    text = text.strip()

    # Try 1: direct JSON parse
    action = _try_json(text)
    if action:
        return action

    # Try 2: extract JSON from surrounding text
    json_match = re.search(r'\{[^{}]+\}', text)
    if json_match:
        action = _try_json(json_match.group())
        if action:
            return action

    # Try 3: coordinate tuple like (0.85, 0.03) → click
    coord_match = re.search(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', text)
    if coord_match:
        x, y = float(coord_match.group(1)), float(coord_match.group(2))
        if 0 <= x <= 1 and 0 <= y <= 1:
            return {"action": "click", "position": [x, y]}

    return None


def _try_json(text: str) -> Optional[dict]:
    """Try to parse text as a JSON action dict."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action" in data:
            action = {"action": data["action"]}
            if "position" in data:
                action["position"] = [float(data["position"][0]), float(data["position"][1])]
            if "value" in data:
                action["value"] = data["value"]
            if "answer" in data:
                action["answer"] = data["answer"]
            return action
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass
    return None


def format_action_text(action: dict) -> str:
    """Format an action dict as a human-readable string for history."""
    a = action["action"]
    if a == "click":
        pos = action["position"]
        return f"click({pos[0]}, {pos[1]})"
    elif a == "type":
        pos = action["position"]
        return f'type({pos[0]}, {pos[1]}, "{action["value"]}")'
    elif a == "scroll":
        return f'scroll({action.get("value", "down")})'
    elif a == "done":
        return f'done({action.get("answer", "")})'
    return str(action)
