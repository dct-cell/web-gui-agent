"""Build model input prompts from screenshots, history, and instructions.

Constructs the messages list in Qwen2-VL chat format:
  [{"role": "user", "content": [{"type": "image", "image": <PIL>}, {"type": "text", "text": "..."}]}]
"""
from PIL import Image

from agent.action_parser import format_action_text


def build_prompt(
    screenshot: Image.Image,
    instruction: str,
    history: list[dict],
) -> list[dict]:
    """Build a navigation prompt with optional history.

    Args:
        screenshot: Current webpage screenshot (PIL Image).
        instruction: Natural language task instruction.
        history: List of {"screenshot": PIL.Image, "action": dict} from previous steps.

    Returns:
        Messages list in Qwen2-VL format.
    """
    content_parts = []

    # Add history screenshots and actions
    if history:
        history_lines = []
        for i, step in enumerate(history):
            content_parts.append({"type": "image", "image": step["screenshot"]})
            action_text = format_action_text(step["action"])
            history_lines.append(f"  Step {i+1}: {action_text}")

        history_str = "\n".join(history_lines)
    else:
        history_str = ""

    # Add current screenshot
    content_parts.append({"type": "image", "image": screenshot})

    # Build text instruction
    if history_str:
        text = (
            f"Task: {instruction}\n"
            f"Previous actions:\n{history_str}\n"
            f"Based on the current screenshot, what action should be performed next? "
            f"Respond with a JSON object: {{\"action\": \"click/type/scroll/done\", \"position\": [x, y], \"value\": \"text\"}}"
        )
    else:
        text = (
            f"Task: {instruction}\n"
            f"Based on the current screenshot, what action should be performed? "
            f"Respond with a JSON object: {{\"action\": \"click/type/scroll/done\", \"position\": [x, y], \"value\": \"text\"}}"
        )

    content_parts.append({"type": "text", "text": text})

    return [{"role": "user", "content": content_parts}]


def build_grounding_prompt(
    screenshot: Image.Image,
    instruction: str,
) -> list[dict]:
    """Build a grounding prompt (single element localization).

    Args:
        screenshot: Webpage screenshot.
        instruction: Description of element to locate.

    Returns:
        Messages list in Qwen2-VL format.
    """
    content_parts = [
        {"type": "image", "image": screenshot},
        {"type": "text", "text": f"In this web page screenshot, click on: {instruction}"},
    ]
    return [{"role": "user", "content": content_parts}]
