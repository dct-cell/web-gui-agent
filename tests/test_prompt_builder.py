"""Tests for prompt builder — constructs model input from screenshot + history + instruction."""
from PIL import Image

from agent.prompt_builder import build_prompt, build_grounding_prompt


def _make_image(w=1344, h=756):
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def test_build_prompt_no_history():
    img = _make_image()
    messages = build_prompt(
        screenshot=img,
        instruction="Search for flights",
        history=[],
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert any(part["type"] == "image" for part in content)
    assert any("Search for flights" in part.get("text", "") for part in content if part["type"] == "text")


def test_build_prompt_with_history():
    img_now = _make_image()
    img_prev = _make_image()
    history = [
        {"screenshot": img_prev, "action": {"action": "click", "position": [0.5, 0.5]}},
    ]
    messages = build_prompt(
        screenshot=img_now,
        instruction="Search for flights",
        history=history,
    )
    assert len(messages) == 1
    content = messages[0]["content"]
    image_parts = [p for p in content if p["type"] == "image"]
    assert len(image_parts) == 2
    text_parts = [p for p in content if p["type"] == "text"]
    full_text = " ".join(p["text"] for p in text_parts)
    assert "click" in full_text


def test_build_grounding_prompt():
    img = _make_image()
    messages = build_grounding_prompt(
        screenshot=img,
        instruction="Click the search button",
    )
    assert len(messages) == 1
    content = messages[0]["content"]
    assert any(part["type"] == "image" for part in content)
    assert any("search button" in part.get("text", "") for part in content if part["type"] == "text")
