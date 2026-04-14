"""Tests for data preprocessing."""
import json
from data.preprocess import (
    convert_mind2web_sample,
    convert_showui_web_sample,
    build_grounding_conversation,
    build_navigation_conversation,
)


def test_convert_mind2web_sample_click():
    """A CLICK action should produce action=click with normalized position."""
    sample = {
        "annotation_id": "test_001",
        "confirmed_task": "Search for flights",
        "actions": [
            {
                "action_uid": "a1",
                "operation": {"op": "CLICK", "value": ""},
                "pos_candidates": [
                    {
                        "is_top_level_target": True,
                        "attributes": '{"bounding_box_rect": "100,200,150,220"}',
                        "backend_node_id": "node1",
                    }
                ],
                "neg_candidates": [],
            }
        ],
    }
    result = convert_mind2web_sample(sample, img_width=1344, img_height=756)
    assert len(result) == 1
    step = result[0]
    assert step["task"] == "Search for flights"
    assert step["action"]["action"] == "click"
    assert 0.08 < step["action"]["position"][0] < 0.10
    assert 0.27 < step["action"]["position"][1] < 0.29


def test_convert_mind2web_sample_type():
    """A TYPE action should produce action=type with value."""
    sample = {
        "annotation_id": "test_002",
        "confirmed_task": "Type a query",
        "actions": [
            {
                "action_uid": "a2",
                "operation": {"op": "TYPE", "value": "hello world"},
                "pos_candidates": [
                    {
                        "is_top_level_target": True,
                        "attributes": '{"bounding_box_rect": "50,100,200,120"}',
                        "backend_node_id": "node2",
                    }
                ],
                "neg_candidates": [],
            }
        ],
    }
    result = convert_mind2web_sample(sample, img_width=1344, img_height=756)
    step = result[0]
    assert step["action"]["action"] == "type"
    assert step["action"]["value"] == "hello world"


def test_build_grounding_conversation():
    """Grounding conversation should have user instruction and assistant coordinates."""
    conv = build_grounding_conversation(
        instruction="Click the search button",
        position=[0.85, 0.03],
    )
    assert len(conv) == 2
    assert conv[0]["role"] == "user"
    assert "Click the search button" in conv[0]["content"]
    assert conv[1]["role"] == "assistant"
    assert "0.85" in conv[1]["content"]
    assert "0.03" in conv[1]["content"]


def test_build_navigation_conversation():
    """Navigation conversation should include task and history actions."""
    conv = build_navigation_conversation(
        task="Search for flights",
        history_actions=["click(0.45, 0.03)"],
        current_action={"action": "type", "value": "NYC", "position": [0.45, 0.03]},
    )
    assert len(conv) == 2
    assert "Search for flights" in conv[0]["content"]
    assert "click(0.45, 0.03)" in conv[0]["content"]
    assert '"type"' in conv[1]["content"]
