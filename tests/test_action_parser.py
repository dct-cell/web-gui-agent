"""Tests for action parser — converts model text output to structured action dict."""
from agent.action_parser import parse_action


def test_parse_click_json():
    text = '{"action": "click", "position": [0.85, 0.03]}'
    action = parse_action(text)
    assert action["action"] == "click"
    assert action["position"] == [0.85, 0.03]


def test_parse_type_json():
    text = '{"action": "type", "value": "iPhone 16", "position": [0.45, 0.03]}'
    action = parse_action(text)
    assert action["action"] == "type"
    assert action["value"] == "iPhone 16"
    assert action["position"] == [0.45, 0.03]


def test_parse_scroll_json():
    text = '{"action": "scroll", "value": "down"}'
    action = parse_action(text)
    assert action["action"] == "scroll"
    assert action["value"] == "down"


def test_parse_done_json():
    text = '{"action": "done"}'
    action = parse_action(text)
    assert action["action"] == "done"


def test_parse_coordinate_tuple():
    """Model might output just coordinates like (0.85, 0.03) for grounding."""
    text = "(0.85, 0.03)"
    action = parse_action(text)
    assert action["action"] == "click"
    assert action["position"] == [0.85, 0.03]


def test_parse_json_with_surrounding_text():
    """Model output might have extra text around the JSON."""
    text = 'The next action is {"action": "click", "position": [0.5, 0.5]} to proceed.'
    action = parse_action(text)
    assert action["action"] == "click"
    assert action["position"] == [0.5, 0.5]


def test_parse_invalid_returns_none():
    text = "I don't know what to do"
    action = parse_action(text)
    assert action is None
