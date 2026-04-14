"""Tests for the task executor loop."""
from unittest.mock import MagicMock
from PIL import Image

from agent.executor import TaskExecutor, ExecutionResult


def _make_image():
    return Image.new("RGB", (1344, 756), (255, 255, 255))


def test_executor_completes_on_done_action():
    mock_model = MagicMock()
    mock_model.predict.return_value = {"action": "done", "answer": "Task completed"}

    mock_browser = MagicMock()
    mock_browser.screenshot.return_value = _make_image()

    executor = TaskExecutor(model=mock_model, browser=mock_browser, max_steps=15)
    result = executor.run("https://example.com", "Do something")

    assert result.success is True
    assert result.answer == "Task completed"
    assert result.steps_taken == 1
    mock_browser.navigate.assert_called_once_with("https://example.com")


def test_executor_stops_at_max_steps():
    mock_model = MagicMock()
    mock_model.predict.return_value = {"action": "click", "position": [0.5, 0.5]}

    mock_browser = MagicMock()
    mock_browser.screenshot.return_value = _make_image()

    executor = TaskExecutor(model=mock_model, browser=mock_browser, max_steps=3)
    result = executor.run("https://example.com", "Do something")

    assert result.success is False
    assert result.steps_taken == 3


def test_executor_handles_parse_failure():
    mock_model = MagicMock()
    mock_model.predict.side_effect = [None, {"action": "done"}]

    mock_browser = MagicMock()
    mock_browser.screenshot.return_value = _make_image()

    executor = TaskExecutor(model=mock_model, browser=mock_browser, max_steps=15)
    result = executor.run("https://example.com", "Do something")

    assert result.success is True
    assert result.steps_taken == 2


def test_executor_records_history():
    mock_model = MagicMock()
    mock_model.predict.side_effect = [
        {"action": "click", "position": [0.1, 0.2]},
        {"action": "type", "position": [0.3, 0.4], "value": "hello"},
        {"action": "done"},
    ]

    mock_browser = MagicMock()
    mock_browser.screenshot.return_value = _make_image()

    executor = TaskExecutor(model=mock_model, browser=mock_browser, max_steps=15)
    result = executor.run("https://example.com", "Do something")

    assert result.steps_taken == 3
    assert len(result.step_history) == 3
    assert result.step_history[0]["action"]["action"] == "click"
    assert result.step_history[1]["action"]["action"] == "type"
    assert result.step_history[2]["action"]["action"] == "done"
