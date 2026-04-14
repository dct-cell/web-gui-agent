"""Tests for Playwright browser wrapper."""
import pytest
from PIL import Image

from agent.browser import Browser


@pytest.fixture
def browser():
    b = Browser(width=1344, height=756, headless=True)
    yield b
    b.close()


def test_browser_navigate_and_screenshot(browser):
    browser.navigate("https://example.com")
    screenshot = browser.screenshot()
    assert isinstance(screenshot, Image.Image)
    assert screenshot.size == (1344, 756)


def test_browser_click(browser):
    browser.navigate("https://example.com")
    browser.click(0.5, 0.5)
    screenshot = browser.screenshot()
    assert isinstance(screenshot, Image.Image)


def test_browser_type_text(browser):
    browser.navigate("https://example.com")
    browser.type_text(0.5, 0.5, "test")


def test_browser_scroll(browser):
    browser.navigate("https://example.com")
    browser.scroll("down")
    browser.scroll("up")


def test_browser_execute_action(browser):
    browser.navigate("https://example.com")
    browser.execute({"action": "click", "position": [0.5, 0.5]})
    browser.execute({"action": "scroll", "value": "down"})
    screenshot = browser.screenshot()
    assert isinstance(screenshot, Image.Image)
