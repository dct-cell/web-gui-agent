"""Playwright browser wrapper for the Web GUI Agent.

Provides screenshot capture and action execution (click, type, scroll)
with normalized coordinates (0-1).
"""
import io
from PIL import Image
from playwright.sync_api import sync_playwright, Browser as PWBrowser, Page


class Browser:
    def __init__(self, width: int = 1344, height: int = 756, headless: bool = True):
        self.width = width
        self.height = height
        self._pw = sync_playwright().start()
        self._browser: PWBrowser = self._pw.chromium.launch(headless=headless)
        self._page: Page = self._browser.new_page(
            viewport={"width": width, "height": height}
        )

    def navigate(self, url: str):
        """Navigate to a URL and wait for load."""
        self._page.goto(url, wait_until="domcontentloaded", timeout=30000)

    def screenshot(self) -> Image.Image:
        """Capture current page as a PIL Image."""
        png_bytes = self._page.screenshot(type="png")
        return Image.open(io.BytesIO(png_bytes))

    def click(self, x: float, y: float):
        """Click at normalized coordinates (0-1)."""
        px = int(x * self.width)
        py = int(y * self.height)
        self._page.mouse.click(px, py)
        self._page.wait_for_timeout(500)

    def type_text(self, x: float, y: float, text: str):
        """Click at position then type text."""
        px = int(x * self.width)
        py = int(y * self.height)
        self._page.mouse.click(px, py)
        self._page.wait_for_timeout(300)
        self._page.keyboard.type(text, delay=50)
        self._page.wait_for_timeout(300)

    def scroll(self, direction: str = "down"):
        """Scroll the page up or down."""
        delta = 500 if direction == "down" else -500
        self._page.mouse.wheel(0, delta)
        self._page.wait_for_timeout(500)

    def execute(self, action: dict):
        """Execute a structured action dict."""
        action_type = action["action"]
        if action_type == "click":
            pos = action["position"]
            self.click(pos[0], pos[1])
        elif action_type == "type":
            pos = action["position"]
            self.type_text(pos[0], pos[1], action["value"])
        elif action_type == "scroll":
            self.scroll(action.get("value", "down"))
        elif action_type == "done":
            pass
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def close(self):
        """Clean up browser resources."""
        self._browser.close()
        self._pw.stop()
