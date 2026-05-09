from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BrowserSnapshot:
    url: str
    title: str
    text: str


class BrowserSession:
    def __init__(self, headless: bool = False):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._page = None

    def start(self) -> None:
        if self._page:
            return

        from playwright.sync_api import Error, sync_playwright

        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=self.headless)
        except Error:
            self._browser = self._playwright.chromium.launch(channel="chrome", headless=self.headless)
        self._page = self._browser.new_page()

    def close(self) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._browser = None
        self._playwright = None
        self._page = None

    def open(self, url: str) -> BrowserSnapshot:
        self.start()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return self.snapshot()

    def snapshot(self) -> BrowserSnapshot:
        self.start()
        title = self._page.title()
        url = self._page.url
        try:
            text = self._page.locator("body").inner_text(timeout=3000)
        except Exception:
            text = ""
        return BrowserSnapshot(url=url, title=title, text=text[:12000])

    def execute(self, action: dict) -> str:
        self.start()
        method = action.get("method") or action.get("browser_method") or "none"

        if method == "none":
            return "No browser action to execute."
        if method == "goto":
            snapshot = self.open(action.get("url") or action.get("text") or "")
            return f"Opened {snapshot.url}"
        if method == "click":
            selector = action.get("selector")
            if not selector:
                return "Missing browser selector."
            self._page.locator(selector).click()
            return f"Clicked {selector}"
        if method == "fill":
            selector = action.get("selector")
            text = action.get("text") or ""
            if not selector:
                return "Missing browser selector."
            self._page.locator(selector).fill(text)
            return f"Filled {selector}"
        if method == "press":
            key = action.get("key") or action.get("text")
            if not key:
                return "Missing browser key."
            self._page.keyboard.press(key)
            return f"Pressed {key}"
        if method == "extract":
            return self.snapshot().text

        return f"Unsupported browser method: {method}"

