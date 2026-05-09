from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def needs_web_search(text: str) -> bool:
    lower = text.lower()
    triggers = (
        "search the web",
        "web search",
        "look up",
        "google",
        "latest",
        "today",
        "current",
        "right now",
        "news",
        "distance",
        "how far",
        "farthest",
        "closest",
        "release date",
        "price",
        "weather",
        "who is the president",
        "who won",
        "is it true",
    )
    return any(trigger in lower for trigger in triggers)


def response_seems_uncertain(text: str) -> bool:
    lower = text.lower()
    uncertain = (
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "i can't access",
        "i cannot access",
        "i don't have access",
        "i do not have access",
        "as of my last",
        "i may be wrong",
    )
    return any(marker in lower for marker in uncertain)


def web_search(query: str, max_results: int = 4) -> list[SearchResult]:
    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    response = requests.get(
        url,
        timeout=12,
        headers={
            "User-Agent": "Mozilla/5.0 LocalAgent/1.0",
        },
    )
    response.raise_for_status()

    parser = DuckDuckGoLiteParser()
    parser.feed(response.text)
    return parser.results[:max_results]


def format_web_context(results: list[SearchResult]) -> str:
    if not results:
        return ""

    lines = ["Web search results:"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}")
        lines.append(f"   URL: {result.url}")
        if result.snippet:
            lines.append(f"   Snippet: {result.snippet}")
    return "\n".join(lines)


class DuckDuckGoLiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_link = False
        self._in_snippet = False
        self._href = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        self._pending_result: SearchResult | None = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a" and attrs_dict.get("class") == "result-link":
            self._in_link = True
            self._href = attrs_dict.get("href", "")
            self._title_parts = []
        elif tag == "td" and attrs_dict.get("class") == "result-snippet":
            self._in_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            title = _clean_text("".join(self._title_parts))
            url = _unwrap_duck_url(self._href)
            if title and url:
                self._pending_result = SearchResult(title=title, url=url, snippet="")
            self._in_link = False
        elif tag == "td" and self._in_snippet:
            snippet = _clean_text("".join(self._snippet_parts))
            if self._pending_result:
                self._pending_result.snippet = snippet
                self.results.append(self._pending_result)
                self._pending_result = None
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_link:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)


def _clean_text(text: str) -> str:
    return " ".join(unescape(text).split())


def _unwrap_duck_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(uddg)
    return url
