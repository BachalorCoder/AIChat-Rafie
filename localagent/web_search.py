from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
import re

import requests


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source_type: str = "unknown"
    credibility: float = 0.0
    credibility_label: str = "unknown"
    reason: str = ""
    page_text: str = ""


HIGH_TRUST_DOMAINS = {
    "usa.gov",
    "congress.gov",
    "whitehouse.gov",
    "irs.gov",
    "sec.gov",
    "bls.gov",
    "bea.gov",
    "census.gov",
    "federalreserve.gov",
    "fda.gov",
    "cdc.gov",
    "nih.gov",
    "nlm.nih.gov",
    "medlineplus.gov",
    "clinicaltrials.gov",
    "who.int",
    "nasa.gov",
    "noaa.gov",
    "nist.gov",
    "usgs.gov",
    "energy.gov",
    "gov.uk",
    "nhs.uk",
    "parliament.uk",
    "europa.eu",
    "ec.europa.eu",
    "un.org",
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "law.cornell.edu",
    "supreme.justia.com",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "nature.com",
    "science.org",
    "nejm.org",
    "thelancet.com",
    "jamanetwork.com",
    "docs.python.org",
    "developer.mozilla.org",
    "learn.microsoft.com",
    "docs.github.com",
    "github.com",
    "pytorch.org",
    "huggingface.co",
    "ollama.com",
    "openai.com",
    "platform.openai.com",
}

RELIABLE_NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "npr.org",
    "pbs.org",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "economist.com",
    "politico.com",
    "axios.com",
    "theverge.com",
    "wired.com",
    "technologyreview.com",
}

LOW_TRUST_DOMAINS = {
    "quora.com",
    "reddit.com",
    "tiktok.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "pinterest.com",
    "medium.com",
    "substack.com",
    "fandom.com",
    "answers.com",
    "wikihow.com",
}

SUSPICIOUS_WORDS = {
    "shocking",
    "secret",
    "exposed",
    "miracle",
    "doctors hate",
    "one weird trick",
    "you won't believe",
    "leaked",
    "coverup",
    "cover-up",
    "truth revealed",
    "mainstream media won't",
}

CURRENT_OR_HIGH_STAKES_TERMS = {
    "latest",
    "current",
    "today",
    "yesterday",
    "tomorrow",
    "recent",
    "right now",
    "now",
    "news",
    "price",
    "cost",
    "weather",
    "forecast",
    "schedule",
    "release date",
    "version",
    "update",
    "law",
    "legal",
    "regulation",
    "rules",
    "tax",
    "medical",
    "health",
    "medicine",
    "drug",
    "dosage",
    "side effects",
    "financial",
    "stock",
    "election",
    "president",
    "governor",
    "mayor",
    "ceo",
    "github",
    "openai",
    "ollama",
    "pytorch",
    "cuda",
    "windows",
    "python package",
    "pip install",
}


def needs_web_search(text: str) -> bool:
    lower = text.lower().strip()

    explicit_triggers = (
        "search the web",
        "web search",
        "look up",
        "google",
        "check online",
        "verify",
        "fact check",
        "is it true",
        "are you sure",
        "source",
        "sources",
        "citation",
        "cite",
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
        "who is the current",
        "who is ceo",
        "what version",
    )

    if any(trigger in lower for trigger in explicit_triggers):
        return True

    if any(term in lower for term in CURRENT_OR_HIGH_STAKES_TERMS):
        return True

    unstable_patterns = (
        r"\bwho is the current\b",
        r"\bwho is president\b",
        r"\bwho is the president\b",
        r"\bwho is ceo\b",
        r"\bwhat version\b",
        r"\bhow much does .* cost\b",
        r"\bis .* still\b",
        r"\bwhen does .* release\b",
        r"\bwhen is .* coming out\b",
        r"\bside effects of\b",
        r"\bis .* safe\b",
    )

    return any(re.search(pattern, lower) for pattern in unstable_patterns)


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
        "my information may be outdated",
        "i can't verify",
        "i cannot verify",
        "i don't have current",
        "i do not have current",
    )

    return any(marker in lower for marker in uncertain)


def web_search(query: str, max_results: int = 4) -> list[SearchResult]:
    query = query.strip()

    if not query:
        return []

    raw_results = _ddgs_search(query, max_results=max_results * 3)

    if not raw_results:
        print("DDGS search returned no results. Falling back to DuckDuckGo Lite.")
        raw_results = _duckduckgo_lite_search(query, max_results=max_results * 3)

    scored_results: list[SearchResult] = []

    for result in raw_results:
        source_type, score, label, reason = score_source(
            result.url,
            result.title,
            result.snippet,
            query,
        )

        result.source_type = source_type
        result.credibility = score
        result.credibility_label = label
        result.reason = reason

        scored_results.append(result)

    scored_results.sort(key=lambda item: item.credibility, reverse=True)

    credible_results = [r for r in scored_results if r.credibility >= 0.45]

    if credible_results:
        final_results = credible_results[:max_results]
    else:
        final_results = scored_results[:max_results]

    for result in final_results:
        if result.credibility >= 0.55:
            result.page_text = fetch_page_text(result.url)

    return final_results


def _ddgs_search(query: str, max_results: int = 12) -> list[SearchResult]:
    try:
        from ddgs import DDGS
    except Exception as exc:
        print(f"DDGS is not available, using DuckDuckGo Lite fallback: {exc}")
        return []

    results: list[SearchResult] = []

    try:
        with DDGS(timeout=12) as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                title = (
                    item.get("title")
                    or item.get("heading")
                    or ""
                ).strip()

                url = (
                    item.get("href")
                    or item.get("url")
                    or ""
                ).strip()

                snippet = (
                    item.get("body")
                    or item.get("snippet")
                    or ""
                ).strip()

                if title and url:
                    results.append(
                        SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                        )
                    )

    except Exception as exc:
        print(f"DDGS search failed, using DuckDuckGo Lite fallback: {exc}")
        return []

    return results[:max_results]


def _duckduckgo_lite_search(query: str, max_results: int = 12) -> list[SearchResult]:
    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"

    try:
        response = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0 LocalAgent/1.0",
            },
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"DuckDuckGo Lite search failed: {exc}")
        return []

    parser = DuckDuckGoLiteParser()
    parser.feed(response.text)

    return parser.results[:max_results]


def format_web_context(results: list[SearchResult]) -> str:
    if not results:
        return ""

    high = [r for r in results if r.credibility_label == "high"]
    medium = [r for r in results if r.credibility_label == "medium"]
    low = [r for r in results if r.credibility_label == "low"]

    lines = [
        "Web research results:",
        "Use the most credible sources first. Prefer official, primary, academic, government, medical institution, official documentation, or established news sources.",
        "Treat weak sources like forums, random blogs, social media, and anonymous posts as unconfirmed unless the user asked for opinions.",
        f"Source quality summary: {len(high)} high, {len(medium)} medium, {len(low)} low.",
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}")
        lines.append(f"   URL: {result.url}")
        lines.append(f"   Credibility: {result.credibility_label} ({result.credibility:.2f})")
        lines.append(f"   Source type: {result.source_type}")
        lines.append(f"   Why trusted or not: {result.reason}")

        if result.snippet:
            lines.append(f"   Snippet: {result.snippet}")

        if result.page_text:
            lines.append(f"   Page excerpt: {result.page_text}")

        lines.append("")

    if not high and not medium:
        lines.append(
            "Warning: These search results are weak. Do not present the answer as confirmed truth."
        )

    return "\n".join(lines).strip()


def score_source(
    url: str,
    title: str = "",
    snippet: str = "",
    query: str = "",
) -> tuple[str, float, str, str]:
    domain = normalized_domain(url)
    root = root_domain(domain)

    title_lower = title.lower()
    snippet_lower = snippet.lower()
    query_lower = query.lower()

    score = 0.35
    reasons: list[str] = []
    source_type = "general web source"

    if domain.endswith(".gov") or root.endswith(".gov"):
        score += 0.40
        source_type = "official government source"
        reasons.append("official government domain")

    if domain.endswith(".edu") or root.endswith(".edu"):
        score += 0.25
        source_type = "academic or educational source"
        reasons.append("academic or educational domain")

    if domain in HIGH_TRUST_DOMAINS or root in HIGH_TRUST_DOMAINS:
        score += 0.35
        source_type = "high-trust or primary source"
        reasons.append("known high-trust source")

    if domain in RELIABLE_NEWS_DOMAINS or root in RELIABLE_NEWS_DOMAINS:
        score += 0.20
        source_type = "established news source"
        reasons.append("established news organization")

    if domain in LOW_TRUST_DOMAINS or root in LOW_TRUST_DOMAINS:
        score -= 0.25
        source_type = "low-trust or user-generated source"
        reasons.append("user-generated or lower-trust source")

    if "wikipedia.org" in domain:
        score += 0.05
        source_type = "general reference source"
        reasons.append("useful overview, but not a primary source")

    if any(word in title_lower for word in SUSPICIOUS_WORDS):
        score -= 0.30
        reasons.append("sensational title language")

    if any(word in snippet_lower for word in SUSPICIOUS_WORDS):
        score -= 0.20
        reasons.append("sensational snippet language")

    if "sponsored" in title_lower or "advertisement" in title_lower:
        score -= 0.20
        reasons.append("possibly promotional")

    if _is_health_query(query_lower):
        if root in {
            "nih.gov",
            "cdc.gov",
            "fda.gov",
            "who.int",
            "nhs.uk",
            "medlineplus.gov",
            "mayoclinic.org",
            "clevelandclinic.org",
        }:
            score += 0.25
            reasons.append("strong source for health topic")
        elif root in LOW_TRUST_DOMAINS:
            score -= 0.25
            reasons.append("weak source for health topic")

    if _is_legal_or_financial_query(query_lower):
        if domain.endswith(".gov") or root.endswith(".gov") or root in {
            "law.cornell.edu",
            "congress.gov",
            "sec.gov",
            "irs.gov",
            "federalreserve.gov",
        }:
            score += 0.25
            reasons.append("strong source for legal or financial topic")

    if _is_technical_query(query_lower):
        if root in {
            "docs.python.org",
            "pytorch.org",
            "github.com",
            "docs.github.com",
            "learn.microsoft.com",
            "developer.mozilla.org",
            "huggingface.co",
            "ollama.com",
            "openai.com",
        }:
            score += 0.25
            reasons.append("official or primary technical source")

    score = max(0.0, min(1.0, score))

    if score >= 0.78:
        label = "high"
    elif score >= 0.55:
        label = "medium"
    else:
        label = "low"

    reason = ", ".join(reasons) if reasons else "no strong credibility signals"

    return source_type, score, label, reason


def fetch_page_text(url: str, max_chars: int = 1800) -> str:
    if not url.startswith(("http://", "https://")):
        return ""

    parsed = urlparse(url)
    path = parsed.path.lower()

    if path.endswith(
        (
            ".pdf",
            ".zip",
            ".exe",
            ".dmg",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".mp4",
            ".mp3",
        )
    ):
        return ""

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 LocalAgent/1.0",
            },
        )
        response.raise_for_status()
    except Exception:
        return ""

    content_type = response.headers.get("content-type", "").lower()

    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return ""

    text = strip_html(response.text)
    text = _clean_text(text)

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."

    return text


def strip_html(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.get_text()


def normalized_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return ""

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def root_domain(domain: str) -> str:
    if not domain:
        return ""

    parts = domain.split(".")

    if len(parts) <= 2:
        return domain

    if parts[-2] in {"co", "gov", "ac", "org"} and parts[-1] in {"uk", "au", "nz", "za"} and len(parts) >= 3:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:])


def _is_health_query(text: str) -> bool:
    terms = {
        "medical",
        "health",
        "medicine",
        "drug",
        "dosage",
        "side effects",
        "symptom",
        "treatment",
        "disease",
        "condition",
        "safe",
    }
    return any(term in text for term in terms)


def _is_legal_or_financial_query(text: str) -> bool:
    terms = {
        "law",
        "legal",
        "regulation",
        "rules",
        "tax",
        "financial",
        "stock",
        "interest rate",
        "sec",
        "irs",
    }
    return any(term in text for term in terms)


def _is_technical_query(text: str) -> bool:
    terms = {
        "python",
        "pytorch",
        "cuda",
        "ollama",
        "github",
        "openai",
        "windows",
        "pip",
        "package",
        "version",
        "install",
        "error",
        "traceback",
    }
    return any(term in text for term in terms)


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


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg", "canvas"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg", "canvas"} and self.skip_depth > 0:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


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