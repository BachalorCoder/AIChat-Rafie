import ast
import json
import re
from typing import Any


DEFAULT_ACTION = {
    "type": "none",
    "x": None,
    "y": None,
    "text": None,
    "keys": [],
    "reason": "No executable action was provided.",
    "risk": "low",
}


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse model JSON, repairing common LLM formatting mistakes."""
    last_error: Exception | None = None

    for candidate in _candidate_json_strings(text):
        for repaired in _repair_variants(candidate):
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return normalize_agent_response(parsed)
            except json.JSONDecodeError as exc:
                last_error = exc

            try:
                parsed = ast.literal_eval(repaired)
                if isinstance(parsed, dict):
                    return normalize_agent_response(parsed)
            except (SyntaxError, ValueError) as exc:
                last_error = exc

    if last_error:
        raise ValueError(f"Could not parse JSON response after repair: {last_error}") from last_error

    raise ValueError("Could not find a JSON object in the model response.")


def normalize_agent_response(parsed: dict[str, Any]) -> dict[str, Any]:
    action = parsed.get("action")
    if not isinstance(action, dict):
        action = DEFAULT_ACTION.copy()
    else:
        normalized_action = DEFAULT_ACTION.copy()
        normalized_action.update(action)
        if not isinstance(normalized_action.get("keys"), list):
            normalized_action["keys"] = []
        if normalized_action.get("risk") not in {"low", "medium", "high"}:
            normalized_action["risk"] = "high"
        if not isinstance(normalized_action.get("type"), str):
            normalized_action["type"] = "none"
        action = normalized_action

    return {
        "screen_summary": str(parsed.get("screen_summary", "")),
        "user_goal_interpretation": str(parsed.get("user_goal_interpretation", "")),
        "suggested_next_step": str(parsed.get("suggested_next_step", "")),
        "spoken_response": str(parsed.get("spoken_response", "")),
        "mode": str(parsed.get("mode", "desktop")),
        "plugin": parsed.get("plugin"),
        "action": action,
    }


def fallback_text_response(text: str, mode: str = "desktop", plugin: str | None = None) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if _looks_like_schema_echo(cleaned):
        cleaned = "I looked at the screen, but the small vision model returned a template instead of an answer. Try the quality vision model for this one, or ask me as a normal chat question."
    return {
        "screen_summary": cleaned,
        "user_goal_interpretation": "",
        "suggested_next_step": "I can keep advising from observation, but I did not get structured action JSON.",
        "spoken_response": cleaned or "I looked, but I could not form a clear response yet.",
        "mode": mode,
        "plugin": plugin,
        "action": DEFAULT_ACTION.copy(),
    }


def _looks_like_schema_echo(text: str) -> bool:
    lower = text.lower()
    schema_markers = (
        '"type": "screen_summary"',
        '"what the user likely wants"',
        '"safe next step"',
        '"none" | "click"',
    )
    return sum(marker in lower for marker in schema_markers) >= 2


def _candidate_json_strings(text: str) -> list[str]:
    cleaned = _strip_markdown_fence(_normalize_text(text or ""))
    candidates: list[str] = []

    if cleaned.strip():
        candidates.append(cleaned.strip())

    candidates.extend(_balanced_object_candidates(cleaned))

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _repair_variants(candidate: str) -> list[str]:
    base = candidate.strip()
    without_comments = _strip_json_comments(base)
    without_trailing = _remove_trailing_commas(without_comments)
    with_quoted_keys = _quote_unquoted_keys(without_trailing)
    pythonish_to_json = _python_literals_to_json(with_quoted_keys)

    return [
        base,
        without_comments,
        without_trailing,
        with_quoted_keys,
        pythonish_to_json,
    ]


def _normalize_text(text: str) -> str:
    replacements = {
        "\ufeff": "",
        "\u200b": "",
        "\u00a0": " ",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    fence = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return stripped


def _balanced_object_candidates(text: str) -> list[str]:
    candidates = []
    stack = 0
    start = None
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if stack == 0:
                start = index
            stack += 1
        elif char == "}":
            if stack:
                stack -= 1
                if stack == 0 and start is not None:
                    candidates.append(text[start : index + 1])
                    start = None

    return candidates


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"(^|[^:])//.*?$", r"\1", text, flags=re.MULTILINE)


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _quote_unquoted_keys(text: str) -> str:
    return re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)", r'\1"\2"\3', text)


def _python_literals_to_json(text: str) -> str:
    return (
        text.replace(": None", ": null")
        .replace(": True", ": true")
        .replace(": False", ": false")
    )
