from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\brap(?:ap)+\b", "rap", text)
    text = re.sub(r"\bwrap(?:ap)+\b", "wrap", text)
    return re.sub(r"\s+", " ", text).strip()


def apply_spelled_word_fixes(command: str) -> str:
    text = normalize_text(command)

    replacements = {
        "r a p": "rap",
        "w r a p": "wrap",
        "n o": "no",
        "k n o w": "know",
        "n i g h t": "night",
        "k n i g h t": "knight"
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    text = re.sub(r"\brap(?:ap)+\b", "rap", text)
    text = re.sub(r"\bwrap(?:ap)+\b", "wrap", text)

    return text.strip()


def maybe_start_disambiguation(command: str) -> dict | None:
    text = apply_spelled_word_fixes(command)

    if _needs_rap_wrap_disambiguation(text):
        return {
            "kind": "rap_wrap",
            "original": text,
            "prompt": "Do you mean R A P, rap like music, or W R A P, wrap like covering something or ending a topic?"
        }

    if _needs_no_know_disambiguation(text):
        return {
            "kind": "no_know",
            "original": text,
            "prompt": "Do you mean N O, no, or K N O W, know?"
        }

    if _needs_knight_night_disambiguation(text):
        return {
            "kind": "knight_night",
            "original": text,
            "prompt": "Do you mean knight, as in a person in armor, or night, as in when it gets dark?"
        }

    return None


def resolve_disambiguation(pending: dict, answer: str) -> str | None:
    kind = pending.get("kind", "")
    original = apply_spelled_word_fixes(pending.get("original", ""))
    answer_text = apply_spelled_word_fixes(answer)

    if kind == "rap_wrap":
        if _answer_means_rap(answer_text):
            return _force_rap_meaning(original)

        if _answer_means_wrap(answer_text):
            return _force_wrap_meaning(original)

    if kind == "no_know":
        if _answer_means_no(answer_text):
            return _replace_ambiguous(original, ["know", "no"], "no")

        if _answer_means_know(answer_text):
            return _replace_ambiguous(original, ["know", "no"], "know")

    if kind == "knight_night":
        if _answer_means_knight(answer_text):
            return _replace_ambiguous(original, ["night", "knight"], "knight")

        if _answer_means_night(answer_text):
            return _replace_ambiguous(original, ["night", "knight"], "night")

    return None


def _replace_ambiguous(original: str, candidates: list[str], replacement: str) -> str:
    text = apply_spelled_word_fixes(original)

    for candidate in candidates:
        pattern = r"(?<![a-z0-9])" + re.escape(candidate) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            return re.sub(pattern, replacement, text)

    return f"{text} {replacement}".strip()


def _force_rap_meaning(original: str) -> str:
    text = apply_spelled_word_fixes(original)

    replacements = [
        "wrap",
        "rap"
    ]

    for candidate in replacements:
        pattern = r"(?<![a-z0-9])" + re.escape(candidate) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            return re.sub(pattern, "rap song", text)

    return f"{text} rap song".strip()


def _force_wrap_meaning(original: str) -> str:
    text = apply_spelled_word_fixes(original)

    replacements = [
        "wrap",
        "rap"
    ]

    for candidate in replacements:
        pattern = r"(?<![a-z0-9])" + re.escape(candidate) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            return re.sub(pattern, "wrap up this topic", text)

    return f"{text} wrap up this topic".strip()


def _answer_means_rap(text: str) -> bool:
    rap_words = {
        "rap",
        "music",
        "song",
        "rhymes",
        "rhyme",
        "beat",
        "hip",
        "hop",
        "lyrics",
        "freestyle"
    }

    if "r a p" in text:
        return True

    words = set(text.split())
    return bool(words & rap_words)


def _answer_means_wrap(text: str) -> bool:
    wrap_words = {
        "wrap",
        "cover",
        "covering",
        "present",
        "gift",
        "finish",
        "ending",
        "topic",
        "up"
    }

    if "w r a p" in text:
        return True

    words = set(text.split())
    return bool(words & wrap_words)


def _answer_means_no(text: str) -> bool:
    return text in {"no", "n o"} or "negative" in text


def _answer_means_know(text: str) -> bool:
    return "know" in text or "knowledge" in text or "understand" in text


def _answer_means_knight(text: str) -> bool:
    return "knight" in text or "armor" in text or "armour" in text or "horse" in text or "sword" in text


def _answer_means_night(text: str) -> bool:
    return "night" in text or "dark" in text or "sleep" in text or "moon" in text


def _needs_rap_wrap_disambiguation(text: str) -> bool:
    text = apply_spelled_word_fixes(text)

    if "rap" not in text and "wrap" not in text:
        return False

    clear_rap_phrases = {
        "rap song",
        "rap music",
        "rap verse",
        "rap battle",
        "make a rap song",
        "write a rap song",
        "perform a rap song",
        "freestyle rap",
        "rap about"
    }

    clear_wrap_phrases = {
        "wrap up",
        "wrap it up",
        "wrap this up",
        "gift wrap",
        "wrap a present",
        "wrap the gift",
        "wrap up this topic"
    }

    if any(phrase in text for phrase in clear_rap_phrases):
        return False

    if any(phrase in text for phrase in clear_wrap_phrases):
        return False

    vague_triggers = {
        "i want you to rap",
        "i want you to wrap",
        "can you rap",
        "can you wrap",
        "do a rap",
        "do a wrap",
        "start rap",
        "start wrap",
        "rafi rap",
        "rafi wrap"
    }

    return any(trigger in text for trigger in vague_triggers)


def _needs_no_know_disambiguation(text: str) -> bool:
    text = apply_spelled_word_fixes(text)

    if text.startswith("do you know"):
        return False

    if text.startswith("i know"):
        return False

    if text.startswith("no i "):
        return False

    if text.startswith("know i "):
        return True

    if text in {"no", "know"}:
        return True

    return False


def _needs_knight_night_disambiguation(text: str) -> bool:
    text = apply_spelled_word_fixes(text)

    if "goodnight" in text or "go to sleep" in text:
        return False

    if "knight" not in text and "night" not in text:
        return False

    clear_night_phrases = {
        "at night",
        "tonight",
        "last night",
        "good night",
        "goodnight"
    }

    if any(phrase in text for phrase in clear_night_phrases):
        return False

    triggers = {
        "tell me about night",
        "tell me about knight",
        "what is night",
        "what is knight",
        "explain night",
        "explain knight"
    }

    return any(trigger in text for trigger in triggers)