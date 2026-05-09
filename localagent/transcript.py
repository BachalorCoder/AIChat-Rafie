from __future__ import annotations

import re


LEADING_FILLERS = (
    "the",
    "to",
    "a",
    "uh",
    "um",
    "hey",
    "okay",
    "ok",
)

DEFAULT_IGNORED_SHORT_PHRASES = {
    "the",
    "a",
    "to",
    "uh",
    "um",
    "hmm",
    "okay",
    "ok",
    "yeah",
    "yes",
    "no",
    "oh",
    "oh my god",
    "rafi",
    "rafie",
    "raffy",
}

PHRASE_FIXES = {
    "meet the steps": "show me the steps",
    "me the steps": "show me the steps",
    "the show me the steps": "show me the steps",
    "to demonstrate demonstrated": "demonstrate another calculation",
    "to demonstrate another calculation for me": "demonstrate another calculation for me",
    "can you explain me": "can you explain to me",
    "far this planet": "farthest planet",
    "what is the far this planet": "what is the farthest planet",
    "let s": "let's",
}


def clean_command(text: str) -> str:
    text = normalize_for_command(text)

    for wrong, right in sorted(PHRASE_FIXES.items(), key=lambda item: len(item[0]), reverse=True):
        if wrong in text:
            text = text.replace(wrong, right)
            break

    words = text.split()
    while len(words) > 1 and words[0] in LEADING_FILLERS:
        words.pop(0)

    return " ".join(words).strip()


def normalize_for_command(text: str) -> str:
    text = text.lower()
    text = text.replace("good night", "goodnight")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_likely_background_noise(command: str, config: dict | None = None) -> bool:
    command = normalize_for_command(command)
    if not command:
        return True

    voice_config = (config or {}).get("voice", {})
    ignored = set(DEFAULT_IGNORED_SHORT_PHRASES)
    ignored.update(normalize_for_command(x) for x in voice_config.get("ignored_short_phrases", []))

    if command in ignored:
        return True

    words = command.split()

    # Vosk often hears tiny fragments like "the" or "to".
    if len(words) == 1 and len(command) <= 4:
        return True

    # Ignore short passive reactions unless they look like a real question/instruction.
    question_words = {"what", "why", "how", "when", "where", "who", "can", "could", "would", "should", "tell", "explain", "give", "show", "make", "open", "search"}
    if len(words) <= 3 and words[0] not in question_words:
        return True

    return False