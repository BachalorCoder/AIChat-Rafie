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
