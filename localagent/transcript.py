from __future__ import annotations

import re


ASSISTANT_NAMES = {
    "rafi",
    "rafie",
    "raffy",
    "raphie"
}

LEADING_FILLERS = (
    "the",
    "to",
    "a",
    "uh",
    "um",
    "hey",
    "okay",
    "ok"
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
    "oh",
    "oh my god",
    "my god",
    "rafi",
    "rafie",
    "raffy",
    "raphie"
}

QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were"
}

COMMAND_WORDS = {
    "tell",
    "explain",
    "give",
    "show",
    "make",
    "open",
    "search",
    "look",
    "click",
    "scroll",
    "stop",
    "pause",
    "continue",
    "sleep",
    "goodnight",
    "wake",
    "list",
    "voice",
    "switch",
    "change",
    "use",
    "set",
    "read",
    "write",
    "talk",
    "say",
    "sing",
    "help",
    "remember",
    "forget",
    "start",
    "play",
    "close",
    "run",
    "act",
    "pretend",
    "cry",
    "crying",
    "laugh",
    "laughing",
    "smile",
    "angry",
    "sad",
    "happy",
    "rap",
    "wrap"
}

SOCIAL_PHRASES = {
    "thank you",
    "thanks",
    "good job",
    "nice",
    "cool",
    "awesome",
    "that was good",
    "that is good"
}

PHRASE_FIXES = {
    "rapture": "rafi",
    "rafiq": "rafi",
    "rafiki": "rafi",
    "roughy": "rafi",
    "murphy": "rafi",
    "raffia": "rafi",

    "seeing me a happy birthday": "sing me a happy birthday",
    "seeing me happy birthday": "sing me happy birthday",
    "singing me a happy birthday": "sing me a happy birthday",
    "send me a happy birthday": "sing me a happy birthday",
    "happy birthday to emmy": "sing happy birthday to emmy",

    "r a p": "rap",
    "w r a p": "wrap",

    "know i like": "no i like",
    "know i want": "no i want",
    "know i mean": "no i mean",

    "meet the steps": "show me the steps",
    "me the steps": "show me the steps",
    "the show me the steps": "show me the steps",
    "to demonstrate demonstrated": "demonstrate another calculation",
    "to demonstrate another calculation for me": "demonstrate another calculation for me",
    "can you explain me": "can you explain to me",
    "far this planet": "farthest planet",
    "what is the far this planet": "what is the farthest planet",
    "let s": "let's",

    "rafi least": "rafi list",
    "rafi lists": "rafi list",
    "rafi lift": "rafi list",
    "rafi lis": "rafi list",

    "switch boys": "switch voice",
    "change boys": "change voice",
    "use boys": "use voice",
    "voice won": "voice one",
    "voice wun": "voice one",
    "voice too": "voice two",
    "voice to": "voice two",
    "voice tree": "voice three",
    "voice free": "voice three",
    "voice for": "voice four",
    "switched to voice": "switch voice",
    "switch to voice": "switch voice"
}


def clean_command(text: str) -> str:
    text = normalize_for_command(text)

    text = re.sub(r"\brap(?:ap)+\b", "rap", text)
    text = re.sub(r"\bwrap(?:ap)+\b", "wrap", text)

    for wrong, right in sorted(PHRASE_FIXES.items(), key=lambda item: len(item[0]), reverse=True):
        text = replace_phrase(text, wrong, right)

    text = re.sub(r"\brap(?:ap)+\b", "rap", text)
    text = re.sub(r"\bwrap(?:ap)+\b", "wrap", text)

    words = text.split()

    while len(words) > 1 and words[0] in LEADING_FILLERS:
        words.pop(0)

    while len(words) > 1 and words[-1] in LEADING_FILLERS:
        words.pop()

    return " ".join(words).strip()


def replace_phrase(text: str, wrong: str, right: str) -> str:
    pattern = r"(?<![a-z0-9])" + re.escape(wrong) + r"(?![a-z0-9])"
    return re.sub(pattern, right, text)


def normalize_for_command(text: str) -> str:
    text = text.lower()
    text = text.replace("good night", "goodnight")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\brap(?:ap)+\b", "rap", text)
    text = re.sub(r"\bwrap(?:ap)+\b", "wrap", text)
    return re.sub(r"\s+", " ", text).strip()


def is_likely_background_noise(command: str, config: dict | None = None) -> bool:
    command = normalize_for_command(command)

    if not command:
        return True

    voice_config = (config or {}).get("voice", {})

    ignored = set(DEFAULT_IGNORED_SHORT_PHRASES)
    ignored.update(normalize_for_command(x) for x in voice_config.get("ignored_short_phrases", []))

    # Never ignore "no" or "yes" globally, because those can be answers to clarification.
    ignored.discard("no")
    ignored.discard("yes")

    words = command.split()
    word_set = set(words)

    if command in ASSISTANT_NAMES:
        return True

    if word_set & ASSISTANT_NAMES and len(words) > 1:
        return False

    if word_set & COMMAND_WORDS:
        return False

    if words and words[0] in QUESTION_WORDS:
        return False

    if command in SOCIAL_PHRASES:
        return False

    if command in ignored:
        return True

    if len(words) == 1 and len(command) <= 4:
        return True

    if len(words) <= 2:
        useful_short_words = QUESTION_WORDS | COMMAND_WORDS | ASSISTANT_NAMES
        if not (word_set & useful_short_words):
            return True

    return False