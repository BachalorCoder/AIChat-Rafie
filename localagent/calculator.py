from __future__ import annotations

import operator
import re
from typing import Callable

from localagent.session_state import ConversationSession
from localagent.transcript import clean_command


ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

OPERATORS: tuple[tuple[tuple[str, ...], str, Callable[[float, float], float]], ...] = (
    (("times", "multiplied by", "multiply by", "x"), "times", operator.mul),
    (("plus", "add", "added to"), "plus", operator.add),
    (("minus", "subtract", "subtracted by"), "minus", operator.sub),
    (("divided by", "over"), "divided by", operator.truediv),
)


def maybe_answer_math(command: str, session: ConversationSession) -> str | None:
    cleaned = clean_command(command)

    if _asks_for_steps(cleaned):
        if session.last_calculation:
            return explain_calculation(session.last_calculation)
        return "I can show the steps, but I need a calculation first. Try something like, what is five times seven?"

    if "demonstrate another calculation" in cleaned:
        calc = _make_calculation(5, "times", 7, operator.mul)
        session.last_calculation = calc
        return explain_calculation(calc)

    if cleaned in {"yes do that", "yes", "do that"} and session.pending_calculation:
        session.last_calculation = session.pending_calculation
        session.pending_calculation = None
        return explain_calculation(session.last_calculation)

    calc = parse_calculation(cleaned)
    if calc:
        session.last_calculation = calc
        session.pending_calculation = None
        return concise_answer(calc)

    return None


def parse_calculation(text: str) -> dict | None:
    text = _strip_question_words(text)

    for phrases, label, func in OPERATORS:
        for phrase in phrases:
            pattern = rf"\b{re.escape(phrase)}\b"
            parts = re.split(pattern, text, maxsplit=1)
            if len(parts) != 2:
                continue

            left = parse_number(parts[0].strip())
            right = parse_number(parts[1].strip())
            if left is None or right is None:
                continue
            if label == "divided by" and right == 0:
                return {
                    "left": left,
                    "right": right,
                    "operator": label,
                    "result": None,
                    "error": "division by zero",
                }
            return _make_calculation(left, label, right, func)

    return None


def parse_number(text: str) -> float | None:
    text = text.strip().replace("-", " ")
    text = re.sub(r"\band\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    numeric = text.replace(",", "")
    try:
        if "." in numeric:
            return float(numeric)
        return int(numeric)
    except ValueError:
        pass

    total = 0
    current = 0
    used = False

    for word in text.split():
        if word in ONES:
            current += ONES[word]
            used = True
        elif word in TENS:
            current += TENS[word]
            used = True
        elif word == "hundred":
            current = max(current, 1) * 100
            used = True
        elif word == "thousand":
            total += max(current, 1) * 1000
            current = 0
            used = True
        elif word == "million":
            total += max(current, 1) * 1_000_000
            current = 0
            used = True
        else:
            return None

    if not used:
        return None
    return total + current


def concise_answer(calc: dict) -> str:
    if calc.get("error") == "division by zero":
        return "I can't divide by zero. That one breaks the rules of arithmetic."

    left = _format_number(calc["left"])
    right = _format_number(calc["right"])
    result = _format_number(calc["result"])
    return f"{left} {calc['operator']} {right} is {result}."


def explain_calculation(calc: dict) -> str:
    if calc.get("error") == "division by zero":
        return "For that calculation, the issue is division by zero. We cannot split a number into zero groups."

    left = _format_number(calc["left"])
    right = _format_number(calc["right"])
    result = _format_number(calc["result"])
    if calc["operator"] == "times":
        return (
            f"Of course. For {left} times {right}, I treat it as {left} groups of {right}. "
            f"So I multiply {left} by {right}, which gives {result}."
        )
    if calc["operator"] == "plus":
        return f"Sure. For {left} plus {right}, I add the two numbers together, which gives {result}."
    if calc["operator"] == "minus":
        return f"Sure. For {left} minus {right}, I take {right} away from {left}, which gives {result}."
    if calc["operator"] == "divided by":
        return f"Sure. For {left} divided by {right}, I split {left} into {right} equal parts, which gives {result}."
    return concise_answer(calc)


def _make_calculation(left, label, right, func) -> dict:
    return {
        "left": left,
        "right": right,
        "operator": label,
        "result": func(left, right),
    }


def _strip_question_words(text: str) -> str:
    prefixes = (
        "what is",
        "what's",
        "calculate",
        "can you calculate",
        "what does",
        "tell me",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _asks_for_steps(text: str) -> bool:
    step_terms = ("step", "steps", "explain", "detail", "how you got", "how did you get")
    return any(term in text for term in step_terms)


def _format_number(value) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}"

