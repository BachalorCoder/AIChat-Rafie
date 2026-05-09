from __future__ import annotations


def maybe_answer_builtin(command: str) -> str | None:
    text = command.lower()

    if "algebra" in text and ("how" in text or "rundown" in text or "work" in text):
        return (
            "Algebra is a way to work with unknown numbers. We use letters like x as placeholders, "
            "then use the same operation on both sides of an equation to keep it balanced. For example, "
            "if x plus 3 equals 7, subtract 3 from both sides, and x equals 4."
        )

    if text in {"say something", "just say something", "rafie say something"}:
        return "I'm here. My thoughts got quiet for a second, but I haven't left you."

    return None
