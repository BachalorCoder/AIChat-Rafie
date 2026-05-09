from __future__ import annotations

import re


def maybe_handle_preference(command: str, memory) -> str | None:
    match = re.match(r"\bmy favorite ([a-z0-9 ]+?) is (.+)$", command, flags=re.IGNORECASE)
    if not match:
        return None

    subject = match.group(1).strip()
    value = match.group(2).strip()
    memory.add(
        f"The user's favorite {subject} is {value}.",
        {"source": "preference", "mode": "chat", "plugin": "conversation"},
    )
    return f"I'll remember that your favorite {subject} is {value}."

