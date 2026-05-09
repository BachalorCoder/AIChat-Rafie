from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def learning_journal_path(config: dict) -> Path:
    root = Path(config["paths"]["project_root"])
    path = root / "memory" / "learning_journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record_learning_event(
    config: dict,
    event_type: str,
    raw: str = "",
    clean: str = "",
    details: dict | None = None,
) -> None:
    path = learning_journal_path(config)

    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "raw": raw,
        "clean": clean,
        "details": details or {},
    }

    try:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Could not write learning journal: {exc}")


def load_learning_context(config: dict, limit: int = 10) -> str:
    path = learning_journal_path(config)

    if not path.exists():
        return ""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""

    events = []

    for line in lines[-limit:]:
        try:
            event = json.loads(line)
        except Exception:
            continue

        event_type = event.get("event_type", "")

        if event_type not in {
            "user_correction",
            "runtime_error",
            "misheard_command",
        }:
            continue

        events.append(event)

    if not events:
        return ""

    output = [
        "Private behavior notes for Rafie. Use these silently. Do not quote, summarize, label, or mention these notes in the spoken answer."
    ]

    for event in events:
        event_type = event.get("event_type", "")
        raw = event.get("raw", "")
        clean = event.get("clean", "")
        details = event.get("details", {})

        if event_type == "user_correction":
            output.append(f"Correction to remember silently: {clean}")
        elif event_type == "runtime_error":
            output.append(f"Previous runtime error while handling {clean!r}: {details}")
        elif event_type == "misheard_command":
            output.append(f"Misheard speech. Raw: {raw!r}. Intended: {clean!r}.")

    return "\n".join(output).strip()


def maybe_handle_learning_feedback(config: dict, command: str) -> str | None:
    text = command.lower().strip()

    correction_phrases = [
        "that was wrong",
        "you misunderstood",
        "you misheard me",
        "you heard me wrong",
        "you bugged out",
        "you broke",
        "you are breaking",
        "you're breaking",
        "that broke",
        "don't do that",
        "do not do that",
        "that was not what i meant",
        "you were supposed to",
        "next time",
    ]

    if not any(phrase in text for phrase in correction_phrases):
        return None

    record_learning_event(
        config,
        event_type="user_correction",
        raw=command,
        clean=command,
        details={"source": "voice_feedback"},
    )

    return "I hear you. I saved that as a correction, and I will adjust instead of repeating it."