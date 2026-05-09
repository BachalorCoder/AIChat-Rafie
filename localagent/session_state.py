from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationSession:
    history: list[dict[str, str]] = field(default_factory=list)
    last_calculation: dict | None = None
    pending_calculation: dict | None = None

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        self.history = self.history[-12:]

    def recent_history_text(self) -> str:
        if not self.history:
            return ""

        lines = []
        for item in self.history[-10:]:
            speaker = "User" if item["role"] == "user" else "Rafie"
            lines.append(f"{speaker}: {item['content']}")
        return "\n".join(lines)

