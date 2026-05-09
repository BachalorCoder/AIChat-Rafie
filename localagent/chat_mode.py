from __future__ import annotations

import re

import ollama

from localagent.persona import persona_prompt
from localagent.session_state import ConversationSession


def answer_chat(
    config: dict,
    user_text: str,
    memory_context: str = "",
    session: ConversationSession | None = None,
    web_context: str = "",
) -> str:
    model = choose_chat_model(config, user_text, web_context)
    recent_history = session.recent_history_text() if session else ""
    prompt = f"""
You are having a spoken conversation with the user.
{persona_prompt(config)}

Answer naturally and briefly, like a companion speaking out loud.
Do not mention JSON. Do not mention prompts. Keep it to 1 to 3 sentences unless the user asks for detail.
Do not add extra factual claims unless you are confident or they are supported by web context.
Use the recent conversation to resolve follow-up phrases like "that", "show me the steps", or "yes do that".
If the user asks about a previous calculation, refer to the most recent relevant calculation in the recent conversation.
If web context is provided, use it and mention "I found" or "I checked" naturally. Do not invent sources.
If the user gives a preference or personal fact, acknowledge it and remember it rather than asking what they meant.

{memory_context}

{web_context}

Recent conversation:
{recent_history}

User said:
{user_text}
""".strip()

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 120},
    )
    answer = response["message"]["content"].strip()
    fallback_models = fallback_chat_models(config, model)
    for fallback_model in fallback_models:
        if answer:
            break
        response = ollama.chat(
            model=fallback_model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 120},
        )
        answer = response["message"]["content"].strip()
    return answer or "I'm sorry, my thoughts went quiet for a moment. Can you ask me that one again?"


def choose_chat_model(config: dict, user_text: str, web_context: str = "") -> str:
    models = config.get("models", {})
    if wants_deep_model(user_text) and models.get("chat_smart"):
        return models["chat_smart"]
    if web_context and models.get("planner"):
        return models["planner"]
    return models.get("chat") or models.get("planner")


def wants_deep_model(user_text: str) -> bool:
    text = user_text.lower()
    hard_terms = (
        "think deeply",
        "use your smart model",
        "use the smart model",
        "deep reasoning",
        "write code",
        "debug this code",
    )
    return any(term in text for term in hard_terms)


def fallback_chat_models(config: dict, current_model: str) -> list[str]:
    models = config.get("models", {})
    candidates = [
        models.get("chat_fast"),
        models.get("chat"),
        models.get("planner"),
        models.get("chat_smart"),
    ]
    unique = []
    for candidate in candidates:
        if candidate and candidate != current_model and candidate not in unique:
            unique.append(candidate)
    return unique


def needs_screen(user_text: str) -> bool:
    text = user_text.lower()
    phrase_terms = (
        "my screen",
        "the screen",
        "this screen",
        "look at",
        "look on",
        "what do you see",
        "what is visible",
        "open vs code",
        "open vscode",
        "vs code search",
        "vscode search",
        "search box",
        "click",
        "scroll",
        "mouse",
        "desktop",
        "browser tab",
        "this window",
        "current window",
    )
    word_terms = {
        "screen",
        "visible",
        "window",
        "hotkey",
        "vscode",
        "browser",
        "desktop",
        "click",
        "scroll",
    }
    words = set(re.findall(r"[a-z0-9]+", text))
    return any(term in text for term in phrase_terms) or bool(words & word_terms)
