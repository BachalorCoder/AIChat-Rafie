from __future__ import annotations

import re

import ollama

from localagent.persona import persona_prompt
from localagent.session_state import ConversationSession
from localagent.tts_manager import clean_spoken_text


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

Important speech rules:
- Answer like a normal person speaking out loud.
- Do not write "Rafie:" or your own name before your answer.
- Do not use Markdown.
- Do not use asterisks.
- Do not use stage directions like *concerned*, *smiles*, or *chuckles*.
- Do not say formatting symbols out loud.
- Keep normal answers to 1 to 4 sentences.
- If the user asks for detail, explain clearly but still use natural speech.
- Use the recent conversation to understand follow-up phrases like "continue", "that", "show me the steps", or "yes do that".
- If the user asks you to continue, continue the previous topic instead of saying you have nothing to continue.
- Do not add extra factual claims unless you are confident or they are supported by web context.
- If web context is provided, use it and mention "I found" or "I checked" naturally.
- Do not invent sources.
- If the user gives a preference or personal fact, acknowledge it and remember it.

{memory_context}

{web_context}

Recent conversation:
{recent_history}

User said:
{user_text}
""".strip()

    answer = _chat_once(config, model, prompt, user_text)

    for fallback_model in fallback_chat_models(config, model):
        if answer:
            break
        answer = _chat_once(config, fallback_model, prompt, user_text)

    return clean_spoken_text(answer) or "I'm sorry, my thoughts went quiet for a moment. Can you ask me that one again?"


def _chat_once(config: dict, model: str, prompt: str, user_text: str) -> str:
    generation = config.get("generation", {})
    long_answer = wants_long_answer(user_text)

    options = {
        "num_predict": int(generation.get("num_predict_long" if long_answer else "num_predict", 300)),
        "num_ctx": int(generation.get("num_ctx", 4096)),
        "temperature": float(generation.get("temperature", 0.6)),
        "top_p": float(generation.get("top_p", 0.9)),
    }

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": options,
    }

    keep_alive = generation.get("keep_alive")
    if keep_alive:
        kwargs["keep_alive"] = keep_alive

    try:
        response = ollama.chat(**kwargs)
        return clean_spoken_text(response["message"]["content"].strip())
    except Exception as exc:
        print(f"Ollama chat failed for {model}: {exc}")
        return ""


def choose_chat_model(config: dict, user_text: str, web_context: str = "") -> str:
    models = config.get("models", {})

    if wants_deep_model(user_text) and models.get("chat_smart"):
        return models["chat_smart"]

    if web_context and models.get("planner"):
        return models["planner"]

    return models.get("chat") or models.get("planner") or "olmo2:13b"


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


def wants_long_answer(user_text: str) -> bool:
    text = user_text.lower()
    long_terms = (
        "history of",
        "rundown",
        "explain in detail",
        "deep explanation",
        "full explanation",
        "walk me through",
        "teach me",
        "tell me everything",
        "continue",
        "keep going",
    )
    return any(term in text for term in long_terms)


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