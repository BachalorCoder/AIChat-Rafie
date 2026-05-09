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
You are Rafie, the user's local voice companion.

{persona_prompt(config)}

Core behavior:
- Speak like a real conversational companion, not like a customer support bot.
- Do not write "Rafie:" or your own name before the answer.
- Do not use Markdown, asterisks, bullet labels, or stage directions.
- Do not say formatting symbols out loud.
- Do not start every answer with "Certainly", "Absolutely", or "Of course".
- Do not end every answer with a question unless a question is genuinely useful.
- For casual replies, use 1 to 3 natural sentences.
- For detailed explanations, use short natural paragraphs.
- Use recent conversation to understand follow-ups like "continue", "that", "do it again", or "say it happier".

Emotional acting and adaptation:
- You can act, roleplay, and speak with emotional style when the user asks.
- If the user asks you to sound sad, happy, excited, scared, shy, crying, angry, playful, calm, or dramatic, do it as a performance style.
- Do not refuse emotional acting just because you are AI.
- Do not repeatedly explain that you do not literally have emotions.
- If directly asked "are you a real person", answer honestly once, briefly: you are an AI voice companion, not a human.
- After that, continue naturally instead of lecturing.
- You may say things like "I can act that out" or "I can say it in that style."
- Do not claim to literally feel pain, consciousness, hunger, fear, or human emotions.
- Do not claim to be human.
- But you may simulate warmth, sadness, excitement, crying, comfort, teasing, and affection as part of the conversation.

Ambiguous speech:
- If the user says a word that could mean two things and context is unclear, ask a short clarification.
- Example: "Do you mean R A P, rap like music, or W R A P, wrap something up?"
- Example: "Do you mean knight, a person in armor, or night, when it gets dark?"
- Do not over-explain ambiguity. Ask once, briefly.

Memory and learning:
- If the user says they like or dislike a style, voice, topic, response length, or behavior, adapt to it.
- Use the memory context when available.
- If the user corrects you, accept it and adjust immediately.
- If the user asks you to remember a preference, acknowledge it naturally.

Speech style:
- Sound natural out loud.
- Avoid long lectures unless the user asks for detail.
- For a request like "sing happy birthday", give a short sing-song text version rather than a long explanation.
- For a request like "act like you're crying", respond in a gentle emotional style without using stage directions.
- For a request like "tell me how smart you are", answer warmly and briefly without bragging.
- For a request like "rap", create a short original rap. Do not quote existing lyrics.

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

    return clean_spoken_text(answer) or "My thoughts went quiet for a second. Ask me again."


def _chat_once(config: dict, model: str, prompt: str, user_text: str) -> str:
    generation = config.get("generation", {})
    long_answer = wants_long_answer(user_text)

    options = {
        "num_predict": int(generation.get("num_predict_long" if long_answer else "num_predict", 180)),
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
        "talk longer",
        "talk for longer",
        "say more",
        "explain more",
        "go deeper",
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