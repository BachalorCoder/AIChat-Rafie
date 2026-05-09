import json
import random
import re
from pathlib import Path

from localagent.chat_mode import answer_chat, needs_screen
from localagent.calculator import maybe_answer_math
from localagent.disambiguation import maybe_start_disambiguation, resolve_disambiguation
from localagent.json_repair import fallback_text_response, parse_json_response
from localagent.knowledge import maybe_answer_builtin
from localagent.memory_store import LocalMemory
from localagent.preferences import maybe_handle_preference
from localagent.plugins import detect_plugin, enforce_plugin_safety, plugin_context
from localagent.session_state import ConversationSession
from localagent.transcript import clean_command, is_likely_background_noise
from localagent.tts_manager import InterruptibleTTS, clean_spoken_text
from localagent.wake_listener import VoskWakeListener, contains_phrase
from localagent.web_search import (
    format_web_context,
    needs_web_search,
    response_seems_uncertain,
    web_search
)
from main import (
    ask_vision_model,
    build_spoken_response,
    load_config,
    maybe_remember,
    print_agent_response,
    take_screenshot
)


DEFAULT_STOP_TALKING_PHRASES = [
    "rafie stop",
    "rafi stop",
    "raffy stop",
    "raphie stop",
    "stop talking",
    "pause",
    "pause talking",
    "be quiet",
    "hold on",
    "wait stop",
    "stop please",
    "stop for a second",
    "quiet",
    "shush"
]

DEFAULT_CONTINUE_PHRASES = [
    "continue",
    "rafie continue",
    "rafi continue",
    "raffy continue",
    "raphie continue",
    "keep going",
    "rafie keep going",
    "rafi keep going",
    "go on",
    "finish what you were saying",
    "continue what you were saying",
    "continue about",
    "say the rest",
    "finish your thought"
]


def main():
    config = load_config()
    listener = VoskWakeListener(config)
    tts = InterruptibleTTS(config)
    memory = LocalMemory(config)

    if config.get("tts", {}).get("preload", False):
        tts.preload(wait=False)

    voice_config = config.get("voice", {})

    sleep_phrases = voice_config.get("sleep_phrases", ["rafie goodnight"])

    stop_talking_phrases = voice_config.get(
        "stop_talking_phrases",
        DEFAULT_STOP_TALKING_PHRASES
    )

    continue_phrases = voice_config.get(
        "continue_phrases",
        DEFAULT_CONTINUE_PHRASES
    )

    greeting_lines = voice_config.get(
        "greeting_lines",
        ["I'm here. What do you need?"]
    )

    sleep_lines = voice_config.get(
        "sleep_lines",
        ["Goodnight. I'll listen quietly."]
    )

    print("Rafie background listener is running.")
    print("Say: Rafie wake up")
    print("Then ask your question. Say: Rafie goodnight to put her back to sleep.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            print("\nSleeping. Waiting for wake phrase...")

            immediate_command = listener.wait_for_wake()
            immediate_command = clean_command(immediate_command)

            print("Wake phrase heard.")
            tts.speak(random.choice(greeting_lines), wait=False)

            session = ConversationSession()
            pending_disambiguation = None

            while True:
                priority_phrases = sleep_phrases + stop_talking_phrases + continue_phrases

                raw_command = immediate_command or listener.listen_for_command(
                    priority_phrases=priority_phrases if tts.is_speaking() else []
                )

                immediate_command = ""

                command = clean_command(raw_command)

                if not command:
                    if tts.is_speaking():
                        continue

                    print("No command heard. Going back to sleep.")
                    break

                print_heard(raw_command, command)

                is_sleep = bool(
                    contains_phrase(raw_command, sleep_phrases)
                    or contains_phrase(command, sleep_phrases)
                )

                is_stop = bool(
                    contains_phrase(raw_command, stop_talking_phrases)
                    or contains_phrase(command, stop_talking_phrases)
                )

                is_continue = bool(
                    contains_phrase(raw_command, continue_phrases)
                    or contains_phrase(command, continue_phrases)
                )

                if is_sleep:
                    pending_disambiguation = None
                    tts.interrupt(save=False)
                    line = random.choice(sleep_lines)
                    print(line)
                    tts.speak(line, wait=True)
                    break

                if is_stop:
                    pending_disambiguation = None
                    tts.interrupt(save=True)
                    print("Speech paused.")
                    continue

                if is_continue:
                    pending_disambiguation = None

                    if tts.continue_speaking(wait=False):
                        print("Speech continued.")
                        continue

                    continue_command = (
                        "Continue your previous answer. "
                        "Add the next useful part without repeating yourself."
                    )
                    handle_chat_command(config, memory, tts, continue_command, session)
                    continue

                voice_switch_response = maybe_handle_voice_switch(config, tts, command)
                if voice_switch_response:
                    pending_disambiguation = None
                    print(f"Rafie: {voice_switch_response}")
                    tts.speak(voice_switch_response, wait=False)
                    continue

                if pending_disambiguation:
                    if is_likely_background_noise(command, config):
                        print(f"Ignored likely background noise while waiting for clarification: '{command}'")
                        continue

                    resolved_command = resolve_disambiguation(pending_disambiguation, command)

                    if resolved_command:
                        print(f"Resolved command: {resolved_command}")
                        command = clean_command(resolved_command)
                        pending_disambiguation = None
                        skip_next_disambiguation = True
                    else:
                        prompt = pending_disambiguation.get(
                            "prompt",
                            "Which one did you mean?"
                        )
                        print(f"Rafie: {prompt}")
                        tts.speak(prompt, wait=False)
                        continue
                else:
                    skip_next_disambiguation = False

                if should_ignore_command(config, tts, command):
                    print(f"Ignored likely background noise: '{command}'")
                    continue

                if not skip_next_disambiguation:
                    ambiguity = maybe_start_disambiguation(command)
                    if ambiguity:
                        pending_disambiguation = ambiguity
                        prompt = ambiguity["prompt"]
                        print(f"Rafie: {prompt}")
                        tts.speak(prompt, wait=False)
                        continue

                if tts.is_speaking():
                    tts.interrupt(save=True)

                if needs_screen(command):
                    handle_screen_command(config, memory, tts, command, session)
                else:
                    handle_chat_command(config, memory, tts, command, session)

    except KeyboardInterrupt:
        print("\nStopping Rafie background listener.")
    finally:
        tts.stop()


def maybe_handle_voice_switch(config, tts, command: str) -> str | None:
    text = normalize_voice_command(command)
    profiles = sorted(config.get("tts", {}).get("voice_profiles", {}).keys())

    if not profiles:
        return None

    if text in {
        "list",
        "list voice",
        "list voices",
        "show voice",
        "show voices",
        "voice list",
        "what voices",
        "what voices do you have"
    }:
        return "I can use these voices: " + ", ".join(profiles) + "."

    if text in {
        "switch voice",
        "switch your voice",
        "change voice",
        "change your voice",
        "use another voice",
        "try another voice"
    }:
        return "Which voice should I use? You can say voice one, two, three, or four."

    profile = extract_voice_profile(text, profiles)

    if profile:
        ok, message = tts.set_voice_profile(profile)
        return message

    return None


def normalize_voice_command(command: str) -> str:
    text = command.lower().strip()

    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    assistant_names = ["rafie", "rafi", "raffy", "raphie"]

    words = text.split()

    while words and words[0] in {"the", "a", "to", "uh", "um", "hey", "okay", "ok"}:
        words.pop(0)

    while words and words[0] in assistant_names:
        words.pop(0)

    while words and words[-1] in {"the", "a", "to", "uh", "um", "okay", "ok"}:
        words.pop()

    text = " ".join(words).strip()

    replacements = {
        "won": "one",
        "wun": "one",
        "too": "two",
        "to": "two",
        "tree": "three",
        "free": "three",
        "for": "four"
    }

    words = [replacements.get(word, word) for word in text.split()]
    return " ".join(words).strip()


def extract_voice_profile(text: str, profiles: list[str]) -> str | None:
    words = text.split()

    for profile in profiles:
        if profile in words:
            if "voice" in words or "switch" in words or "change" in words or "use" in words:
                return profile

    patterns = [
        r"switch voice (?P<profile>[a-z0-9]+)",
        r"switch voice to (?P<profile>[a-z0-9]+)",
        r"switch to voice (?P<profile>[a-z0-9]+)",
        r"change voice (?P<profile>[a-z0-9]+)",
        r"change voice to (?P<profile>[a-z0-9]+)",
        r"use voice (?P<profile>[a-z0-9]+)",
        r"use the voice (?P<profile>[a-z0-9]+)",
        r"set voice (?P<profile>[a-z0-9]+)",
        r"set voice to (?P<profile>[a-z0-9]+)",
        r"voice (?P<profile>[a-z0-9]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            profile = match.group("profile").strip()

            if profile in profiles:
                return profile

    return None


def print_heard(raw_command: str, command: str) -> None:
    raw = (raw_command or "").strip()

    if raw and raw != command:
        print(f"Heard raw: {raw_command}")
        print(f"Cleaned: {command}")
    else:
        print(f"Heard: {command}")


def should_ignore_command(config: dict, tts: InterruptibleTTS, command: str) -> bool:
    normalized = command.lower().strip()
    words = set(normalized.split())

    assistant_names = {"rafi", "rafie", "raffy", "raphie"}

    important_words = {
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
        "happy",
        "sad",
        "rap",
        "wrap"
    }

    question_words = {
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

    if words & assistant_names and len(words) > 1:
        return False

    if words & important_words:
        return False

    first_word = normalized.split()[0] if normalized.split() else ""
    if first_word in question_words:
        return False

    if is_likely_background_noise(command, config):
        return True

    if tts.is_speaking() and tts.looks_like_echo(command):
        return True

    return False


def handle_chat_command(config, memory, tts, command, session):
    builtin_response = maybe_answer_builtin(command)

    if builtin_response:
        response = builtin_response
    else:
        math_response = maybe_answer_math(command, session)
        response = math_response

    if response:
        say_response(config, memory, tts, command, response, session)
        return

    preference_response = maybe_handle_preference(command, memory)

    if preference_response:
        say_response(config, memory, tts, command, preference_response, session)
        return

    memory_context = build_memory_context(config, memory, command)
    web_context = build_web_context(config, command) if needs_web_search(command) else ""

    response = answer_chat(
        config,
        command,
        memory_context,
        session=session,
        web_context=web_context
    )

    if response_seems_uncertain(response):
        web_context = build_web_context(config, command)

        if web_context:
            response = answer_chat(
                config,
                command,
                memory_context,
                session=session,
                web_context=web_context
            )

    say_response(config, memory, tts, command, response, session)


def say_response(config, memory, tts, command, response, session):
    response = clean_spoken_text(response)

    session.add_turn(command, response)

    print(f"Rafie: {response}")

    memory.add(
        f"User said: {command}\nRafie answered: {response}",
        {
            "source": "background_voice",
            "mode": "chat",
            "plugin": "conversation"
        }
    )

    append_last_conversation(config, command, response)

    tts.speak(response, wait=False)


def build_memory_context(config, memory, command: str) -> str:
    parts = []

    try:
        stored_memory = memory.format_context(command)
        if stored_memory:
            parts.append(stored_memory)
    except Exception as exc:
        print(f"Memory context failed: {exc}")

    last_conversation = load_last_conversation_context(config)
    if last_conversation:
        parts.append(last_conversation)

    return "\n\n".join(parts).strip()


def conversation_history_path(config: dict) -> Path:
    root = Path(config["paths"]["project_root"])
    path = root / "memory" / "last_conversation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_last_conversation_context(config: dict, limit: int = 16) -> str:
    path = conversation_history_path(config)

    if not path.exists():
        return ""

    try:
        turns = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if not isinstance(turns, list):
        return ""

    turns = turns[-limit:]

    lines = ["Recent saved conversation context:"]

    for turn in turns:
        user_text = turn.get("user", "").strip()
        rafie_text = turn.get("rafie", "").strip()

        if user_text:
            lines.append(f"User: {user_text}")

        if rafie_text:
            lines.append(f"Rafie: {rafie_text}")

    return "\n".join(lines).strip()


def append_last_conversation(config: dict, user_text: str, rafie_text: str, limit: int = 80) -> None:
    path = conversation_history_path(config)

    try:
        if path.exists():
            turns = json.loads(path.read_text(encoding="utf-8"))
        else:
            turns = []

        if not isinstance(turns, list):
            turns = []

        turns.append(
            {
                "user": user_text,
                "rafie": rafie_text
            }
        )

        turns = turns[-limit:]

        path.write_text(json.dumps(turns, indent=2, ensure_ascii=False), encoding="utf-8")

    except Exception as exc:
        print(f"Could not save last conversation: {exc}")


def build_web_context(config, command):
    web_config = config.get("web_search", {})

    if not web_config.get("enabled", True):
        return ""

    try:
        results = web_search(command, max_results=int(web_config.get("max_results", 4)))
    except Exception as exc:
        print(f"Web search failed: {exc}")
        return ""

    if results:
        print("Web results:")

        for result in results:
            print(f"- {result.title}: {result.url}")

    return format_web_context(results)


def handle_screen_command(config, memory, tts, command, session):
    plugin = detect_plugin(command)
    mode = "coach" if plugin.coach_only else plugin.mode

    memory_context = build_memory_context(config, memory, command)
    extra_context = plugin_context(plugin, command, config)

    screenshot_path, model_screenshot_path = take_screenshot(config)

    print(f"Screenshot saved: {screenshot_path}")
    print(f"Model image: {model_screenshot_path}")
    print(f"Mode: {mode} | Plugin: {plugin.plugin_id}")

    raw_answer = ask_vision_model(
        config,
        model_screenshot_path,
        command,
        mode,
        plugin,
        memory_context,
        extra_context
    )

    try:
        parsed = parse_json_response(raw_answer)
    except Exception:
        parsed = fallback_text_response(raw_answer, mode=mode, plugin=plugin.plugin_id)

    parsed = enforce_plugin_safety(plugin, parsed)

    print_agent_response(raw_answer, parsed, config)
    maybe_remember(memory, command, parsed)

    spoken = clean_spoken_text(build_spoken_response(parsed, config))

    session.add_turn(command, spoken)
    append_last_conversation(config, command, spoken)

    tts.speak(spoken, wait=False)


if __name__ == "__main__":
    main()