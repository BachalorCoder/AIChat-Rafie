import random

from localagent.chat_mode import answer_chat, needs_screen
from localagent.calculator import maybe_answer_math
from localagent.json_repair import fallback_text_response, parse_json_response
from localagent.knowledge import maybe_answer_builtin
from localagent.memory_store import LocalMemory
from localagent.preferences import maybe_handle_preference
from localagent.plugins import detect_plugin, enforce_plugin_safety, plugin_context
from localagent.session_state import ConversationSession
from localagent.tts_manager import InterruptibleTTS
from localagent.transcript import clean_command
from localagent.wake_listener import VoskWakeListener, contains_phrase
from localagent.web_search import format_web_context, needs_web_search, response_seems_uncertain, web_search
from main import (
    ask_vision_model,
    build_spoken_response,
    load_config,
    maybe_remember,
    print_agent_response,
    take_screenshot,
)


def main():
    config = load_config()
    listener = VoskWakeListener(config)
    tts = InterruptibleTTS(config)
    memory = LocalMemory(config)
    voice_config = config.get("voice", {})
    sleep_phrases = voice_config.get("sleep_phrases", ["rafie goodnight"])
    greeting_lines = voice_config.get("greeting_lines", ["I'm here. What do you need?"])
    sleep_lines = voice_config.get("sleep_lines", ["Goodnight. I'll listen quietly."])

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
            tts.speak(random.choice(greeting_lines), wait=True)
            session = ConversationSession()

            while True:
                raw_command = immediate_command or listener.listen_for_command()
                immediate_command = ""
                command = clean_command(raw_command)

                if not command:
                    print("No command heard. Going back to sleep.")
                    break

                if raw_command.strip() and raw_command.strip() != command:
                    print(f"Heard raw: {raw_command}")
                    print(f"Cleaned: {command}")
                else:
                    print(f"Heard: {command}")

                if contains_phrase(raw_command, sleep_phrases) or contains_phrase(command, sleep_phrases):
                    line = random.choice(sleep_lines)
                    print(line)
                    tts.speak(line, wait=True)
                    break

                if needs_screen(command):
                    handle_screen_command(config, memory, tts, command, session)
                else:
                    handle_chat_command(config, memory, tts, command, session)

    except KeyboardInterrupt:
        print("\nStopping Rafie background listener.")
    finally:
        tts.stop()


def handle_chat_command(config, memory, tts, command, session):
    builtin_response = maybe_answer_builtin(command)
    if builtin_response:
        response = builtin_response
    else:
        math_response = maybe_answer_math(command, session)
        response = math_response

    if response:
        pass
    else:
        preference_response = maybe_handle_preference(command, memory)
        if preference_response:
            response = preference_response
            session.add_turn(command, response)
            print(f"Rafie: {response}")
            tts.speak(response, wait=True)
            return

        memory_context = memory.format_context(command)
        web_context = build_web_context(config, command) if needs_web_search(command) else ""
        response = answer_chat(config, command, memory_context, session=session, web_context=web_context)

        if response_seems_uncertain(response):
            web_context = build_web_context(config, command)
            if web_context:
                response = answer_chat(config, command, memory_context, session=session, web_context=web_context)

    session.add_turn(command, response)
    print(f"Rafie: {response}")
    memory.add(
        f"User said: {command}\nRafie answered: {response}",
        {"source": "background_voice", "mode": "chat", "plugin": "conversation"},
    )
    tts.speak(response, wait=True)


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
    memory_context = memory.format_context(command)
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
        extra_context,
    )

    try:
        parsed = parse_json_response(raw_answer)
    except Exception:
        parsed = fallback_text_response(raw_answer, mode=mode, plugin=plugin.plugin_id)

    parsed = enforce_plugin_safety(plugin, parsed)
    print_agent_response(raw_answer, parsed, config)
    maybe_remember(memory, command, parsed)
    spoken = build_spoken_response(parsed, config)
    session.add_turn(command, spoken)
    tts.speak(spoken, wait=True)


if __name__ == "__main__":
    main()
