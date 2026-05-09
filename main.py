import json
from pathlib import Path

import mss
import ollama
from PIL import Image

from localagent.actions import safe_execute
from localagent.browser_mode import BrowserSession
from localagent.json_repair import fallback_text_response, parse_json_response
from localagent.memory_store import LocalMemory
from localagent.persona import fallback_spoken_prefix, persona_prompt
from localagent.plugins import (
    detect_plugin,
    enforce_plugin_safety,
    list_plugins,
    plugin_context,
    plugin_prompt,
)
from localagent.tts_manager import InterruptibleTTS
from localagent.voice_input import listen_once


ROOT = Path(r"G:\LocalAgent")
CONFIG_PATH = ROOT / "config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def take_screenshot(config):
    screenshots_dir = Path(config["paths"]["screenshots"])
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    shot_path = screenshots_dir / "current_screen.png"
    model_shot_path = screenshots_dir / "current_screen_for_model.jpg"

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save(shot_path)

    vision_config = config.get("vision", {})
    max_width = int(vision_config.get("max_image_width", 1600))
    image_quality = int(vision_config.get("image_quality", 88))
    model_img = img
    if max_width > 0 and img.width > max_width:
        scale = max_width / img.width
        new_size = (max_width, max(1, int(img.height * scale)))
        model_img = img.resize(new_size, Image.Resampling.LANCZOS)
    model_img.save(model_shot_path, quality=image_quality, optimize=True)

    return shot_path, model_shot_path


def ask_vision_model(config, screenshot_path, user_goal, mode, plugin, memory_context, extra_context):
    model = config["models"]["vision"]
    plugin_rules = plugin_prompt(plugin)
    persona_rules = persona_prompt(config)

    prompt = f"""
You are LocalAgent, a local screen-understanding assistant.
{persona_rules}

The user's request has ALREADY been submitted to you.
Do NOT suggest typing the user's request again.
Do NOT suggest pressing Enter to submit the current LocalAgent prompt.
Do NOT suggest typing YES into the LocalAgent terminal.
Do NOT control the terminal that is running LocalAgent unless the user explicitly asks you to.

Current mode: {mode}
{plugin_rules}

The user goal is:
{user_goal}

{memory_context}

{extra_context}

Look at the screenshot and return JSON only.

Important behavior:
- First describe what is visible.
- Then suggest a useful next step.
- Also write spoken_response as a warm, natural voice reply to the user.
- spoken_response should sound like a person talking, not an internal instruction or prompt.
- spoken_response should follow the assistant persona above.
- For screen-description requests, spoken_response should include the visible screen summary.
- Keep spoken_response concise: usually 1 to 3 sentences.
- If the safest next step is just advice, use action type "none".
- If the user asks to open VS Code search, prefer the hotkey Ctrl+Shift+F.
- If the user asks to open Windows Search, Start search, or the taskbar search box, prefer the hotkey Win+S.
- If the user asks to open browser address/search bar, prefer Ctrl+L.
- If the user asks to switch apps, prefer Alt+Tab only after confirmation.
- Do not click/type unless the user clearly wants computer control.
- In coach mode, never click, type, hotkey, scroll, or otherwise control the computer.
- Do not suggest cheating in online competitive games.
- Do not automate passwords, purchases, deleting files, sending messages, forms, or ranked/online game actions.
- If action is not clearly safe, set action type to "none".

Return JSON only in this exact shape:

{{
  "screen_summary": "what is visible",
  "user_goal_interpretation": "what the user likely wants",
  "suggested_next_step": "safe next step",
  "spoken_response": "natural voice reply to say out loud to the user",
  "mode": "{mode}",
  "plugin": "{plugin.plugin_id}",
  "action": {{
    "type": "none | click | type | hotkey | scroll",
    "x": null,
    "y": null,
    "text": null,
    "keys": [],
    "reason": "why this action would help",
    "risk": "low | medium | high"
  }}
}}

Examples:
For opening VS Code search:
{{
  "type": "hotkey",
  "x": null,
  "y": null,
  "text": null,
  "keys": ["ctrl", "shift", "f"],
  "reason": "Ctrl+Shift+F opens VS Code global search without needing mouse coordinates.",
  "risk": "low"
}}

For opening Windows Search:
{{
  "type": "hotkey",
  "x": null,
  "y": null,
  "text": null,
  "keys": ["win", "s"],
  "reason": "Win+S opens the Windows search bar without typing into the LocalAgent terminal.",
  "risk": "low"
}}

For only describing or coaching:
{{
  "type": "none",
  "x": null,
  "y": null,
  "text": null,
  "keys": [],
  "reason": "The user asked for observation/advice only.",
  "risk": "low"
}}
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(screenshot_path)],
            }
        ],
    )

    return response["message"]["content"]


def ask_browser_model(config, snapshot, user_goal, memory_context):
    model = config["models"]["planner"]
    persona_rules = persona_prompt(config)
    prompt = f"""
You are LocalAgent browser mode.
{persona_rules}

Use Playwright-style browser actions only for websites.
Do not submit forms, send messages, purchase items, delete data, or enter passwords.
If the next step is advice or observation, use action type "none".
If a browser action is safe, use action type "browser" with method "goto", "click", "fill", "press", or "extract".

User goal:
{user_goal}

{memory_context}

Current page:
URL: {snapshot.url}
Title: {snapshot.title}
Visible text excerpt:
{snapshot.text[:8000]}

Return JSON only:
{{
  "screen_summary": "page summary",
  "user_goal_interpretation": "what the user likely wants",
  "suggested_next_step": "safe browser next step",
  "spoken_response": "natural voice reply to say out loud to the user",
  "mode": "browser",
  "plugin": "browser",
  "action": {{
    "type": "none | browser",
    "method": "none | goto | click | fill | press | extract",
    "selector": null,
    "url": null,
    "text": null,
    "key": null,
    "reason": "why this action would help",
    "risk": "low | medium | high"
  }}
}}
"""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def handle_browser_command(config, command, memory, tts, browser):
    target = command.removeprefix("/browser").strip()

    if target.lower() in {"close", "quit", "exit"}:
        if browser:
            browser.close()
            print("Browser closed.")
        return None

    if browser is None:
        browser = BrowserSession(headless=bool(config.get("browser", {}).get("headless", False)))

    try:
        if target and _looks_like_url(target):
            snapshot = browser.open(target)
            user_goal = f"Open and inspect {target}"
        else:
            snapshot = browser.snapshot()
            user_goal = target or input("Browser goal: ").strip()

        memory_context = memory.format_context(user_goal)
        raw_answer = ask_browser_model(config, snapshot, user_goal, memory_context)
        try:
            parsed = parse_json_response(raw_answer)
        except Exception as exc:
            print("Could not parse JSON from model response.")
            print(exc)
            parsed = fallback_text_response(raw_answer, mode="browser", plugin="browser")
        print_agent_response(raw_answer, parsed, config)
        maybe_remember(memory, user_goal, parsed)

        speak_parsed_response(tts, parsed, config)

        safe_execute(parsed.get("action", {"type": "none"}), user_goal, config, mode="browser", browser=browser)
    except Exception as exc:
        print(f"Browser mode error: {exc}")

    return browser


def print_agent_response(raw_answer, parsed, config=None):
    print("\nRaw model answer:")
    print(raw_answer)
    print()

    print("Screen summary:")
    print(parsed.get("screen_summary", ""))

    print("\nSuggested next step:")
    print(parsed.get("suggested_next_step", ""))

    print("\nSpoken response:")
    print(build_spoken_response(parsed, config))

    print("\nProposed action:")
    print(json.dumps(parsed.get("action", {"type": "none"}), indent=2))


def build_spoken_response(parsed, config=None):
    spoken = (parsed.get("spoken_response") or "").strip()
    if spoken:
        return spoken

    screen_summary = (parsed.get("screen_summary") or "").strip()
    next_step = (parsed.get("suggested_next_step") or "").strip()
    prefix = fallback_spoken_prefix(config)

    if screen_summary and next_step:
        return f"{prefix} {screen_summary} {next_step}"
    if screen_summary:
        return f"{prefix} {screen_summary}"
    if next_step:
        return next_step
    return "I looked at the screen, but I do not have a useful summary yet."


def speak_parsed_response(tts, parsed, config=None):
    spoken = build_spoken_response(parsed, config)
    if spoken:
        tts.speak(spoken)


def maybe_remember(memory, user_goal, parsed):
    summary = parsed.get("spoken_response") or parsed.get("suggested_next_step") or parsed.get("screen_summary")
    if not summary:
        return
    memory.add(
        f"User goal: {user_goal}\nAgent note: {summary}",
        {
            "source": "interaction",
            "mode": parsed.get("mode", "desktop"),
            "plugin": parsed.get("plugin") or "",
        },
    )


def _looks_like_url(text):
    lower = text.lower()
    return lower.startswith(("http://", "https://")) or "." in lower.split()[0]


def print_help():
    print(
        """
Commands:
  /voice                 listen once with Vosk, then run the heard command
  /stop                  stop current TTS playback
  /coach <goal>          force observe/advice-only coach mode
  /browser <url|goal>    use Playwright browser mode; /browser close closes it
  /remember <text>       store a memory in Chroma
  /recall <query>        recall related Chroma memories
  /plugins              list available plugins
  q                      quit
""".strip()
    )


def main():
    config = load_config()
    memory = LocalMemory(config)
    tts = InterruptibleTTS(config)
    browser = None

    print("LocalAgent")
    print("Mode: observe -> suggest -> ask permission")
    print("Type /help for commands. Emergency stop for mouse actions: move mouse to a screen corner.")
    print()

    while True:
        user_goal = input("What do you want me to help with? Type q to quit: ").strip()

        if user_goal.lower() in {"q", "quit", "exit"}:
            break

        if user_goal == "/help":
            print_help()
            continue

        if user_goal == "/plugins":
            print(list_plugins())
            continue

        if user_goal == "/stop":
            tts.stop()
            print("TTS stopped.")
            continue

        if user_goal.startswith("/remember "):
            memory.add(user_goal.removeprefix("/remember ").strip(), {"source": "manual"})
            print("Remembered.")
            continue

        if user_goal.startswith("/recall "):
            memories = memory.recall(user_goal.removeprefix("/recall ").strip())
            if memories:
                print("\n".join(f"- {memory_text}" for memory_text in memories))
            else:
                print("No relevant memory found.")
            continue

        if user_goal.startswith("/browser"):
            browser = handle_browser_command(config, user_goal, memory, tts, browser)
            print("\nDone.\n")
            continue

        explicit_mode = None
        if user_goal == "/voice":
            try:
                heard = listen_once(config)
            except Exception as exc:
                print(f"Voice input error: {exc}")
                continue
            if not heard:
                print("No speech recognized.")
                continue
            print(f"Heard: {heard}")
            user_goal = heard
        elif user_goal.startswith("/coach "):
            explicit_mode = "coach"
            user_goal = user_goal.removeprefix("/coach ").strip()

        plugin = detect_plugin(user_goal, explicit_mode=explicit_mode)
        mode = "coach" if explicit_mode == "coach" or plugin.coach_only else plugin.mode
        memory_context = memory.format_context(user_goal)
        extra_context = plugin_context(plugin, user_goal, config)

        screenshot_path, model_screenshot_path = take_screenshot(config)
        print(f"Screenshot saved: {screenshot_path}")
        print(f"Model image: {model_screenshot_path}")
        print(f"Mode: {mode} | Plugin: {plugin.plugin_id}")

        raw_answer = ask_vision_model(
            config,
            model_screenshot_path,
            user_goal,
            mode,
            plugin,
            memory_context,
            extra_context,
        )

        try:
            parsed = parse_json_response(raw_answer)
        except Exception as exc:
            print("Could not parse JSON from model response.")
            print(exc)
            parsed = fallback_text_response(raw_answer, mode=mode, plugin=plugin.plugin_id)

        parsed = enforce_plugin_safety(plugin, parsed)
        print_agent_response(raw_answer, parsed, config)
        maybe_remember(memory, user_goal, parsed)

        speak_parsed_response(tts, parsed, config)

        safe_execute(parsed.get("action", {"type": "none"}), user_goal, config, mode=mode, plugin=plugin)

        print("\nDone.\n")

    if browser:
        browser.close()
    tts.stop()


if __name__ == "__main__":
    main()
