from __future__ import annotations

import json

import pyautogui


def normalize_hotkey_keys(keys):
    aliases = {
        "windows": "win",
        "windows key": "win",
        "cmd": "win",
        "command": "win",
        "super": "win",
        "control": "ctrl",
        "return": "enter",
        "esc": "escape",
    }

    normalized = []
    for key in keys:
        if not isinstance(key, str):
            continue
        clean_key = key.strip().lower()
        if clean_key:
            normalized.append(aliases.get(clean_key, clean_key))
    return normalized


def safe_execute(action, user_goal, config, mode="desktop", plugin=None, browser=None):
    action_type = action.get("type", "none")
    risk = action.get("risk", "high")

    if mode == "coach" or getattr(plugin, "coach_only", False):
        print("Coach mode is observe/advice only. No action to execute.")
        return

    if action_type == "none":
        print("No action to execute.")
        return

    if risk != "low":
        print("Action blocked because risk is not low.")
        return

    text = action.get("text") or ""
    blocked_self_loop_texts = {
        "YES",
        user_goal.strip(),
        user_goal.strip() + "\n",
    }

    if action_type == "type" and text.strip() in blocked_self_loop_texts:
        print("Blocked likely self-loop typing into the LocalAgent terminal.")
        return

    if _is_blocked_desktop_action(action, config):
        return

    print("\nAbout to execute:")
    print(json.dumps(action, indent=2))

    confirm = input(f"Execute this {action_type} action? Type YES exactly: ")
    if confirm != "YES":
        print("Action cancelled.")
        return

    pyautogui.FAILSAFE = bool(config.get("safety", {}).get("pyautogui_failsafe", True))

    if action_type == "click":
        x = action.get("x")
        y = action.get("y")
        if x is None or y is None:
            print("Missing x/y coordinates.")
            return
        pyautogui.click(x, y)
    elif action_type == "type":
        if not text:
            print("No text provided.")
            return
        parts = text.split("\n")
        for index, part in enumerate(parts):
            if part:
                pyautogui.write(part, interval=0.01)
            if index < len(parts) - 1:
                pyautogui.press("enter")
    elif action_type == "hotkey":
        keys = normalize_hotkey_keys(action.get("keys") or [])
        if keys:
            pyautogui.hotkey(*keys)
        else:
            print("No hotkey keys provided.")
    elif action_type == "scroll":
        pyautogui.scroll(-5)
    elif action_type == "browser":
        if not browser:
            print("No browser session is active.")
            return
        print(browser.execute(action))
    else:
        print(f"Unsupported action type: {action_type}")


def _is_blocked_desktop_action(action, config):
    safety = config.get("safety", {})
    action_type = action.get("type", "none")

    if action_type == "click" and safety.get("ask_before_clicking", True) is False:
        return False
    if action_type == "type" and safety.get("ask_before_typing", True) is False:
        return False

    risky_text = " ".join(
        str(action.get(key) or "") for key in ("text", "reason", "suggested_next_step")
    ).lower()
    blocked_terms = (
        "password",
        "purchase",
        "buy now",
        "delete",
        "send message",
        "submit form",
        "ranked game",
        "online game",
    )
    if any(term in risky_text for term in blocked_terms):
        print("Action blocked by safety policy.")
        return True
    return False

