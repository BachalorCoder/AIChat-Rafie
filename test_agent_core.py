from localagent.actions import normalize_hotkey_keys
from localagent.calculator import maybe_answer_math, parse_calculation
from localagent.chat_mode import needs_screen
from localagent.json_repair import parse_json_response
from localagent.knowledge import maybe_answer_builtin
from localagent.preferences import maybe_handle_preference
from localagent.plugins import detect_plugin, enforce_plugin_safety
from localagent.session_state import ConversationSession
from localagent.transcript import clean_command
from localagent.wake_listener import contains_phrase, text_after_phrase
from localagent.web_search import DuckDuckGoLiteParser, needs_web_search
from main import build_spoken_response


def test_json_repair_handles_fences_and_trailing_commas():
    messy = """
    Here you go:
    ```json
    {
      screen_summary: "VS Code is open",
      "user_goal_interpretation": "describe",
      "suggested_next_step": "answer only",
      "spoken_response": "I can see VS Code open.",
      "action": {
        "type": "none",
        "risk": "low",
      },
    }
    ```
    """
    parsed = parse_json_response(messy)
    assert parsed["screen_summary"] == "VS Code is open"
    assert parsed["spoken_response"] == "I can see VS Code open."
    assert parsed["action"]["type"] == "none"


def test_hotkey_normalization():
    assert normalize_hotkey_keys(["Windows", "S", "control", "return"]) == [
        "win",
        "s",
        "ctrl",
        "enter",
    ]


def test_game_plugin_forces_coach_mode():
    plugin = detect_plugin("review this chess position")
    parsed = {
        "screen_summary": "board",
        "user_goal_interpretation": "play",
        "suggested_next_step": "move",
        "action": {"type": "click", "x": 10, "y": 10, "risk": "low"},
    }
    safe = enforce_plugin_safety(plugin, parsed)
    assert safe["mode"] == "coach"
    assert safe["action"]["type"] == "none"


def test_spoken_response_fallback_uses_screen_summary():
    spoken = build_spoken_response(
        {
            "screen_summary": "VS Code is open with a terminal at the bottom.",
            "suggested_next_step": "No action is needed.",
        },
        {"persona": {"fallback_prefix": "I'm with you. Here's what I see:"}},
    )
    assert spoken.startswith("I'm with you. Here's what I see:")
    assert "VS Code is open" in spoken


def test_wake_phrase_variants():
    text = "hey rafie wake up what is on my screen"
    phrase = contains_phrase(text, ["rafie wake up", "raffy wake up"])
    assert phrase == "rafie wake up"
    assert text_after_phrase(text, phrase) == "what is on my screen"


def test_transcript_cleanup_removes_vosk_fillers():
    assert clean_command("the what is ten times ten") == "what is ten times ten"
    assert clean_command("meet the steps") == "show me the steps"


def test_calculator_handles_spoken_math_and_followups():
    session = ConversationSession()
    calc = parse_calculation("what is one hundred times two hundred and fifty one")
    assert calc["result"] == 25100

    first = maybe_answer_math("the what is ten times ten", session)
    assert "100" in first

    steps = maybe_answer_math("meet the steps", session)
    assert "10 times 10" in steps
    assert "100" in steps


def test_screen_routing_does_not_match_apple_or_games_by_substring():
    assert needs_screen("what is your favorite apple") is False
    assert needs_screen("my favorite apple is red apple") is False
    assert needs_screen("let's play a game") is False
    assert needs_screen("what is on my screen") is True


class FakeMemory:
    def __init__(self):
        self.items = []

    def add(self, text, metadata=None):
        self.items.append((text, metadata or {}))


def test_preference_memory():
    memory = FakeMemory()
    response = maybe_handle_preference("my favorite apple is red apple", memory)
    assert "favorite apple" in response
    assert memory.items


def test_web_search_detection_and_parser():
    assert needs_web_search("look up the latest ollama release")
    assert needs_web_search("what is the farthest planet")
    html = """
    <html><body>
      <a class="result-link" href="https://example.com">Example Title</a>
      <td class="result-snippet">Example snippet text.</td>
    </body></html>
    """
    parser = DuckDuckGoLiteParser()
    parser.feed(html)
    assert parser.results[0].title == "Example Title"
    assert parser.results[0].snippet == "Example snippet text."


def test_builtin_algebra_answer():
    answer = maybe_answer_builtin("can you give me a rundown of how algebra works")
    assert "x plus 3 equals 7" in answer


if __name__ == "__main__":
    test_json_repair_handles_fences_and_trailing_commas()
    test_hotkey_normalization()
    test_game_plugin_forces_coach_mode()
    test_spoken_response_fallback_uses_screen_summary()
    test_wake_phrase_variants()
    test_transcript_cleanup_removes_vosk_fillers()
    test_calculator_handles_spoken_math_and_followups()
    test_screen_routing_does_not_match_apple_or_games_by_substring()
    test_preference_memory()
    test_web_search_detection_and_parser()
    test_builtin_algebra_answer()
    print("agent core smoke tests passed")
