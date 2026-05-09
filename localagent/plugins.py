from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class AgentPlugin:
    plugin_id: str
    name: str
    aliases: tuple[str, ...]
    mode: str
    coach_only: bool
    description: str
    prompt_rules: tuple[str, ...]


PLUGINS: tuple[AgentPlugin, ...] = (
    AgentPlugin(
        plugin_id="chess",
        name="Chess Coach",
        aliases=("chess", "stockfish", "fen", "pgn", "checkmate", "opening"),
        mode="coach",
        coach_only=True,
        description="Chess study and position review using observation and advice only.",
        prompt_rules=(
            "Give chess advice, candidate moves, plans, and tactical themes.",
            "Never click, type, or play moves in an online/ranked game.",
            "If a FEN appears in the user goal, you may reference the engine hint supplied in context.",
        ),
    ),
    AgentPlugin(
        plugin_id="checkers",
        name="Checkers Coach",
        aliases=("checkers", "draughts", "king me", "checkerboard"),
        mode="coach",
        coach_only=True,
        description="Checkers study and board-shape advice without control.",
        prompt_rules=(
            "Give strategy advice such as trades, king races, tempo, traps, and safe moves.",
            "Never click, type, or play moves in an online/ranked game.",
        ),
    ),
    AgentPlugin(
        plugin_id="poker_study",
        name="Poker Study",
        aliases=("poker", "holdem", "hold'em", "flop", "turn", "river", "range", "pot odds"),
        mode="coach",
        coach_only=True,
        description="Poker study, hand review, and odds/range coaching.",
        prompt_rules=(
            "Only help with study, review, and education.",
            "Do not advise actions during live online real-money or ranked poker hands.",
            "Prefer explaining ranges, pot odds, blockers, and post-hand review reasoning.",
        ),
    ),
    AgentPlugin(
        plugin_id="shooter_vod",
        name="Shooter VOD Review",
        aliases=("vod", "shooter", "fps", "aim", "crosshair", "valorant", "counter-strike", "overwatch"),
        mode="coach",
        coach_only=True,
        description="Shooter gameplay VOD review: positioning, awareness, crosshair, and rotations.",
        prompt_rules=(
            "Review visible gameplay and give coaching notes only.",
            "Do not automate inputs or aim.",
            "Do not provide cheating, exploit, macro, recoil-script, or anti-cheat evasion advice.",
        ),
    ),
    AgentPlugin(
        plugin_id="generic_desktop",
        name="Generic Desktop Control",
        aliases=("desktop", "windows", "app", "computer", "screen", "click", "type", "hotkey"),
        mode="desktop",
        coach_only=False,
        description="General observe, suggest, and permission-gated desktop control.",
        prompt_rules=(
            "Prefer hotkeys over coordinates when possible.",
            "Use action type none when advice is enough.",
            "Never automate passwords, purchases, destructive file operations, sending messages, or forms.",
        ),
    ),
)


PLUGIN_BY_ID = {plugin.plugin_id: plugin for plugin in PLUGINS}


def list_plugins() -> str:
    lines = []
    for plugin in PLUGINS:
        suffix = "coach-only" if plugin.coach_only else "permission-gated control"
        lines.append(f"- {plugin.plugin_id}: {plugin.name} ({suffix}) - {plugin.description}")
    return "\n".join(lines)


def detect_plugin(user_goal: str, explicit_mode: str | None = None) -> AgentPlugin:
    lower_goal = user_goal.lower()

    if explicit_mode == "coach":
        for plugin in PLUGINS:
            if plugin.plugin_id != "generic_desktop" and _matches(plugin, lower_goal):
                return plugin
        return PLUGIN_BY_ID["shooter_vod"] if "vod" in lower_goal else PLUGIN_BY_ID["generic_desktop"]

    for plugin in PLUGINS:
        if plugin.plugin_id != "generic_desktop" and _matches(plugin, lower_goal):
            return plugin

    return PLUGIN_BY_ID["generic_desktop"]


def plugin_prompt(plugin: AgentPlugin) -> str:
    rules = "\n".join(f"- {rule}" for rule in plugin.prompt_rules)
    control_rule = "- This plugin is coach-only: the returned action type must be none." if plugin.coach_only else ""
    return f"""
Active plugin: {plugin.name} ({plugin.plugin_id})
Plugin mode: {plugin.mode}
Plugin description: {plugin.description}
Plugin rules:
{rules}
{control_rule}
""".strip()


def plugin_context(plugin: AgentPlugin, user_goal: str, config: dict[str, Any]) -> str:
    if plugin.plugin_id == "chess":
        return _chess_context(user_goal, config)
    if plugin.plugin_id == "poker_study":
        return _poker_context(user_goal)
    return ""


def enforce_plugin_safety(plugin: AgentPlugin, parsed: dict[str, Any]) -> dict[str, Any]:
    if not plugin.coach_only:
        return parsed

    parsed["mode"] = "coach"
    parsed["plugin"] = plugin.plugin_id
    parsed["action"] = {
        "type": "none",
        "x": None,
        "y": None,
        "text": None,
        "keys": [],
        "reason": f"{plugin.name} is coach-only, so LocalAgent will observe and advise without control.",
        "risk": "low",
    }
    return parsed


def _matches(plugin: AgentPlugin, lower_goal: str) -> bool:
    return any(alias in lower_goal for alias in plugin.aliases)


def _chess_context(user_goal: str, config: dict[str, Any]) -> str:
    fen = _extract_fen(user_goal)
    if not fen:
        return ""

    try:
        import chess
        from stockfish import Stockfish

        board = chess.Board(fen)
        stockfish_path = config["paths"]["stockfish"]
        engine = Stockfish(path=stockfish_path)
        engine.set_fen_position(fen)
        best_move = engine.get_best_move()
        legal_count = board.legal_moves.count()
        return (
            "Chess structured context:\n"
            f"- FEN: {fen}\n"
            f"- Side to move: {'white' if board.turn == chess.WHITE else 'black'}\n"
            f"- Legal moves: {legal_count}\n"
            f"- Stockfish quick best move: {best_move}\n"
        )
    except Exception as exc:
        return f"Chess structured context unavailable: {exc}"


def _extract_fen(text: str) -> str | None:
    match = re.search(r"fen\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _poker_context(user_goal: str) -> str:
    try:
        import eval7  # noqa: F401

        return "Poker tools available: eval7 is installed for study/equity extensions."
    except Exception:
        return ""

