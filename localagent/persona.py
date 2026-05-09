from __future__ import annotations


def persona_prompt(config: dict) -> str:
    persona = config.get("persona", {})
    name = persona.get("name", "LocalAgent")
    style = persona.get("style", "")
    traits = persona.get("traits", [])
    speech_rules = persona.get("speech_rules", [])

    trait_lines = "\n".join(f"- {trait}" for trait in traits)
    rule_lines = "\n".join(f"- {rule}" for rule in speech_rules)

    return f"""
Assistant persona name: {name}
Persona style:
{style}

Core traits:
{trait_lines}

Speech rules:
{rule_lines}
""".strip()


def fallback_spoken_prefix(config: dict | None = None) -> str:
    if not config:
        return "Here's what I see:"

    persona = config.get("persona", {})
    prefix = persona.get("fallback_prefix", "").strip()
    return prefix or "Here's what I see:"

