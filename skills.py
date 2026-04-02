"""Skills system: markdown files with YAML frontmatter defining reusable prompt templates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator


@dataclass
class SkillDef:
    name: str
    description: str
    triggers: list       # ["/commit", "commit changes"]
    tools: list          # ["Bash", "Read"]
    prompt: str          # full prompt body after frontmatter
    file_path: str


SKILL_PATHS = [
    Path.cwd() / ".nano_claude" / "skills",    # project-level (priority)
    Path.home() / ".nano_claude" / "skills",    # user-level
]


def _parse_list_field(value: str) -> list:
    """Parse YAML-like list: ``[a, b, c]`` or ``"a, b, c"``.

    Args:
        value: raw string from frontmatter field
    Returns:
        list of stripped string items
    """
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item.strip().strip('"').strip("'") for item in value.split(",") if item.strip()]


def _parse_skill_file(path: Path) -> Optional[SkillDef]:
    """Parse a markdown file with ``---`` frontmatter into a SkillDef.

    The file format is:
        ---
        name: ...
        description: ...
        triggers: [/commit, commit changes]
        tools: [Bash, Read]
        ---
        Prompt body here...

    Args:
        path: path to the .md skill file
    Returns:
        SkillDef or None on parse failure / missing name
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    # Must start with --- and have a closing ---
    if not text.startswith("---"):
        return None

    parts = text.split("---", 2)
    # parts[0] is empty (before first ---), parts[1] is frontmatter, parts[2] is body
    if len(parts) < 3:
        return None

    frontmatter = parts[1].strip()
    prompt = parts[2].strip()

    # Simple YAML-like key: value parsing (no full YAML dependency)
    fields = {}
    for line in frontmatter.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        fields[key.strip()] = val.strip()

    name = fields.get("name", "")
    if not name:
        return None

    return SkillDef(
        name=name,
        description=fields.get("description", ""),
        triggers=_parse_list_field(fields.get("triggers", "")),
        tools=_parse_list_field(fields.get("tools", "")),
        prompt=prompt,
        file_path=str(path),
    )


def load_skills() -> list:
    """Scan SKILL_PATHS and return unique skills (project-level overrides user-level by name).

    Returns:
        list of SkillDef, deduplicated by name (project-level wins)
    """
    seen = {}
    # Iterate in reverse (user-level first), so project-level (index 0) overwrites
    for skill_dir in reversed(SKILL_PATHS):
        if not skill_dir.is_dir():
            continue
        for md_file in sorted(skill_dir.glob("*.md")):
            skill = _parse_skill_file(md_file)
            if skill:
                seen[skill.name] = skill

    return list(seen.values())


def find_skill(query: str) -> Optional[SkillDef]:
    """Find a skill matching the query.

    Extracts the first word from query and matches against trigger strings
    in all loaded skills.

    Args:
        query: user input string, e.g. "/commit fix typo"
    Returns:
        matching SkillDef or None
    """
    query = query.strip()
    if not query:
        return None

    first_word = query.split()[0]
    skills = load_skills()

    for skill in skills:
        for trigger in skill.triggers:
            # Exact match on first word against trigger
            if first_word == trigger:
                return skill
            # Also check if trigger starts with first_word (for multi-word triggers)
            if trigger.startswith(first_word + " "):
                return skill
    return None


def execute_skill(skill: SkillDef, args: str, state, config: dict,
                  system_prompt: str) -> Generator:
    """Execute a skill by injecting its prompt as a user message via agent.run().

    Args:
        skill: the SkillDef to execute
        args: additional user context / arguments
        state: AgentState
        config: config dict
        system_prompt: current system prompt
    Yields:
        events from agent.run()
    """
    from agent import run

    message = f"[Skill: {skill.name}]\n\n{skill.prompt}\n\nUser context: {args}"
    yield from run(message, state, config, system_prompt)
