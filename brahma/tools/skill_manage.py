"""Bootstrap tool: skill_manage — save/load/list/delete persistent skills."""

from __future__ import annotations

from brahma.constants import get_skills_dir
from brahma.tools.registry import registry

SKILL_MANAGE_SCHEMA = {
    "description": (
        "Manage persistent skills — save, load, list, or delete."
        " Skills are saved under the Brahma home skills directory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "load", "list", "delete"],
                "description": "Action to perform.",
            },
            "name": {
                "type": "string",
                "description": "Skill name (without extension).",
            },
            "content": {
                "type": "string",
                "description": "Skill content (required for 'save').",
            },
        },
        "required": ["action", "name"],
    },
}


def skill_manage(action: str, name: str, content: str = "") -> str:
    """Manage skills — save, load, list, or delete.

    This is the 'save' meta-capability. Skills persist on disk.
    """
    skills_dir = get_skills_dir()
    skills_dir.mkdir(parents=True, exist_ok=True)

    if action == "save":
        skill_path = skills_dir / f"{name}.md"
        skill_path.write_text(content, encoding="utf-8")
        return f"Skill '{name}' saved ({len(content)} chars)."

    elif action == "load":
        skill_path = skills_dir / f"{name}.md"
        if not skill_path.exists():
            return f"ERROR: Skill '{name}' not found."
        return skill_path.read_text(encoding="utf-8")

    elif action == "list":
        skills = list(skills_dir.glob("*.md"))
        if not skills:
            return "No skills saved yet."
        return "Saved skills:\n" + "\n".join(f"  - {s.stem}" for s in sorted(skills))

    elif action == "delete":
        skill_path = skills_dir / f"{name}.md"
        if not skill_path.exists():
            return f"ERROR: Skill '{name}' not found."
        skill_path.unlink()
        return f"Skill '{name}' deleted."

    else:
        return f"ERROR: Unknown action '{action}'. Use: save, load, list, delete."


registry.register(
    name="skill_manage",
    schema=SKILL_MANAGE_SCHEMA,
    handler=skill_manage,
)
