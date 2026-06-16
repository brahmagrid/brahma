"""Bootstrap tool: skill_manage — save/load/list/delete persistent skills.

Skills persist on disk under ~/.brahma/skills/ and can be:
- **downloaded** — found via web_search, installed via pip/download
- **generated** — written from scratch when nothing suitable existed
- **adapted**  — existing tool found, installed, and wrapped with custom code
"""

from __future__ import annotations

from datetime import UTC, datetime

from brahma.constants import get_skills_dir
from brahma.tools.registry import registry

SKILL_MANAGE_SCHEMA = {
    "description": (
        "Manage persistent skills — save, load, list, delete, or install. "
        "Skills can be downloaded (from the web), generated (from scratch), "
        "or adapted (existing tool + wrapper). Saved skills include source "
        "metadata for future reference."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "load", "list", "delete", "install"],
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
            "source": {
                "type": "string",
                "enum": ["generated", "downloaded", "adapted"],
                "description": "How was this skill acquired? (for 'save').",
            },
            "url": {
                "type": "string",
                "description": "Source URL (e.g., PyPI, GitHub) — for 'save' and 'install'.",
            },
            "package": {
                "type": "string",
                "description": "Pip package name to install (required for 'install').",
            },
        },
        "required": ["action", "name"],
    },
}


def skill_manage(
    action: str,
    name: str,
    content: str = "",
    source: str = "generated",
    url: str = "",
    package: str = "",
) -> str:
    """Manage skills — save, load, list, delete, or install.

    This is the SAVE meta-capability. Skills persist on disk
    and can be acquired via download or generation.

    Args:
        action: One of save, load, list, delete, install.
        name: Skill name (without extension).
        content: Skill markdown content (for save).
        source: How the skill was acquired (for save).
        url: Source URL (for save and install).
        package: Pip package name (for install).

    Returns:
        Status message describing the result.
    """
    skills_dir = get_skills_dir()
    skills_dir.mkdir(parents=True, exist_ok=True)

    if action == "save":
        return _save_skill(skills_dir, name, content, source, url)

    elif action == "load":
        return _load_skill(skills_dir, name)

    elif action == "list":
        return _list_skills(skills_dir)

    elif action == "delete":
        return _delete_skill(skills_dir, name)

    elif action == "install":
        return _install_skill(skills_dir, name, package, url)

    else:
        return f"ERROR: Unknown action '{action}'. Use: save, load, list, delete, install."


def _save_skill(
    skills_dir: object,
    name: str,
    content: str,
    source: str,
    url: str,
) -> str:
    """Save a skill with metadata header.

    Args:
        skills_dir: Path to skills directory.
        name: Skill name.
        content: Skill body (markdown).
        source: How acquired.
        url: Source URL.

    Returns:
        Status message.
    """
    from pathlib import Path

    skills_dir_path = skills_dir if isinstance(skills_dir, Path) else Path(str(skills_dir))

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    metadata = f"<!-- skill_meta: source={source} saved={now}"
    if url:
        metadata += f" url={url}"
    metadata += " -->\n\n"

    skill_path = skills_dir_path / f"{name}.md"
    skill_path.write_text(metadata + content, encoding="utf-8")
    return f"Skill '{name}' saved ({source}, {len(content)} chars)."


def _load_skill(skills_dir: object, name: str) -> str:
    """Load a skill's content from disk.

    Args:
        skills_dir: Path to skills directory.
        name: Skill name.

    Returns:
        Skill content or error message.
    """
    from pathlib import Path

    skills_dir_path = skills_dir if isinstance(skills_dir, Path) else Path(str(skills_dir))
    skill_path = skills_dir_path / f"{name}.md"

    if not skill_path.exists():
        return f"ERROR: Skill '{name}' not found."
    return skill_path.read_text(encoding="utf-8")


def _list_skills(skills_dir: object) -> str:
    """List all saved skills with metadata.

    Args:
        skills_dir: Path to skills directory.

    Returns:
        Formatted list of skills.
    """
    from pathlib import Path

    skills_dir_path = skills_dir if isinstance(skills_dir, Path) else Path(str(skills_dir))
    skills = sorted(skills_dir_path.glob("*.md"))

    if not skills:
        return "No skills saved yet."

    import re

    lines = ["Saved skills:"]
    for s in skills:
        text = s.read_text(encoding="utf-8")
        # Extract source from metadata comment if present
        match = re.search(r"<!-- skill_meta: (.+?) -->", text)
        meta = match.group(1) if match else "source=unknown"
        lines.append(f"  - {s.stem}  ({meta})")

    return "\n".join(lines)


def _delete_skill(skills_dir: object, name: str) -> str:
    """Delete a skill from disk.

    Args:
        skills_dir: Path to skills directory.
        name: Skill name.

    Returns:
        Status message.
    """
    from pathlib import Path

    skills_dir_path = skills_dir if isinstance(skills_dir, Path) else Path(str(skills_dir))
    skill_path = skills_dir_path / f"{name}.md"

    if not skill_path.exists():
        return f"ERROR: Skill '{name}' not found."
    skill_path.unlink()
    return f"Skill '{name}' deleted."


def _install_skill(
    skills_dir: object,
    name: str,
    package: str,
    url: str,
) -> str:
    """Install a pip package and create a skill documenting it.

    Args:
        skills_dir: Path to skills directory.
        name: Skill name for the documentation file.
        package: Pip package name to install.
        url: Optional project URL (PyPI, GitHub).

    Returns:
        Status message with install output and skill path.
    """
    import subprocess
    from pathlib import Path

    skills_dir_path = skills_dir if isinstance(skills_dir, Path) else Path(str(skills_dir))

    if not package:
        return "ERROR: 'package' parameter is required for install action."

    # Install the package
    try:
        result = subprocess.run(
            ["pip", "install", package],
            capture_output=True,
            text=True,
            timeout=120,
        )
        install_output = result.stdout.strip() or result.stderr.strip()
        success = result.returncode == 0
    except FileNotFoundError:
        return "ERROR: pip not found. Is Python installed?"
    except subprocess.TimeoutExpired:
        return f"ERROR: pip install {package} timed out."

    # Create skill documentation
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    source_url = url or f"https://pypi.org/project/{package}/"

    skill_content = f"""# {name}

**Source:** downloaded
**Package:** `{package}`
**Installed:** {now}
**URL:** {source_url}

## Description

This skill provides the `{package}` Python package.

## Installation

```bash
pip install {package}
```

## Usage

Import and use `{package}` in Python:

```python
import {package.replace("-", "_")}
```

## Install Output

```
{install_output[:2000]}
```

## Notes

- Installed via pip. Verify with: `pip show {package}`
- Update with: `pip install --upgrade {package}`
"""

    skill_path = skills_dir_path / f"{name}.md"
    _save_skill(skills_dir_path, name, skill_content, "downloaded", source_url)

    status = "✓" if success else "⚠ (non-zero exit)"
    return (
        f"Package '{package}' installed. {status}\n"
        f"Output: {install_output[:500]}\n"
        f"Skill saved: {skill_path}"
    )


registry.register(
    name="skill_manage",
    schema=SKILL_MANAGE_SCHEMA,
    handler=skill_manage,
)
