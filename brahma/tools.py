"""
Brahma — Tool system.

Bootstrap tools (always loaded):
  read_file, write_file, search_files, terminal,
  delegate_task, skill_manage, web_fetch

All other tools are generated at runtime by the agent itself.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# ── Tool Result ─────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Result of a single tool execution, including success/failure metadata."""

    tool: str
    success: bool
    output: str
    error: str = ""
    call_id: str = ""


# ── Tool Registry ──────────────────────────────────────────────────────


@dataclass
class ToolRegistry:
    """
    Holds available tools: name → (callable, schema).

    Schema is the JSON Schema for the tool's input parameters.
    """

    _tools: dict[str, tuple[Callable, dict]] = field(default_factory=dict)

    def register(self, name: str, fn: Callable, schema: dict) -> None:
        """Register a tool function with its JSON Schema."""
        self._tools[name] = (fn, schema)

    def schemas(self) -> list[dict]:
        """Return tool schemas in Anthropic/OpenAI-compatible format."""
        return [
            {
                "name": name,
                "description": schema.get("description", ""),
                "input_schema": schema.get("input_schema", schema),
            }
            for name, (_, schema) in self._tools.items()
        ]

    def execute(self, name: str, params: dict) -> ToolResult:
        """Execute a tool by name with the given parameters."""
        if name not in self._tools:
            return ToolResult(tool=name, success=False, output="", error=f"Unknown tool: {name}")

        fn, _ = self._tools[name]
        try:
            output = fn(**params)
            return ToolResult(tool=name, success=True, output=str(output))
        except Exception as exc:
            return ToolResult(tool=name, success=False, output="", error=str(exc))

    def get(self, name: str) -> Callable | None:
        """Look up a tool function by name, or None if not registered."""
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def list_tools(self) -> list[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())


# ── Bootstrap Tool Implementations ─────────────────────────────────────

# --- read_file ---


def _read_file(path: str, offset: int = 1, limit: int = 500) -> str:
    """Read a text file with line numbers."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"ERROR: File not found: {path}"

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, offset - 1)
    end = min(len(lines), start + limit)

    result = []
    for i in range(start, end):
        result.append(f"{i + 1:6d}|{lines[i]}")

    output = "\n".join(result)
    if end < len(lines):
        output += f"\n... ({len(lines) - end} more lines)"
    if start > 0:
        output = f"... ({start} lines above)\n" + output

    return output


READ_FILE_SCHEMA = {
    "description": "Read a text file with line numbers and pagination.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start from (1-indexed).",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to return.",
                "default": 500,
            },
        },
        "required": ["path"],
    },
}

# --- write_file ---


def _write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    size = p.stat().st_size
    return f"Wrote {size} bytes to {path}"


WRITE_FILE_SCHEMA = {
    "description": (
        "Write content to a file. Creates parent directories. Overwrites existing files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {
                "type": "string",
                "description": "Complete content to write.",
            },
        },
        "required": ["path", "content"],
    },
}

# --- search_files ---


def _search_files(pattern: str, path: str = ".", file_glob: str | None = None) -> str:
    """Search file contents with ripgrep. Falls back to grep if rg unavailable."""
    p = Path(path).expanduser()
    cmd = ["rg", "--line-number", "--no-heading", "--color=never", pattern, str(p)]

    if file_glob:
        cmd.extend(["--glob", file_glob])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if not output:
            return f"No matches for '{pattern}' in {path}"
        lines = output.split("\n")
        if len(lines) > 50:
            output = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
        return output
    except FileNotFoundError:
        # ripgrep not installed — fall back to grep
        try:
            grep_cmd = ["grep", "-rn", "--color=never", pattern, str(p)]
            if file_glob:
                grep_cmd.insert(2, f"--include={file_glob}")
            result = subprocess.run(grep_cmd, capture_output=True, text=True, timeout=30)
            return result.stdout.strip() or f"No matches for '{pattern}' in {path}"
        except Exception:
            return "ERROR: Neither ripgrep nor grep available."
    except subprocess.TimeoutExpired:
        return "ERROR: Search timed out."


SEARCH_FILES_SCHEMA = {
    "description": ("Search file contents with ripgrep. Use for finding code, patterns, or text."),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in.",
                "default": ".",
            },
            "file_glob": {
                "type": "string",
                "description": "Optional file glob filter (e.g., '*.py').",
            },
        },
        "required": ["pattern"],
    },
}

# --- terminal ---


def _terminal(command: str, timeout: int = 120) -> str:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable="/bin/bash",
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += "\n[stderr]\n" + result.stderr.strip()
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout}s"


TERMINAL_SCHEMA = {
    "description": (
        "Execute a shell command. Use for builds, git, package management, and running scripts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait.",
                "default": 120,
            },
        },
        "required": ["command"],
    },
}

# --- delegate_task ---


def _delegate_task(goal: str, context: str = "", tools: list[str] | None = None) -> str:
    """
    Spawn a child agent to work on a task independently.

    This is the spawn meta-capability. In MVP, it creates a new Agent instance
    with a subset of tools and runs it in-process. Future: subprocess/K8s Job.
    """
    # Avoid circular import
    from brahma.agent import Agent
    from brahma.bootstrap import BOOTSTRAP_PROMPT

    # Build child's toolset
    child_tools = ToolRegistry()
    for _tool_name in tools or ["read_file", "write_file", "search_files", "terminal"]:
        # Inherit tool from parent if available
        pass  # Child gets its own minimal toolset — see spawn() below

    child = Agent(
        system_prompt=BOOTSTRAP_PROMPT,
        model="deepseek:deepseek-chat",  # Default cheap model for children
        tools=child_tools,
        max_turns=30,
    )

    task_prompt = f"GOAL: {goal}"
    if context:
        task_prompt += f"\n\nCONTEXT:\n{context}"

    return child.run(task_prompt)


DELEGATE_TASK_SCHEMA = {
    "description": ("Spawn a child Brahma agent to work on a subtask independently."),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What the child agent should accomplish.",
            },
            "context": {
                "type": "string",
                "description": "Background information for the child agent.",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tool names to give the child agent.",
            },
        },
        "required": ["goal"],
    },
}

# --- skill_manage ---

SKILLS_DIR = Path.home() / ".brahma" / "skills"


def _skill_manage(action: str, name: str, content: str = "") -> str:
    """
    Manage skills — save, load, list, or delete.

    This is the 'save' meta-capability. Skills persist on disk.
    """
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    if action == "save":
        skill_path = SKILLS_DIR / f"{name}.md"
        skill_path.write_text(content, encoding="utf-8")
        return f"Skill '{name}' saved ({len(content)} chars)."

    elif action == "load":
        skill_path = SKILLS_DIR / f"{name}.md"
        if not skill_path.exists():
            return f"ERROR: Skill '{name}' not found."
        return skill_path.read_text(encoding="utf-8")

    elif action == "list":
        skills = list(SKILLS_DIR.glob("*.md"))
        if not skills:
            return "No skills saved yet."
        return "Saved skills:\n" + "\n".join(f"  - {s.stem}" for s in sorted(skills))

    elif action == "delete":
        skill_path = SKILLS_DIR / f"{name}.md"
        if not skill_path.exists():
            return f"ERROR: Skill '{name}' not found."
        skill_path.unlink()
        return f"Skill '{name}' deleted."

    else:
        return f"ERROR: Unknown action '{action}'. Use: save, load, list, delete."


SKILL_MANAGE_SCHEMA = {
    "description": (
        "Manage persistent skills — save, load, list, or delete."
        " Skills are saved to ~/.brahma/skills/."
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

# --- web_fetch ---


def _web_fetch(url: str) -> str:
    """Fetch a URL and return text content."""
    import httpx

    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        text = response.text

        # Truncate if too large
        if len(text) > 100_000:
            text = text[:100_000] + f"\n... (truncated {len(response.text) - 100_000} chars)"

        return text
    except Exception as exc:
        return f"ERROR: Failed to fetch {url}: {exc}"


WEB_FETCH_SCHEMA = {
    "description": "Fetch content from a URL. Returns text content (HTML, JSON, etc.).",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch."},
        },
        "required": ["url"],
    },
}

# ── Bootstrap Toolset ──────────────────────────────────────────────────


def bootstrap_tools() -> ToolRegistry:
    """Create a ToolRegistry with the 7 bootstrap tools."""
    registry = ToolRegistry()
    registry.register("read_file", _read_file, READ_FILE_SCHEMA)
    registry.register("write_file", _write_file, WRITE_FILE_SCHEMA)
    registry.register("search_files", _search_files, SEARCH_FILES_SCHEMA)
    registry.register("terminal", _terminal, TERMINAL_SCHEMA)
    registry.register("delegate_task", _delegate_task, DELEGATE_TASK_SCHEMA)
    registry.register("skill_manage", _skill_manage, SKILL_MANAGE_SCHEMA)
    registry.register("web_fetch", _web_fetch, WEB_FETCH_SCHEMA)
    return registry
