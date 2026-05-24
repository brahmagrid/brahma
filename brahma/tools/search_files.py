"""Bootstrap tool: search_files — search file contents with ripgrep."""

from __future__ import annotations

import subprocess
from pathlib import Path

from brahma.tools.registry import registry

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


def search_files(pattern: str, path: str = ".", file_glob: str | None = None) -> str:
    """Search file contents with ripgrep. Falls back to grep if rg unavailable."""
    p = Path(path).expanduser()
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--color=never",
        pattern,
        str(p),
    ]

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


registry.register(
    name="search_files",
    schema=SEARCH_FILES_SCHEMA,
    handler=search_files,
)
