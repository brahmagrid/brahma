"""Bootstrap tool: read_file — read a text file with line numbers."""

from __future__ import annotations

from pathlib import Path

from brahma.tools.registry import registry

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


def read_file(path: str, offset: int = 1, limit: int = 500) -> str:
    """Read a text file with line numbers."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"ERROR: File not found: {path}"

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, offset - 1)
    end = min(len(lines), start + limit)

    result_parts = []
    for i in range(start, end):
        result_parts.append(f"{i + 1:6d}|{lines[i]}")

    output = "\n".join(result_parts)
    if end < len(lines):
        output += f"\n... ({len(lines) - end} more lines)"
    if start > 0:
        output = f"... ({start} lines above)\n" + output

    return output


registry.register(
    name="read_file",
    schema=READ_FILE_SCHEMA,
    handler=read_file,
)
