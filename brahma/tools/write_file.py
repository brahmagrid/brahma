"""Bootstrap tool: write_file — write content to a file."""

from __future__ import annotations

from pathlib import Path

from brahma.tools.registry import registry

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


def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    size = p.stat().st_size
    return f"Wrote {size} bytes to {path}"


registry.register(
    name="write_file",
    schema=WRITE_FILE_SCHEMA,
    handler=write_file,
)
