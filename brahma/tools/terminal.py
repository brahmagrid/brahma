"""Bootstrap tool: terminal — execute a shell command."""

from __future__ import annotations

import subprocess

from brahma.tools.registry import registry

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


def terminal(command: str, timeout: int = 120) -> str:
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


registry.register(
    name="terminal",
    schema=TERMINAL_SCHEMA,
    handler=terminal,
)
