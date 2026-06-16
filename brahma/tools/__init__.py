"""Brahma — tool system (self-registering architecture).

Each tool file imports ``registry`` from ``.registry`` and calls
``registry.register(...)`` at module level.  Importing a tool module is
all it takes to make its tools available.

``bootstrap_tools()`` triggers discovery by importing every built-in
tool module.  No manual registration needed — add a file to this
directory and it is automatically included.

Mirrors Hermes Agent's ``tools/`` directory architecture.
"""

from __future__ import annotations

# Import every tool module — each self-registers on load.
from brahma.tools import (
    delegate_task,  # noqa: F401
    hitl_request,  # noqa: F401
    read_file,  # noqa: F401
    search_files,  # noqa: F401
    skill_manage,  # noqa: F401
    terminal,  # noqa: F401
    web_fetch,  # noqa: F401
    web_search,  # noqa: F401
    write_file,  # noqa: F401
)
from brahma.tools.delegate_task import delegate_task as _delegate_task
from brahma.tools.hitl_request import set_hitl_queue
from brahma.tools.read_file import READ_FILE_SCHEMA
from brahma.tools.read_file import read_file as _read_file
from brahma.tools.registry import ToolRegistry, ToolResult, registry
from brahma.tools.search_files import search_files as _search_files
from brahma.tools.skill_manage import skill_manage as _skill_manage
from brahma.tools.terminal import terminal as _terminal
from brahma.tools.web_fetch import web_fetch as _web_fetch
from brahma.tools.web_search import web_search as _web_search
from brahma.tools.write_file import WRITE_FILE_SCHEMA
from brahma.tools.write_file import write_file as _write_file


def bootstrap_tools() -> ToolRegistry:
    """Return the global ToolRegistry with all bootstrap tools."""
    return registry


__all__ = [
    "READ_FILE_SCHEMA",
    "WRITE_FILE_SCHEMA",
    "ToolRegistry",
    "ToolResult",
    "_delegate_task",
    "_read_file",
    "_search_files",
    "_skill_manage",
    "_terminal",
    "_web_fetch",
    "_web_search",
    "_write_file",
    "bootstrap_tools",
    "registry",
    "set_hitl_queue",
]
