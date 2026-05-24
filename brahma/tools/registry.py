"""Brahma — tool registry (thread-safe, self-registering).

Mirrors Hermes Agent's ``tools/registry.py`` architecture:
  - global singleton ``registry`` imported by every tool file
  - ``register()`` called at module level by each tool
  - ``bootstrap_tools()`` triggers discovery by importing tool modules
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════
# ToolResult
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ToolResult:
    """Result of a single tool execution, including success/failure metadata."""

    tool: str
    success: bool
    output: str
    error: str = ""
    call_id: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# ToolEntry  (richer metadata — mirrors Hermes' ToolEntry)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ToolEntry:
    """Metadata for a single registered tool."""

    name: str
    schema: dict
    handler: Callable[..., object]
    check_fn: Callable[[], bool] | None = None
    toolset: str = "bootstrap"


# ═══════════════════════════════════════════════════════════════════════════
# ToolRegistry  (thread-safe, with availability checks)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ToolRegistry:
    """Thread-safe registry that collects tool schemas + handlers.

    Mirrors Hermes Agent's ``ToolRegistry``:
      - ``threading.RLock()`` on every mutation
      - ``check_fn`` support for availability gating
      - Generation counter for cache invalidation
    """

    _tools: dict[str, ToolEntry] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _generation: int = field(default=0, init=False)

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        name: str,
        handler: Callable[..., object],
        schema: dict,
        check_fn: Callable[[], bool] | None = None,
        toolset: str = "bootstrap",
    ) -> None:
        """Register a tool. Thread-safe; bumps the generation counter."""
        with self._lock:
            self._tools[name] = ToolEntry(
                name=name,
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                toolset=toolset,
            )
            self._generation += 1

    def deregister(self, name: str) -> None:
        """Remove a tool from the registry. Thread-safe."""
        with self._lock:
            self._tools.pop(name, None)
            self._generation += 1

    # ── Query ─────────────────────────────────────────────────────────

    def get(self, name: str) -> Callable[..., object] | None:
        """Look up a tool handler by name, or None."""
        with self._lock:
            entry = self._tools.get(name)
            return entry.handler if entry else None

    def get_entry(self, name: str) -> ToolEntry | None:
        """Return full ToolEntry metadata, or None."""
        with self._lock:
            return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered (thread-safe)."""
        with self._lock:
            return name in self._tools

    def __len__(self) -> int:
        """Return the number of registered tools (thread-safe)."""
        with self._lock:
            return len(self._tools)

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        with self._lock:
            return list(self._tools.keys())

    # ── Schema retrieval (with availability filtering) ────────────────

    def schemas(self) -> list[dict]:
        """Return tool schemas in Anthropic/OpenAI-compatible format.

        Only tools whose ``check_fn`` returns True (or have no check_fn)
        are included.
        """
        with self._lock:
            entries = list(self._tools.values())

        result: list[dict] = []
        for entry in sorted(entries, key=lambda e: e.name):
            if entry.check_fn is not None:
                try:
                    if not entry.check_fn():
                        continue
                except Exception:
                    continue
            result.append(
                {
                    "name": entry.name,
                    "description": entry.schema.get("description", ""),
                    "input_schema": entry.schema.get("input_schema", entry.schema),
                }
            )
        return result

    # ── Dispatch ──────────────────────────────────────────────────────

    def execute(self, name: str, params: dict) -> ToolResult:
        """Execute a tool by name. Thread-safe dispatch."""
        with self._lock:
            entry = self._tools.get(name)

        if not entry:
            return ToolResult(tool=name, success=False, output="", error=f"Unknown tool: {name}")

        try:
            output = entry.handler(**params)
            return ToolResult(tool=name, success=True, output=str(output))
        except Exception as exc:
            return ToolResult(tool=name, success=False, output="", error=str(exc))

    # ── Misc ──────────────────────────────────────────────────────────

    @property
    def generation(self) -> int:
        """Monotonically-increasing counter bumped on every mutation."""
        return self._generation

    def available_tools(self) -> list[str]:
        """Return names of tools whose check_fn passes (or has none)."""
        result: list[str] = []
        with self._lock:
            entries = list(self._tools.values())
        for entry in entries:
            if entry.check_fn is not None:
                try:
                    if not entry.check_fn():
                        continue
                except Exception:
                    continue
            result.append(entry.name)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════════════════

registry = ToolRegistry()
