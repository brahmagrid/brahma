"""
Brahma — The Bootstrap Agent Loop.

A minimal, self-extending agent runtime. Starts with only meta-capabilities
(spawn, generate, validate, save) and generates all other capabilities at runtime.

Architecture: Think → Act → Observe → Repeat
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from brahma.models import ModelResponse, call_model
from brahma.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """
    The Brahma agent loop. Minimal by design — bootstrap tools + task execution.

    Args:
        system_prompt: The bootstrap meta-capability prompt.
        model: Provider + model name (e.g., "anthropic:claude-sonnet-4-20250514").
        tools: Registry of available tool functions.
        max_turns: Safety limit — maximum LLM calls per task.
    """

    system_prompt: str
    model: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    max_turns: int = 50

    # Internal state
    _messages: list[dict] = field(default_factory=list, init=False, repr=False)
    _turn_count: int = field(default=0, init=False, repr=False)
    _token_usage: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the message history with the system prompt."""
        self._messages = [{"role": "system", "content": self.system_prompt}]

    # ── Public API ──────────────────────────────────────────────────

    def run(self, task: str) -> str:
        """Execute a task. Returns the agent's final text response."""
        self._messages.append({"role": "user", "content": task})
        self._turn_count = 0
        self._token_usage = {"input": 0, "output": 0, "total": 0}

        logger.info("=== Brahma run started ===")
        logger.info("Task: %s", _truncate(task, 200))

        while self._turn_count < self.max_turns:
            self._turn_count += 1

            logger.debug("─ Turn %d ─ calling %s", self._turn_count, self.model)

            response = call_model(
                model=self.model,
                messages=self._messages,
                tools=self.tools.schemas(),
            )

            self._accumulate_tokens(response)

            logger.debug(
                "─ Turn %d ─ stop_reason=%s text=%s tool_calls=%d "
                "tokens(in=%d out=%d total=%d)",
                self._turn_count,
                response.stop_reason,
                _truncate(response.text, 80),
                len(response.tool_calls),
                response.usage.input_tokens,
                response.usage.output_tokens,
                self._token_usage["total"],
            )

            if response.stop_reason == "end_turn":
                logger.info("=== Run complete — %d turns, %d total tokens ===",
                            self._turn_count, self._token_usage["total"])
                return response.text

            if response.stop_reason == "tool_use":
                self._messages.append(response.to_assistant_message())
                tool_results = self._execute_tools(response.tool_calls)
                self._messages.append(_tool_results_message(tool_results))

                for tc, tr in zip(response.tool_calls, tool_results, strict=True):
                    status = "✓" if tr.success else "✗"
                    logger.debug(
                        "─ Turn %d ─ tool %s %s(%s) → %s",
                        self._turn_count,
                        status,
                        tc.get("name", "?"),
                        _truncate(_summarize_input(tc.get("input", {})), 60),
                        _truncate(tr.output or tr.error, 120),
                    )
                continue

            # Should not reach here — unknown stop reason
            raise RuntimeError(f"Unknown stop_reason: {response.stop_reason}")

        logger.warning("=== Max turns (%d) reached ===", self.max_turns)
        return "⚠️ Max turns reached without completion."

    # ── Internal ────────────────────────────────────────────────────

    def _execute_tools(self, tool_calls: list[dict]) -> list[ToolResult]:
        """Execute each tool call and collect results."""
        results: list[ToolResult] = []
        for call in tool_calls:
            tool_name = call.get("name", "unknown")
            tool_input = call.get("input", {})
            call_id = call.get("id", "")

            try:
                result = self.tools.execute(tool_name, tool_input)
            except Exception as exc:
                result = ToolResult(
                    tool=tool_name,
                    success=False,
                    output="",
                    error=f"{type(exc).__name__}: {exc}",
                )

            result.call_id = call_id
            results.append(result)

        return results

    def _accumulate_tokens(self, response: ModelResponse) -> None:
        """Add token usage from a model response to the running totals."""
        self._token_usage["input"] += response.usage.input_tokens
        self._token_usage["output"] += response.usage.output_tokens
        self._token_usage["total"] += response.usage.input_tokens + response.usage.output_tokens

    # ── Properties ──────────────────────────────────────────────────

    @property
    def context_tokens(self) -> int:
        """Estimate context usage by counting characters / 4 (rough token estimate)."""
        total_chars = sum(len(_serialize_content(msg.get("content", ""))) for msg in self._messages)
        return total_chars // 4

    @property
    def turn_count(self) -> int:
        """Current turn number (increments each LLM call)."""
        return self._turn_count


# ── Helpers ────────────────────────────────────────────────────────────


def _tool_results_message(results: list[ToolResult]) -> dict:
    """Build a user message containing tool results."""
    content: list[dict] = []
    for r in results:
        block = {
            "type": "tool_result",
            "tool_use_id": r.call_id,
            "content": r.output if r.success else f"ERROR: {r.error}",
        }
        content.append(block)

    return {"role": "user", "content": content}


def _serialize_content(content: object) -> str:
    """Coerce message content to a string for token estimation."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            _extract_text(block) if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


def _extract_text(block: dict) -> str:
    """Extract the text field from a content block dict."""
    return block.get("text", "")


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending '…' if truncated."""
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _summarize_input(input_dict: dict) -> str:
    """Summarize a tool input dict for logging — key:value pairs."""
    if not input_dict:
        return ""
    parts = []
    for k, v in input_dict.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "..."
        parts.append(f"{k}={v}")
    return ", ".join(parts)
