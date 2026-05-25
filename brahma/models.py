"""
Brahma — Model routing (DeepSeek-only for PoC).

Supports the DeepSeek API via its OpenAI-compatible endpoint.

Model strings: just the model name (e.g., "deepseek-chat").
The "deepseek:" prefix is accepted but optional.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# DeepSeek V4-Pro supports up to 65,536 output tokens.
# We set 32,768 as a safe default — high enough for code generation
# in the bootstrap GENERATE step, leaving headroom for tool calls.
DEEPSEEK_MAX_OUTPUT_TOKENS = 32768

# Model context window sizes (input tokens).
# Used by Agent.context_tokens for budget monitoring.
# Source: DeepSeek API docs — https://api-docs.deepseek.com/quick_start/pricing
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-v4-pro": 1_048_576,
    "deepseek-v4-flash": 1_048_576,
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
}

# ── Response Types ─────────────────────────────────────────────────────


@dataclass
class Usage:
    """Token usage counts for a single model response."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelResponse:
    """Structured response from an LLM call — text, tool calls, and usage."""

    text: str = ""
    stop_reason: str = ""  # "end_turn" | "tool_use"
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    reasoning_content: str = ""

    def to_assistant_message(self) -> dict:
        """Build an assistant message from this response for the message history."""
        content: list[dict] = []
        if self.reasoning_content:
            content.append({"type": "reasoning", "text": self.reasoning_content})
        if self.text:
            content.append({"type": "text", "text": self.text})
        for tc in self.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                }
            )
        return {"role": "assistant", "content": content}


# ── DeepSeek API Config ────────────────────────────────────────────────

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"


# ── Public API ─────────────────────────────────────────────────────────


def call_model(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = DEEPSEEK_MAX_OUTPUT_TOKENS,
) -> ModelResponse:
    """
    Call the DeepSeek API and return a structured response.

    Args:
        model: Model name (e.g., "deepseek-chat"). The "deepseek:" prefix
               is stripped if present for backward compatibility.
        messages: List of message dicts in Anthropic/OpenAI format.
        tools: Optional tool schemas.
        max_tokens: Max output tokens (default 8192 for DeepSeek V3).

    Returns:
        A ModelResponse with text, tool_calls, stop_reason, and usage.
    """
    model_name = _strip_provider_prefix(model)

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Convert tools to OpenAI function format
    openai_tools = None
    if tools:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    body: dict = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": _normalize_messages(messages),
    }
    if openai_tools:
        body["tools"] = openai_tools

    response = httpx.post(
        DEEPSEEK_BASE_URL,
        headers=headers,
        json=body,
        timeout=120,
    )
    if response.status_code >= 400:
        logger.error(
            "DeepSeek %d: %s",
            response.status_code,
            response.text[:500],
        )
    response.raise_for_status()
    data = response.json()

    return _parse_response(data)


# ── Response Parsing ───────────────────────────────────────────────────


def _parse_response(data: dict) -> ModelResponse:
    """Parse a DeepSeek (OpenAI-compatible) chat completion response."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    text = message.get("content", "") or ""
    reasoning_content = message.get("reasoning_content", "") or ""
    tool_calls: list[dict] = []

    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            tool_calls.append(
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"]),
                }
            )

    stop_reason = "tool_use" if tool_calls else "end_turn"
    if finish_reason == "stop" and not tool_calls:
        stop_reason = "end_turn"

    usage = Usage(
        input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
        output_tokens=data.get("usage", {}).get("completion_tokens", 0),
    )

    return ModelResponse(
        text=text,
        stop_reason=stop_reason,
        tool_calls=tool_calls,
        usage=usage,
        reasoning_content=reasoning_content,
    )


# ── Message Normalization ──────────────────────────────────────────────


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Convert internal message format to DeepSeek (OpenAI-compatible) format."""
    normalized: list[dict] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        # Complex content with blocks (text, tool_use, tool_result)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_parts: list[dict] = []
        tool_results_parts: list[dict] = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "reasoning":
                reasoning_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls_parts.append(block)
            elif block.get("type") == "tool_result":
                tool_results_parts.append(block)

        if tool_calls_parts and role == "assistant":
            assistant_msg: dict[str, object] = {
                "role": role,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"]),
                        },
                    }
                    for tc in tool_calls_parts
                ],
            }
            if reasoning_parts:
                assistant_msg["reasoning_content"] = "\n".join(reasoning_parts)
            if text_parts:
                assistant_msg["content"] = "\n".join(text_parts)
            normalized.append(assistant_msg)
        elif tool_results_parts and role == "user":
            for tr in tool_results_parts:
                normalized.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr.get("content", ""),
                    }
                )
        else:
            normalized.append(
                {
                    "role": role,
                    "content": "\n".join(text_parts) if text_parts else "",
                }
            )

    return normalized


# ── Helpers ────────────────────────────────────────────────────────────


def _strip_provider_prefix(model: str) -> str:
    """Strip 'deepseek:' prefix if present (backward compat)."""
    if ":" in model:
        return model.split(":", 1)[1]
    return model


def get_model_context_window(model: str) -> int:
    """Return the context window size for a model, falling back to env or 64K.

    Resolution order:
        1. Exact match in MODEL_CONTEXT_WINDOWS
        2. BRAHMA_CONTEXT_WINDOW env var (if set)
        3. Conservative default: 64,000 tokens
    """
    model_name = _strip_provider_prefix(model)
    if model_name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_name]

    env_val = os.getenv("BRAHMA_CONTEXT_WINDOW", "")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            logger.warning(
                "BRAHMA_CONTEXT_WINDOW=%r is not an integer — using default 64000",
                env_val,
            )

    return 64_000
