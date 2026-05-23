"""
Brahma — Model routing.

Supports Anthropic and OpenAI-compatible APIs.
The model string format is "provider:model_name".

Examples:
  - "anthropic:claude-sonnet-4-20250514"
  - "openai:gpt-4o"
  - "deepseek:deepseek-chat"
  - "openrouter:anthropic/claude-sonnet-4"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

load_dotenv()


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

    def to_assistant_message(self) -> dict:
        """Build an assistant message from this response for the message history."""
        content = []
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


# ── Provider Config ────────────────────────────────────────────────────

PROVIDERS = {
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/messages",
        "api_key_env": "ANTHROPIC_API_KEY",
        "header_name": "x-api-key",
        "api_version": "2023-06-01",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "header_name": "Authorization",
        "header_prefix": "Bearer ",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "header_name": "Authorization",
        "header_prefix": "Bearer ",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
        "header_name": "Authorization",
        "header_prefix": "Bearer ",
    },
}


# ── Public API ─────────────────────────────────────────────────────────


def call_model(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
) -> ModelResponse:
    """
    Call an LLM and return a structured response.

    model: "provider:model_name" (e.g., "anthropic:claude-sonnet-4-20250514")
    messages: List of message dicts in Anthropic/OpenAI format.
    tools: Optional tool schemas.
    """
    provider, model_name = _parse_model(model)

    if provider == "anthropic":
        return _call_anthropic(model_name, messages, tools, max_tokens)
    else:
        return _call_openai_compatible(provider, model_name, messages, tools, max_tokens)


# ── Provider Implementations ───────────────────────────────────────────


def _call_anthropic(
    model_name: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
) -> ModelResponse:
    cfg = PROVIDERS["anthropic"]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"${cfg['api_key_env']} not set")

    headers = {
        cfg["header_name"]: api_key,
        "anthropic-version": cfg["api_version"],
        "content-type": "application/json",
    }

    # Convert tools to Anthropic format
    anthropic_tools = None
    if tools:
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", {}),
            }
            for t in tools
        ]

    body = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": _normalize_messages_for_anthropic(messages),
    }
    if anthropic_tools:
        body["tools"] = anthropic_tools

    response = httpx.post(
        cfg["base_url"],
        headers=headers,
        json=body,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    return _parse_anthropic_response(data)


def _call_openai_compatible(
    provider: str,
    model_name: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
) -> ModelResponse:
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"${cfg['api_key_env']} not set")

    headers = {
        "content-type": "application/json",
    }
    auth_value = cfg.get("header_prefix", "") + api_key
    headers[cfg["header_name"]] = auth_value

    # Convert tools to OpenAI format
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

    body = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": _normalize_messages_for_openai(messages),
    }
    if openai_tools:
        body["tools"] = openai_tools

    response = httpx.post(
        cfg["base_url"],
        headers=headers,
        json=body,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    return _parse_openai_response(data)


# ── Response Parsing ───────────────────────────────────────────────────


def _parse_anthropic_response(data: dict) -> ModelResponse:
    text = ""
    tool_calls = []
    stop_reason = data.get("stop_reason", "end_turn")

    for block in data.get("content", []):
        if block["type"] == "text":
            text += block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "name": block["name"],
                    "input": block["input"],
                }
            )

    if tool_calls and not text:
        stop_reason = "tool_use"
    elif text and not tool_calls:
        stop_reason = "end_turn"

    usage = Usage(
        input_tokens=data.get("usage", {}).get("input_tokens", 0),
        output_tokens=data.get("usage", {}).get("output_tokens", 0),
    )

    return ModelResponse(text=text, stop_reason=stop_reason, tool_calls=tool_calls, usage=usage)


def _parse_openai_response(data: dict) -> ModelResponse:
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    text = message.get("content", "") or ""
    tool_calls = []

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

    return ModelResponse(text=text, stop_reason=stop_reason, tool_calls=tool_calls, usage=usage)


# ── Message Normalization ──────────────────────────────────────────────


def _normalize_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Ensure messages are in Anthropic format."""
    normalized = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Anthropic doesn't support system messages in the messages array
        if role == "system":
            continue

        # String content → [{type: "text", text: ...}]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        normalized.append({"role": role, "content": content})
    return normalized


def _normalize_messages_for_openai(messages: list[dict]) -> list[dict]:
    """Ensure messages are in OpenAI format."""
    normalized = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        # OpenAI expects string or array of content blocks
        text_parts = []
        tool_calls_parts = []
        tool_results_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls_parts.append(block)
            elif block.get("type") == "tool_result":
                tool_results_parts.append(block)

        if tool_calls_parts and role == "assistant":
            normalized.append(
                {
                    "role": role,
                    "content": "\n".join(text_parts) if text_parts else None,
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
            )
        elif tool_results_parts and role == "user":
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_results_parts[0].get("tool_use_id", ""),
                    "content": tool_results_parts[0].get("content", ""),
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


def _parse_model(model: str) -> tuple[str, str]:
    """Parse 'provider:model_name' into (provider, model_name)."""
    if ":" in model:
        provider, model_name = model.split(":", 1)
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Known: {list(PROVIDERS.keys())}")
        return provider, model_name
    # Default to DeepSeek if no provider specified
    return "deepseek", model
