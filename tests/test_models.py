"""
Tests for Brahma model routing and response types.

Covers Usage, ModelResponse, message normalization, and provider parsing.
"""

from __future__ import annotations

import pytest

from brahma.models import (
    PROVIDERS,
    ModelResponse,
    Usage,
    _normalize_messages_for_anthropic,
    _normalize_messages_for_openai,
    _parse_anthropic_response,
    _parse_model,
    _parse_openai_response,
)

# ═══════════════════════════════════════════════════════════════════════════
# Usage
# ═══════════════════════════════════════════════════════════════════════════


class TestUsage:
    """Tests for the Usage dataclass."""

    def test_defaults(self) -> None:
        """Default values are zero."""
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0

    def test_custom_values(self) -> None:
        """Custom token counts are stored."""
        u = Usage(input_tokens=100, output_tokens=50)
        assert u.input_tokens == 100
        assert u.output_tokens == 50


# ═══════════════════════════════════════════════════════════════════════════
# ModelResponse
# ═══════════════════════════════════════════════════════════════════════════


class TestModelResponse:
    """Tests for the ModelResponse dataclass."""

    def test_defaults(self) -> None:
        """Default values are sensible."""
        mr = ModelResponse()
        assert mr.text == ""
        assert mr.stop_reason == ""
        assert mr.tool_calls == []
        assert mr.usage == Usage()

    def test_to_assistant_message_text_only(self) -> None:
        """Text-only response produces a text content block."""
        mr = ModelResponse(
            text="Hello!",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        msg = mr.to_assistant_message()
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello!"

    def test_to_assistant_message_with_tool_calls(self) -> None:
        """Response with tool calls includes both text and tool_use blocks."""
        mr = ModelResponse(
            text="Let me check that.",
            stop_reason="tool_use",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "read_file",
                    "input": {"path": "/tmp/test.txt"},
                }
            ],
            usage=Usage(input_tokens=20, output_tokens=10),
        )
        msg = mr.to_assistant_message()
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][1]["id"] == "call_1"
        assert msg["content"][1]["name"] == "read_file"

    def test_to_assistant_message_tool_calls_only(self) -> None:
        """Tool-call-only response produces only tool_use blocks."""
        mr = ModelResponse(
            text="",
            stop_reason="tool_use",
            tool_calls=[{"id": "call_1", "name": "terminal", "input": {"command": "ls"}}],
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        msg = mr.to_assistant_message()
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "tool_use"

    def test_to_assistant_message_empty(self) -> None:
        """Empty response produces empty content array."""
        mr = ModelResponse()
        msg = mr.to_assistant_message()
        assert msg["role"] == "assistant"
        assert msg["content"] == []


# ═══════════════════════════════════════════════════════════════════════════
# _parse_model
# ═══════════════════════════════════════════════════════════════════════════


class TestParseModel:
    """Tests for the _parse_model helper."""

    def test_with_provider(self) -> None:
        """Parses provider:model_name correctly."""
        provider, model = _parse_model("anthropic:claude-sonnet-4-20250514")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_without_provider_defaults_to_deepseek(self) -> None:
        """Without a provider prefix, defaults to deepseek."""
        provider, model = _parse_model("deepseek-chat")
        assert provider == "deepseek"
        assert model == "deepseek-chat"

    def test_unknown_provider_raises(self) -> None:
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            _parse_model("nonexistent:some-model")

    def test_deepseek_provider(self) -> None:
        """DeepSeek provider is recognized."""
        provider, model = _parse_model("deepseek:deepseek-chat")
        assert provider == "deepseek"
        assert model == "deepseek-chat"

    def test_openrouter_provider(self) -> None:
        """OpenRouter provider is recognized."""
        provider, model = _parse_model("openrouter:anthropic/claude-sonnet-4")
        assert provider == "openrouter"
        assert model == "anthropic/claude-sonnet-4"

    def test_all_known_providers(self) -> None:
        """All configured providers parse correctly."""
        for provider_name in PROVIDERS:
            result = _parse_model(f"{provider_name}:any-model")
            assert result == (provider_name, "any-model")


# ═══════════════════════════════════════════════════════════════════════════
# Message normalization — Anthropic
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeMessagesForAnthropic:
    """Tests for _normalize_messages_for_anthropic."""

    def test_strips_system_messages(self) -> None:
        """System messages are removed (Anthropic uses top-level system)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_messages_for_anthropic(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_converts_string_content_to_block(self) -> None:
        """String content is wrapped in a text block."""
        messages = [{"role": "user", "content": "plain text"}]
        result = _normalize_messages_for_anthropic(messages)
        assert result[0]["content"] == [{"type": "text", "text": "plain text"}]

    def test_preserves_list_content(self) -> None:
        """Already-list content is preserved."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ]
        result = _normalize_messages_for_anthropic(messages)
        assert result[0]["content"] == [{"type": "text", "text": "hello"}]


# ═══════════════════════════════════════════════════════════════════════════
# Message normalization — OpenAI
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeMessagesForOpenai:
    """Tests for _normalize_messages_for_openai."""

    def test_passes_through_string_content(self) -> None:
        """String content messages pass through unchanged."""
        messages = [{"role": "user", "content": "plain text"}]
        result = _normalize_messages_for_openai(messages)
        assert result[0] == {"role": "user", "content": "plain text"}

    def test_text_block_extraction(self) -> None:
        """Content blocks with text parts are extracted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        result = _normalize_messages_for_openai(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "part one\npart two"

    def test_tool_use_normalization(self) -> None:
        """Tool-use blocks are normalized to OpenAI tool_calls format."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "read_file",
                        "input": {"path": "/tmp/test.txt"},
                    }
                ],
            }
        ]
        result = _normalize_messages_for_openai(messages)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "read_file"

    def test_tool_result_normalization(self) -> None:
        """Tool-result blocks are normalized to OpenAI tool role."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "file contents here",
                    }
                ],
            }
        ]
        result = _normalize_messages_for_openai(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "file contents here"


# ═══════════════════════════════════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestParseAnthropicResponse:
    """Tests for _parse_anthropic_response."""

    def test_text_only_response(self) -> None:
        """Pure text response is parsed correctly."""
        data = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello there"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = _parse_anthropic_response(data)
        assert result.text == "Hello there"
        assert result.stop_reason == "end_turn"
        assert result.tool_calls == []
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_tool_use_response(self) -> None:
        """Tool-use response is parsed with tool calls."""
        data = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_abc",
                    "name": "read_file",
                    "input": {"path": "/tmp/x"},
                }
            ],
            "usage": {"input_tokens": 15, "output_tokens": 8},
        }
        result = _parse_anthropic_response(data)
        assert result.text == ""
        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_abc"

    def test_missing_usage_defaults(self) -> None:
        """Missing usage data defaults to zero."""
        data = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
        }
        result = _parse_anthropic_response(data)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


class TestParseOpenAIResponse:
    """Tests for _parse_openai_response."""

    def test_text_only_response(self) -> None:
        """Pure text response is parsed correctly."""
        data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = _parse_openai_response(data)
        assert result.text == "Hello"
        assert result.stop_reason == "end_turn"
        assert result.tool_calls == []

    def test_tool_calls_response(self) -> None:
        """Response with tool calls is parsed correctly."""
        data = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/tmp/x"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        result = _parse_openai_response(data)
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[0]["input"] == {"path": "/tmp/x"}
        assert result.stop_reason == "tool_use"

    def test_empty_content_defaults_to_empty_string(self) -> None:
        """None/null content defaults to empty string."""
        data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": None},
                }
            ],
        }
        result = _parse_openai_response(data)
        assert result.text == ""
