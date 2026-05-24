"""
Tests for Brahma model routing (DeepSeek-only).

Covers Usage, ModelResponse, message normalization, response parsing,
and provider prefix stripping.
"""

from __future__ import annotations

from brahma.models import (
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    ModelResponse,
    Usage,
    _normalize_messages,
    _parse_response,
    _strip_provider_prefix,
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
# _strip_provider_prefix
# ═══════════════════════════════════════════════════════════════════════════


class TestStripProviderPrefix:
    """Tests for _strip_provider_prefix."""

    def test_with_prefix(self) -> None:
        """Strips the 'deepseek:' prefix."""
        assert _strip_provider_prefix("deepseek:deepseek-chat") == "deepseek-chat"

    def test_without_prefix(self) -> None:
        """Passes through when no prefix present."""
        assert _strip_provider_prefix("deepseek-chat") == "deepseek-chat"

    def test_other_prefix_stripped(self) -> None:
        """Any prefix before ':' is stripped (backward compat)."""
        assert _strip_provider_prefix("anthropic:claude-4") == "claude-4"

    def test_multiple_colons(self) -> None:
        """Only the first colon is treated as a delimiter."""
        assert _strip_provider_prefix("deepseek:deepseek-chat:v2") == "deepseek-chat:v2"


# ═══════════════════════════════════════════════════════════════════════════
# _normalize_messages
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeMessages:
    """Tests for _normalize_messages (DeepSeek / OpenAI-compatible format)."""

    def test_passes_through_string_content(self) -> None:
        """String content messages pass through unchanged."""
        messages = [{"role": "user", "content": "plain text"}]
        result = _normalize_messages(messages)
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
        result = _normalize_messages(messages)
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
        result = _normalize_messages(messages)
        assert result[0]["role"] == "assistant"
        assert "content" not in result[0]  # omitted when no text
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "read_file"

    def test_tool_use_with_text_keeps_content(self) -> None:
        """When a tool-use message also has text, content is included."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me look that up."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "search_files",
                        "input": {"pattern": "TODO"},
                    },
                ],
            }
        ]
        result = _normalize_messages(messages)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me look that up."
        assert len(result[0]["tool_calls"]) == 1

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
        result = _normalize_messages(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "file contents here"

    def test_multiple_tool_results(self) -> None:
        """Multiple tool results each become their own 'tool' role message."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "result one",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_2",
                        "content": "result two",
                    },
                ],
            }
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "result one"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_2"
        assert result[1]["content"] == "result two"

    def test_system_message_passes_through(self) -> None:
        """System role messages pass through (DeepSeek supports them)."""
        messages = [{"role": "system", "content": "You are helpful."}]
        result = _normalize_messages(messages)
        assert result[0] == {"role": "system", "content": "You are helpful."}

    def test_mixed_blocks_with_strings(self) -> None:
        """Mixed list of dicts and strings is handled (joined with newlines)."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello "}, "world"]}]
        result = _normalize_messages(messages)
        assert result[0]["content"] == "hello \nworld"


# ═══════════════════════════════════════════════════════════════════════════
# _parse_response
# ═══════════════════════════════════════════════════════════════════════════


class TestParseResponse:
    """Tests for _parse_response (DeepSeek / OpenAI-compatible format)."""

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
        result = _parse_response(data)
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
        result = _parse_response(data)
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
        result = _parse_response(data)
        assert result.text == ""

    def test_missing_usage_defaults_zero(self) -> None:
        """Missing usage data defaults to zero."""
        data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }
        result = _parse_response(data)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_multiple_tool_calls(self) -> None:
        """Multiple tool calls in one response are all parsed."""
        data = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "/a"}'},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": '{"path": "/b", "content": "x"}',
                                },
                            },
                        ],
                    },
                }
            ],
        }
        result = _parse_response(data)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[1]["name"] == "write_file"

    def test_text_with_tool_calls(self) -> None:
        """Response with both text and tool calls."""
        data = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "Let me check that first.",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "search_files",
                                    "arguments": '{"pattern": "TODO"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
        result = _parse_response(data)
        assert result.text == "Let me check that first."
        assert len(result.tool_calls) == 1
        assert result.stop_reason == "tool_use"


# ═══════════════════════════════════════════════════════════════════════════
# DEEPSEEK_MAX_OUTPUT_TOKENS
# ═══════════════════════════════════════════════════════════════════════════


class TestMaxOutputTokens:
    """Tests for the DeepSeek output token constant."""

    def test_max_tokens_is_32768(self) -> None:
        """Default max_tokens is set to 32,768 for V4-Pro headroom."""
        assert DEEPSEEK_MAX_OUTPUT_TOKENS == 32768

    def test_exceeds_default_4096(self) -> None:
        """Our setting far exceeds DeepSeek's default of 4096."""
        assert DEEPSEEK_MAX_OUTPUT_TOKENS > 4096
