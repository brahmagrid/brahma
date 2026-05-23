"""
Tests for the Brahma agent loop.

Covers Agent initialization, properties, tool execution, helpers, and the
run loop (using a mock call_model).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from brahma.agent import Agent, _extract_text, _serialize_content, _tool_results_message
from brahma.models import ModelResponse, Usage
from brahma.tools import ToolRegistry, ToolResult

# ═══════════════════════════════════════════════════════════════════════════
# Helpers — extract_text
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractText:
    """Tests for the _extract_text helper."""

    def test_extracts_text_field(self) -> None:
        """Returns the 'text' value from a dict."""
        assert _extract_text({"text": "hello"}) == "hello"

    def test_missing_text_returns_empty_string(self) -> None:
        """Returns '' when 'text' key is missing."""
        assert _extract_text({"other": "value"}) == ""

    def test_empty_text_returns_empty_string(self) -> None:
        """Returns '' when 'text' is empty."""
        assert _extract_text({"text": ""}) == ""


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — serialize_content
# ═══════════════════════════════════════════════════════════════════════════


class TestSerializeContent:
    """Tests for the _serialize_content helper."""

    def test_string_passthrough(self) -> None:
        """Strings pass through unchanged."""
        assert _serialize_content("hello world") == "hello world"

    def test_list_of_blocks(self) -> None:
        """List of dict blocks extracts text fields."""
        content = [
            {"type": "text", "text": "first "},
            {"type": "text", "text": "second"},
        ]
        assert _serialize_content(content) == "first second"

    def test_list_of_strings(self) -> None:
        """List of strings joins them."""
        assert _serialize_content(["a", "b"]) == "ab"

    def test_mixed_list(self) -> None:
        """Mixed list of dicts and strings is serialized."""
        content = [{"type": "text", "text": "hello "}, "world"]
        assert _serialize_content(content) == "hello world"

    def test_other_types(self) -> None:
        """Non-string, non-list content is stringified."""
        assert _serialize_content(42) == "42"
        assert _serialize_content(None) == "None"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — tool_results_message
# ═══════════════════════════════════════════════════════════════════════════


class TestToolResultsMessage:
    """Tests for the _tool_results_message helper."""

    def test_single_success_result(self) -> None:
        """Builds correct message for one successful result."""
        result = ToolResult(
            tool="read_file", success=True, output="file contents", call_id="c1"
        )
        msg = _tool_results_message([result])
        assert msg["role"] == "user"
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["content"] == "file contents"

    def test_single_failure_result(self) -> None:
        """Builds correct message for a failed result."""
        result = ToolResult(
            tool="broken", success=False, output="", error="boom", call_id="c2"
        )
        msg = _tool_results_message([result])
        assert "ERROR" in msg["content"][0]["content"]

    def test_multiple_results(self) -> None:
        """Handles multiple results."""
        results = [
            ToolResult(tool="a", success=True, output="ok", call_id="1"),
            ToolResult(tool="b", success=True, output="good", call_id="2"),
        ]
        msg = _tool_results_message(results)
        assert len(msg["content"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Agent — initialization and defaults
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentInit:
    """Tests for Agent initialization and default values."""

    def test_creates_with_defaults(self) -> None:
        """Agent can be created with only required fields."""
        agent = Agent(
            system_prompt="You are helpful.",
            model="deepseek:deepseek-chat",
        )
        assert agent.system_prompt == "You are helpful."
        assert agent.model == "deepseek:deepseek-chat"
        assert agent.max_turns == 50
        assert isinstance(agent.tools, ToolRegistry)
        assert len(agent.tools) == 0

    def test_custom_max_turns(self) -> None:
        """max_turns can be customized."""
        agent = Agent(
            system_prompt="test",
            model="openai:gpt-4o",
            max_turns=10,
        )
        assert agent.max_turns == 10

    def test_custom_tools(self) -> None:
        """Custom tool registry can be supplied."""
        tools = ToolRegistry()
        agent = Agent(
            system_prompt="test",
            model="test:model",
            tools=tools,
        )
        assert agent.tools is tools

    def test_post_init_sets_system_message(self) -> None:
        """__post_init__ adds the system prompt as the first message."""
        agent = Agent(system_prompt="Be concise.", model="test:model")
        assert len(agent._messages) == 1
        assert agent._messages[0]["role"] == "system"
        assert agent._messages[0]["content"] == "Be concise."


# ═══════════════════════════════════════════════════════════════════════════
# Agent — properties
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentProperties:
    """Tests for Agent properties."""

    def test_turn_count_starts_at_zero(self) -> None:
        """Initial turn count is zero."""
        agent = Agent(system_prompt="test", model="test:model")
        assert agent.turn_count == 0

    def test_context_tokens_estimate(self) -> None:
        """context_tokens provides a rough character/4 estimate."""
        agent = Agent(system_prompt="short", model="test:model")
        # system prompt counts as characters / 4
        tokens = agent.context_tokens
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_context_tokens_grows_with_messages(self) -> None:
        """context_tokens increases when messages are added."""
        agent = Agent(system_prompt="x" * 100, model="test:model")
        initial = agent.context_tokens
        agent._messages.append({"role": "user", "content": "y" * 400})
        assert agent.context_tokens > initial


# ═══════════════════════════════════════════════════════════════════════════
# Agent — _execute_tools
# ═══════════════════════════════════════════════════════════════════════════


class TestExecuteTools:
    """Tests for Agent._execute_tools."""

    @pytest.fixture
    def agent(self) -> Agent:
        """Agent with a registry containing one tool."""
        tools = ToolRegistry()
        tools.register(
            "echo",
            lambda message: f"echo: {message}",
            {"description": "Echo a message"},
        )
        return Agent(system_prompt="test", model="test:model", tools=tools)

    def test_executes_registered_tool(self, agent: Agent) -> None:
        """Executes a tool and returns results."""
        results = agent._execute_tools(
            [{"name": "echo", "input": {"message": "hello"}, "id": "c1"}]
        )
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "echo: hello"
        assert results[0].call_id == "c1"

    def test_unknown_tool_returns_error(self, agent: Agent) -> None:
        """Unknown tool call returns a failure result."""
        results = agent._execute_tools(
            [{"name": "no_such_tool", "input": {}, "id": "c2"}]
        )
        assert results[0].success is False
        assert "Unknown tool" in results[0].error

    def test_multiple_tool_calls(self, agent: Agent) -> None:
        """Multiple tool calls are all executed."""
        results = agent._execute_tools(
            [
                {"name": "echo", "input": {"message": "a"}, "id": "1"},
                {"name": "echo", "input": {"message": "b"}, "id": "2"},
            ]
        )
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_missing_name_defaults_to_unknown(self, agent: Agent) -> None:
        """Tool call without 'name' field defaults to 'unknown'."""
        results = agent._execute_tools([{"input": {}, "id": "x"}])
        assert results[0].tool == "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Agent — _accumulate_tokens
# ═══════════════════════════════════════════════════════════════════════════


class TestAccumulateTokens:
    """Tests for Agent._accumulate_tokens."""

    def test_accumulates_input_output_and_total(self) -> None:
        """Token usage is summed across multiple responses."""
        agent = Agent(system_prompt="test", model="test:model")
        agent._token_usage = {"input": 0, "output": 0, "total": 0}

        resp = ModelResponse(
            text="hi",
            stop_reason="end_turn",
            usage=Usage(input_tokens=100, output_tokens=50),
        )
        agent._accumulate_tokens(resp)
        assert agent._token_usage["input"] == 100
        assert agent._token_usage["output"] == 50
        assert agent._token_usage["total"] == 150

        # Second response accumulates
        agent._accumulate_tokens(resp)
        assert agent._token_usage["input"] == 200
        assert agent._token_usage["output"] == 100
        assert agent._token_usage["total"] == 300


# ═══════════════════════════════════════════════════════════════════════════
# Agent — run loop (mocked call_model)
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentRun:
    """Tests for the Agent.run() loop using mocked call_model."""

    @pytest.fixture
    def agent(self) -> Agent:
        """Agent with a simple tool."""
        tools = ToolRegistry()
        tools.register(
            "double",
            lambda n: n * 2,
            {"description": "Double a number"},
        )
        return Agent(system_prompt="test", model="test:model", tools=tools)

    def test_single_turn_text_response(self, agent: Agent) -> None:
        """Single-turn text response returns immediately."""
        mock_resp = ModelResponse(
            text="Here is the answer.",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        with patch("brahma.agent.call_model", return_value=mock_resp):
            result = agent.run("What is 2+2?")
        assert result == "Here is the answer."
        assert agent.turn_count == 1

    def test_tool_use_then_text_response(self, agent: Agent) -> None:
        """Agent uses a tool, then returns a text response."""
        tool_resp = ModelResponse(
            text="",
            stop_reason="tool_use",
            tool_calls=[
                {"name": "double", "input": {"n": 21}, "id": "c1"}
            ],
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        text_resp = ModelResponse(
            text="The answer is 42.",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        with patch(
            "brahma.agent.call_model", side_effect=[tool_resp, text_resp]
        ):
            result = agent.run("Double 21")
        assert result == "The answer is 42."
        assert agent.turn_count == 2

    def test_max_turns_exceeded(self, agent: Agent) -> None:
        """Returns warning when max turns is reached."""
        agent.max_turns = 3
        tool_resp = ModelResponse(
            text="",
            stop_reason="tool_use",
            tool_calls=[
                {"name": "double", "input": {"n": 1}, "id": "c1"}
            ],
            usage=Usage(input_tokens=1, output_tokens=1),
        )
        with patch(
            "brahma.agent.call_model", return_value=tool_resp
        ):
            result = agent.run("Loop forever")
        assert "Max turns reached" in result

    def test_unknown_stop_reason_raises(self, agent: Agent) -> None:
        """Unknown stop_reason raises RuntimeError."""
        bad_resp = ModelResponse(
            text="",
            stop_reason="weird_unknown_reason",
            usage=Usage(),
        )
        with (
            patch("brahma.agent.call_model", return_value=bad_resp),
            pytest.raises(RuntimeError, match="Unknown stop_reason"),
        ):
            agent.run("Do something")

    def test_resets_counters_on_run(self, agent: Agent) -> None:
        """Each run() call resets turn count and token usage."""
        mock_resp = ModelResponse(
            text="done",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        with patch("brahma.agent.call_model", return_value=mock_resp):
            agent.run("task 1")
            agent.run("task 2")
        assert agent.turn_count == 1  # Reset, then one turn
        assert agent._token_usage["total"] == 15  # Only last call's tokens
