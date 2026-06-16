"""
Tests for the _delegate_task bootstrap tool.

Covers tool creation, parameter passing, tool filtering, and model inheritance.
Mocks Agent + call_model to avoid real LLM calls.
"""

from __future__ import annotations

from unittest.mock import patch

from brahma.tools import ToolRegistry, _delegate_task

# ═══════════════════════════════════════════════════════════════════════════
# _delegate_task — mocked Agent
# ═══════════════════════════════════════════════════════════════════════════


class TestDelegateTask:
    """Tests for _delegate_task with mocked Agent and LLM."""

    def test_creates_child_with_tools(self) -> None:
        """Child agent is created with the default (all) bootstrap tools."""
        # Agent is imported inside _delegate_task (deferred import)
        with patch("brahma.agent.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.run.return_value = "child result"

            result = _delegate_task(goal="Write a test")

        assert result == "child result"
        # Verify Agent was created
        mock_agent_cls.assert_called_once()
        call_kwargs = mock_agent_cls.call_args.kwargs
        assert "system_prompt" in call_kwargs
        assert "model" in call_kwargs
        assert "tools" in call_kwargs
        assert "max_turns" in call_kwargs
        assert call_kwargs["max_turns"] == 30
        # Child gets all bootstrap tools by default
        assert isinstance(call_kwargs["tools"], ToolRegistry)
        assert len(call_kwargs["tools"]) == 9

    def test_creates_child_with_subset_of_tools(self) -> None:
        """Child agent gets only the requested tools."""
        with patch("brahma.agent.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.run.return_value = "ok"

            _delegate_task(goal="Read a file", tools=["read_file", "write_file"])

        call_kwargs = mock_agent_cls.call_args.kwargs
        child_tools: ToolRegistry = call_kwargs["tools"]
        assert len(child_tools) == 2
        assert "read_file" in child_tools
        assert "write_file" in child_tools
        assert "terminal" not in child_tools

    def test_unknown_tool_names_skipped(self) -> None:
        """Unknown tool names are silently skipped."""
        with patch("brahma.agent.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.run.return_value = "ok"

            _delegate_task(goal="Test", tools=["read_file", "nonexistent_tool"])

        call_kwargs = mock_agent_cls.call_args.kwargs
        child_tools: ToolRegistry = call_kwargs["tools"]
        assert len(child_tools) == 1
        assert "read_file" in child_tools
        assert "nonexistent_tool" not in child_tools

    def test_default_model_is_deepseek_v4_pro(self) -> None:
        """When no model specified, defaults to deepseek-v4-pro."""
        with patch("brahma.agent.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.run.return_value = "ok"

            _delegate_task(goal="Test")

        call_kwargs = mock_agent_cls.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-v4-pro"

    def test_custom_model_is_passed(self) -> None:
        """Custom model string is passed to the child agent."""
        with patch("brahma.agent.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.run.return_value = "ok"

            _delegate_task(goal="Test", model="anthropic:claude-sonnet-4-20250514")

        call_kwargs = mock_agent_cls.call_args.kwargs
        assert call_kwargs["model"] == "anthropic:claude-sonnet-4-20250514"

    def test_context_is_included_in_prompt(self) -> None:
        """Context is appended to the task prompt."""
        with patch("brahma.agent.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.run.return_value = "ok"

            _delegate_task(goal="Do X", context="Here is some background info.")

        # Verify context was included in the prompt
        mock_agent_cls.return_value.run.assert_called_once()
        prompt_arg: str = mock_agent_cls.return_value.run.call_args.args[0]
        assert "GOAL: Do X" in prompt_arg
        assert "CONTEXT:" in prompt_arg
        assert "Here is some background info." in prompt_arg

    def test_no_context_in_prompt_when_empty(self) -> None:
        """When context is empty, only the goal is in the prompt."""
        with patch("brahma.agent.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.run.return_value = "ok"

            _delegate_task(goal="Simple task")

        prompt_arg: str = mock_agent_cls.return_value.run.call_args.args[0]
        assert prompt_arg == "GOAL: Simple task"

    def test_empty_tools_list_gives_zero_tools(self) -> None:
        """An empty tools list gives the child zero tools."""
        with patch("brahma.agent.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.run.return_value = "ok"

            _delegate_task(goal="Test", tools=[])

        call_kwargs = mock_agent_cls.call_args.kwargs
        child_tools: ToolRegistry = call_kwargs["tools"]
        assert len(child_tools) == 0
