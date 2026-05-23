"""
Tests for Brahma tool system.

Covers ToolRegistry, ToolResult, and all 7 bootstrap tool implementations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import pytest

from brahma.tools import (
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    ToolRegistry,
    ToolResult,
    _read_file,
    _search_files,
    _skill_manage,
    _terminal,
    _write_file,
    bootstrap_tools,
)

# ═══════════════════════════════════════════════════════════════════════════
# ToolResult
# ═══════════════════════════════════════════════════════════════════════════


class TestToolResult:
    """Tests for the ToolResult dataclass."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        result = ToolResult(tool="test", success=True, output="hello")
        assert result.tool == "test"
        assert result.success is True
        assert result.output == "hello"
        assert result.error == ""
        assert result.call_id == ""

    def test_failure_result(self) -> None:
        """Failure result stores error message."""
        result = ToolResult(tool="fail", success=False, output="", error="something broke")
        assert result.success is False
        assert result.error == "something broke"

    def test_call_id_assignment(self) -> None:
        """call_id can be set after construction."""
        result = ToolResult(tool="x", success=True, output="ok")
        result.call_id = "call_abc123"
        assert result.call_id == "call_abc123"


# ═══════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ═══════════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    """Tests for the ToolRegistry dataclass."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Return an empty registry."""
        return ToolRegistry()

    @pytest.fixture
    def populated_registry(self) -> ToolRegistry:
        """Return a registry with two tools registered."""
        reg = ToolRegistry()

        def greet(name: str) -> str:
            return f"Hello, {name}"

        def add(a: int, b: int) -> int:
            return a + b

        reg.register("greet", greet, {"description": "Greet someone"})
        reg.register("add", add, {"description": "Add two numbers"})
        return reg

    # -- register / get / contains / len -------------------------------

    def test_register_and_get(self, registry: ToolRegistry) -> None:
        """Tools can be registered and retrieved."""
        fn = lambda x: x  # noqa: E731
        registry.register("identity", fn, {"description": "Identity function"})
        assert registry.get("identity") is fn

    def test_get_unknown_returns_none(self, registry: ToolRegistry) -> None:
        """Getting an unregistered tool returns None."""
        assert registry.get("nonexistent") is None

    def test_contains(self, populated_registry: ToolRegistry) -> None:
        """__contains__ checks registration status."""
        assert "greet" in populated_registry
        assert "multiply" not in populated_registry

    def test_len(self, populated_registry: ToolRegistry) -> None:
        """__len__ returns count of registered tools."""
        assert len(populated_registry) == 2

    def test_list_tools(self, populated_registry: ToolRegistry) -> None:
        """list_tools returns all registered tool names."""
        names = populated_registry.list_tools()
        assert set(names) == {"greet", "add"}

    # -- execute -------------------------------------------------------

    def test_execute_success(self, populated_registry: ToolRegistry) -> None:
        """Execute a registered tool successfully."""
        result = populated_registry.execute("greet", {"name": "World"})
        assert result.success is True
        assert result.output == "Hello, World"

    def test_execute_unknown_tool(self, populated_registry: ToolRegistry) -> None:
        """Executing an unknown tool returns a failure result."""
        result = populated_registry.execute("no_such_tool", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_execute_tool_raises(self, populated_registry: ToolRegistry) -> None:
        """Tool exceptions are caught and returned as failures."""

        def blow_up() -> str:
            raise ValueError("kapow")

        populated_registry.register("fragile", blow_up, {"description": "Will fail"})
        result = populated_registry.execute("fragile", {})
        assert result.success is False
        assert "kapow" in result.error

    # -- schemas -------------------------------------------------------

    def test_schemas_format(self, populated_registry: ToolRegistry) -> None:
        """schemas() returns Anthropic/OpenAI-compatible format."""
        schemas = populated_registry.schemas()
        assert len(schemas) == 2
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "input_schema" in s


# ═══════════════════════════════════════════════════════════════════════════
# _read_file
# ═══════════════════════════════════════════════════════════════════════════


class TestReadFile:
    """Tests for the _read_file bootstrap tool."""

    def test_reads_existing_file(self) -> None:
        """Reads all lines from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line one\nline two\nline three\n")
            path = f.name
        try:
            output = _read_file(path)
            assert "line one" in output
            assert "line two" in output
            assert "line three" in output
        finally:
            Path(path).unlink()

    def test_file_not_found(self) -> None:
        """Returns error for missing file."""
        output = _read_file("/nonexistent/path/file.txt")
        assert output.startswith("ERROR: File not found")

    def test_offset(self) -> None:
        """Offset skips lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("A\nB\nC\nD\nE\n")
            path = f.name
        try:
            output = _read_file(path, offset=3)
            assert "A" not in output.splitlines()[0]
            assert "C" in output
            assert "... (2 lines above)" in output
        finally:
            Path(path).unlink()

    def test_limit(self) -> None:
        """Limit restricts lines returned."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(str(i) for i in range(20)))
            path = f.name
        try:
            output = _read_file(path, limit=5)
            lines = output.splitlines()
            assert any(f"{i + 1:6d}" in line for i, line in enumerate(lines) if "|" in line)
            assert "... (15 more lines)" in output
        finally:
            Path(path).unlink()

    def test_line_numbers(self) -> None:
        """Output includes line numbers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("first\nsecond\n")
            path = f.name
        try:
            output = _read_file(path)
            assert "     1|" in output
            assert "     2|" in output
        finally:
            Path(path).unlink()


# ═══════════════════════════════════════════════════════════════════════════
# _write_file
# ═══════════════════════════════════════════════════════════════════════════


class TestWriteFile:
    """Tests for the _write_file bootstrap tool."""

    def test_writes_new_file(self) -> None:
        """Creates a file with the given content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.txt")
            result = _write_file(path, "hello world")
            assert "Wrote" in result
            assert Path(path).read_text() == "hello world"

    def test_creates_parent_directories(self) -> None:
        """Creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "deep" / "nested" / "file.txt")
            result = _write_file(path, "nested content")
            assert "Wrote" in result
            assert Path(path).read_text() == "nested content"

    def test_overwrites_existing_file(self) -> None:
        """Overwrites files that already exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.txt")
            _write_file(path, "first write")
            _write_file(path, "second write")
            assert Path(path).read_text() == "second write"

    def test_expands_tilde(self) -> None:
        """Expands ~ in path to home directory."""
        result = _write_file("~/test_brahma_write_test.txt", "tilde test")
        p = Path.home() / "test_brahma_write_test.txt"
        try:
            assert "Wrote" in result
            assert p.exists()
        finally:
            p.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# _search_files
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchFiles:
    """Tests for the _search_files bootstrap tool."""

    def test_finds_matches(self, tmp_path: Path) -> None:
        """Finds lines matching a pattern."""
        (tmp_path / "a.py").write_text("import os\nimport sys\n")
        (tmp_path / "b.py").write_text("from pathlib import Path\n")
        output = _search_files("import", path=str(tmp_path))
        assert "import os" in output
        assert "import sys" in output

    def test_no_matches(self, tmp_path: Path) -> None:
        """Reports when no matches are found."""
        (tmp_path / "a.py").write_text("hello world\n")
        output = _search_files("nonexistent_pattern_xyz", path=str(tmp_path))
        assert output.startswith("No matches")

    def test_file_glob_filter(self, tmp_path: Path) -> None:
        """Filters files by glob pattern."""
        (tmp_path / "code.py").write_text("import os\n")
        (tmp_path / "notes.txt").write_text("import os\n")
        output = _search_files("import", path=str(tmp_path), file_glob="*.py")
        assert "code.py" in output
        assert "notes.txt" not in output


# ═══════════════════════════════════════════════════════════════════════════
# _terminal
# ═══════════════════════════════════════════════════════════════════════════


class TestTerminal:
    """Tests for the _terminal bootstrap tool."""

    def test_simple_command(self) -> None:
        """Runs a simple echo command."""
        output = _terminal("echo hello")
        assert output == "hello"

    def test_stderr_captured(self) -> None:
        """Stderr is appended to output."""
        output = _terminal("echo error >&2")
        assert "[stderr]" in output
        assert "error" in output

    def test_exit_code_reported(self) -> None:
        """Non-zero exit codes are reported."""
        output = _terminal("exit 1")
        assert "[exit code: 1]" in output

    def test_no_output_command(self) -> None:
        """Commands with no output return '(no output)'."""
        output = _terminal("true")
        assert output == "(no output)"

    def test_timeout(self) -> None:
        """Command timeout produces an error message."""
        output = _terminal("sleep 5", timeout=1)
        assert "timed out" in output.lower()


# ═══════════════════════════════════════════════════════════════════════════
# _skill_manage
# ═══════════════════════════════════════════════════════════════════════════


class TestSkillManage:
    """Tests for the _skill_manage bootstrap tool."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Redirect SKILLS_DIR to a temp directory for isolation."""
        monkeypatch.setattr("brahma.tools.SKILLS_DIR", tmp_path / ".brahma" / "skills")
        self.skills_dir = tmp_path / ".brahma" / "skills"

    def test_save_and_load(self) -> None:
        """Skills can be saved and loaded back."""
        save_result = _skill_manage("save", "test-skill", "# Hello World")
        assert "saved" in save_result
        load_result = _skill_manage("load", "test-skill")
        assert load_result == "# Hello World"

    def test_load_nonexistent(self) -> None:
        """Loading a missing skill returns an error."""
        result = _skill_manage("load", "does-not-exist")
        assert "ERROR" in result
        assert "not found" in result

    def test_list_empty(self) -> None:
        """Listing when no skills exist returns appropriate message."""
        result = _skill_manage("list", "dummy")
        assert "No skills saved yet" in result

    def test_list_with_skills(self) -> None:
        """List returns saved skill names."""
        _skill_manage("save", "alpha", "content")
        _skill_manage("save", "beta", "content")
        result = _skill_manage("list", "dummy")
        assert "alpha" in result
        assert "beta" in result

    def test_delete(self) -> None:
        """Skills can be deleted."""
        _skill_manage("save", "to-delete", "content")
        result = _skill_manage("delete", "to-delete")
        assert "deleted" in result
        load_result = _skill_manage("load", "to-delete")
        assert "ERROR" in load_result

    def test_delete_nonexistent(self) -> None:
        """Deleting a missing skill returns an error."""
        result = _skill_manage("delete", "no-such-skill")
        assert "ERROR" in result

    def test_unknown_action(self) -> None:
        """Unknown actions return an error."""
        result = _skill_manage("rename", "x", "y")
        assert "ERROR" in result
        assert "Unknown action" in result


# ═══════════════════════════════════════════════════════════════════════════
# bootstrap_tools factory
# ═══════════════════════════════════════════════════════════════════════════


class TestBootstrapTools:
    """Tests for the bootstrap_tools() factory function."""

    def test_returns_registry_with_7_tools(self) -> None:
        """The factory returns a registry with all 7 bootstrap tools."""
        registry = bootstrap_tools()
        assert len(registry) == 7

    def test_all_expected_tools_present(self) -> None:
        """All expected tool names are registered."""
        registry = bootstrap_tools()
        expected = {
            "read_file",
            "write_file",
            "search_files",
            "terminal",
            "delegate_task",
            "skill_manage",
            "web_fetch",
        }
        assert set(registry.list_tools()) == expected

    def test_tools_are_callable(self) -> None:
        """Each tool function can be retrieved and is callable."""
        registry = bootstrap_tools()
        for name in registry.list_tools():
            fn = registry.get(name)
            assert fn is not None, f"Tool '{name}' not retrievable"
            assert callable(fn), f"Tool '{name}' is not callable"

    def test_tools_have_schemas(self) -> None:
        """Each tool schema has required fields."""
        registry = bootstrap_tools()
        schemas = registry.schemas()
        assert len(schemas) == 7
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            assert schema["description"] != "", f"Tool '{schema['name']}' has no description"

    def test_read_file_can_be_executed(self) -> None:
        """A bootstrap tool can be executed through the registry."""
        registry = bootstrap_tools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content\n")
            path = f.name
        try:
            result = registry.execute("read_file", {"path": path})
            assert result.success is True
            assert "test content" in result.output
        finally:
            Path(path).unlink()


# ═══════════════════════════════════════════════════════════════════════════
# Schema integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestSchemas:
    """Tests for tool schema definitions."""

    def test_read_file_schema_has_required_path(self) -> None:
        """The read_file schema requires 'path'."""
        input_schema = cast(dict, READ_FILE_SCHEMA["input_schema"])
        assert "path" in input_schema["required"]

    def test_write_file_schema_requires_both_fields(self) -> None:
        """The write_file schema requires 'path' and 'content'."""
        input_schema = cast(dict, WRITE_FILE_SCHEMA["input_schema"])
        assert "path" in input_schema["required"]
        assert "content" in input_schema["required"]
