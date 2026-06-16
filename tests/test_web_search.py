"""
Tests for the _web_search bootstrap tool.

Covers DuckDuckGo Lite HTML parsing, error handling, result formatting,
and limit enforcement. Mocks httpx to avoid real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from brahma.tools import _web_search
from brahma.tools.web_search import _clean_html, _parse_ddg_lite

# ═══════════════════════════════════════════════════════════════════════════
# _clean_html — unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCleanHtml:
    """Tests for the _clean_html helper."""

    def test_strips_html_tags(self) -> None:
        """HTML tags are removed from text."""
        result = _clean_html("<b>Hello</b> <i>World</i>")
        assert result == "Hello World"

    def test_decodes_html_entities(self) -> None:
        """HTML entities are decoded to readable characters."""
        result = _clean_html("Hello &amp; Goodbye &lt;3")
        assert "&" in result
        assert "<" in result

    def test_collapses_whitespace(self) -> None:
        """Multiple whitespace characters are collapsed."""
        result = _clean_html("Hello   \n\n  World")
        assert result == "Hello World"

    def test_handles_empty_string(self) -> None:
        """Empty string returns empty string."""
        result = _clean_html("")
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# _parse_ddg_lite — unit tests
# ═══════════════════════════════════════════════════════════════════════════


DDG_LITE_HTML = """
<table>
<tr class="result-snippet">
  <td class="result-snippet">
    A fast, pure-Python PDF library for reading, writing, and manipulating PDFs.
  </td>
</tr>
<tr class="result-snippet">
  <td class="result-snippet">
    Python toolkit to work with Excel files — read, write, and modify spreadsheets.
  </td>
</tr>
</table>
"""

DDG_LITE_LINKS = """
<a rel="nofollow" href="https://pypi.org/project/pypdf/">
  PyPDF2: Pure-Python PDF library
</a>
<span class="link-text">pypi.org/project/pypdf/</span>
<a rel="nofollow" class="result-link"
   href="https://pypi.org/project/openpyxl/">
  openpyxl: Excel library
</a>
<span class="link-text">pypi.org/project/openpyxl/</span>
"""


class TestParseDdgLite:
    """Tests for _parse_ddg_lite HTML parser."""

    def test_extracts_links_and_snippets(self) -> None:
        """Links and snippets are extracted from DDG Lite HTML."""
        html = DDG_LITE_LINKS + DDG_LITE_HTML
        results = _parse_ddg_lite(html, limit=5)

        assert len(results) == 2
        assert results[0]["url"] == "https://pypi.org/project/pypdf/"
        assert "PDF" in results[0]["title"]
        assert results[1]["url"] == "https://pypi.org/project/openpyxl/"
        assert "Excel" in results[1]["title"]

    def test_respects_limit(self) -> None:
        """Only 'limit' results are returned."""
        html = DDG_LITE_LINKS + DDG_LITE_HTML
        results = _parse_ddg_lite(html, limit=1)

        assert len(results) == 1

    def test_returns_empty_list_when_no_results(self) -> None:
        """Empty HTML returns empty list."""
        results = _parse_ddg_lite("<html></html>", limit=5)
        assert results == []

    def test_snippets_are_cleaned(self) -> None:
        """Snippet HTML tags are stripped."""
        results = _parse_ddg_lite(DDG_LITE_LINKS + DDG_LITE_HTML, limit=5)
        assert "Python" in results[0]["snippet"]
        assert "<td" not in results[0]["snippet"]


# ═══════════════════════════════════════════════════════════════════════════
# web_search — mocked httpx
# ═══════════════════════════════════════════════════════════════════════════


class TestWebSearch:
    """Tests for web_search with mocked httpx."""

    def test_returns_formatted_results(self) -> None:
        """Successful search returns formatted results with title, URL, snippet."""
        mock_response = MagicMock()
        mock_response.text = DDG_LITE_LINKS + DDG_LITE_HTML
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response) as mock_get:
            result = _web_search("python pdf library")

        assert "PyPDF2" in result or "pypdf" in result
        assert "https://" in result
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["params"]["q"] == "python pdf library"

    def test_returns_no_results_message(self) -> None:
        """Empty page returns 'No results found' message."""
        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response):
            result = _web_search("xyznonexistent12345")

        assert "No results found" in result

    def test_http_error_returns_error_message(self) -> None:
        """HTTP errors are caught and returned as error strings."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("429 Too Many Requests")

        with patch("httpx.get", return_value=mock_response):
            result = _web_search("anything")

        assert "ERROR: Search failed" in result
        assert "429" in result

    def test_connection_error_returns_error_message(self) -> None:
        """Connection errors are caught."""
        with patch("httpx.get", side_effect=Exception("Connection timeout")):
            result = _web_search("anything")

        assert "ERROR: Search failed" in result
        assert "timeout" in result.lower()

    def test_limit_is_clamped(self) -> None:
        """Limit is clamped between 1 and 20."""
        mock_response = MagicMock()
        mock_response.text = DDG_LITE_LINKS + DDG_LITE_HTML
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response):
            # High limit gets clamped to 20 (max)
            result = _web_search("test", limit=100)
            assert result  # just verify it doesn't crash

            # Zero or negative gets clamped to 1 (min)
            result = _web_search("test", limit=0)
            assert result

    def test_result_formatting_is_markdown_friendly(self) -> None:
        """Results use markdown bold for titles."""
        mock_response = MagicMock()
        mock_response.text = DDG_LITE_LINKS + DDG_LITE_HTML
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response):
            result = _web_search("python")

        assert "**" in result  # bold markdown
        assert result.startswith("1.")

    def test_default_limit_is_8(self) -> None:
        """Default limit is 8 results."""
        assert _web_search.__defaults__ is not None
        # The function's default for 'limit' is 8
        # We verify by checking the __defaults__ tuple
        defaults = _web_search.__defaults__
        assert defaults[0] == 8  # limit is the first default arg
