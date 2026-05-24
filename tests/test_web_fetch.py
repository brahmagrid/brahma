"""
Tests for the _web_fetch bootstrap tool.

Covers HTTP fetching, error handling, redirects, and truncation.
Mocks httpx to avoid real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from brahma.tools import _web_fetch

# ═══════════════════════════════════════════════════════════════════════════
# _web_fetch — mocked httpx
# ═══════════════════════════════════════════════════════════════════════════


class TestWebFetch:
    """Tests for _web_fetch with mocked httpx."""

    def test_fetches_url_successfully(self) -> None:
        """Returns text content from a successful HTTP response."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Hello World</body></html>"
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response) as mock_get:
            result = _web_fetch("https://example.com")

        assert "Hello World" in result
        mock_get.assert_called_once_with("https://example.com", follow_redirects=True, timeout=30)

    def test_http_error_returns_error_message(self) -> None:
        """HTTP errors are caught and returned as error strings."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        with patch("httpx.get", return_value=mock_response):
            result = _web_fetch("https://example.com/missing")

        assert result.startswith("ERROR: Failed to fetch")
        assert "404 Not Found" in result

    def test_connection_error_returns_error_message(self) -> None:
        """Connection errors are caught and returned as error strings."""
        with patch(
            "httpx.get",
            side_effect=Exception("Connection refused"),
        ):
            result = _web_fetch("https://unreachable.example.com")

        assert result.startswith("ERROR: Failed to fetch")
        assert "Connection refused" in result

    def test_follows_redirects(self) -> None:
        """Httpx is called with follow_redirects=True."""
        mock_response = MagicMock()
        mock_response.text = "final destination"
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response) as mock_get:
            _web_fetch("https://redirect.example.com")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["follow_redirects"] is True

    def test_truncates_large_responses(self) -> None:
        """Responses over 100,000 chars are truncated."""
        large_text = "x" * 150_000
        mock_response = MagicMock()
        mock_response.text = large_text
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response):
            result = _web_fetch("https://example.com/large")

        assert len(result) < len(large_text)
        assert "(truncated" in result
        assert "50000 chars" in result  # 150000 - 100000 = 50000

    def test_small_response_not_truncated(self) -> None:
        """Responses under 100,000 chars are returned in full."""
        small_text = "Hello" * 100  # 500 chars
        mock_response = MagicMock()
        mock_response.text = small_text
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response):
            result = _web_fetch("https://example.com/small")

        assert result == small_text
        assert "(truncated" not in result

    def test_empty_response(self) -> None:
        """Empty response body is handled."""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response):
            result = _web_fetch("https://example.com/empty")

        assert result == ""

    def test_json_response(self) -> None:
        """JSON responses are returned as text."""
        mock_response = MagicMock()
        mock_response.text = '{"key": "value"}'
        mock_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_response):
            result = _web_fetch("https://api.example.com/data")

        assert '"key": "value"' in result
