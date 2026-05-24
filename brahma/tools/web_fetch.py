"""Bootstrap tool: web_fetch — fetch content from a URL."""

from __future__ import annotations

from brahma.tools.registry import registry

WEB_FETCH_SCHEMA = {
    "description": "Fetch content from a URL. Returns text content (HTML, JSON, etc.).",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch."},
        },
        "required": ["url"],
    },
}


def web_fetch(url: str) -> str:
    """Fetch a URL and return text content."""
    import httpx  # Heavy third-party dep — deferred import

    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        text = response.text

        # Truncate if too large
        if len(text) > 100_000:
            text = text[:100_000] + f"\n... (truncated {len(response.text) - 100_000} chars)"

        return text
    except Exception as exc:
        return f"ERROR: Failed to fetch {url}: {exc}"


registry.register(
    name="web_fetch",
    schema=WEB_FETCH_SCHEMA,
    handler=web_fetch,
)
