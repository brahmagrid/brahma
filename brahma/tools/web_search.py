"""Bootstrap tool: web_search — search the internet for existing solutions.

Uses DuckDuckGo Lite (no API key required). Returns title, URL, and snippet
for each result. The agent uses this during the bootstrap GENERATE step to
discover existing libraries, tools, and solutions before writing code from scratch.
"""

from __future__ import annotations

import re
from html import unescape

from brahma.tools.registry import registry

WEB_SEARCH_SCHEMA = {
    "description": (
        "Search the web for information, libraries, tools, or solutions. "
        "Use this before generating code from scratch — an existing library or "
        "tool might already solve the problem. Returns title, URL, and snippet "
        "for each result."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'python library for PDF parsing').",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 8, max 20).",
                "default": 8,
            },
        },
        "required": ["query"],
    },
}


def web_search(query: str, limit: int = 8) -> str:
    """Search DuckDuckGo Lite and return structured results.

    Args:
        query: Search query string.
        limit: Max results to return (1-20, default 8).

    Returns:
        Formatted search results with title, URL, and snippet per result.
    """
    import httpx  # Heavy third-party dep — deferred import

    limit = max(1, min(limit, 20))

    try:
        # DuckDuckGo Lite — simple HTML, easy to parse, no JS required
        response = httpx.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            follow_redirects=True,
            timeout=15,
        )
        response.raise_for_status()
        html = response.text
    except Exception as exc:
        return f"ERROR: Search failed — {exc}"

    results = _parse_ddg_lite(html, limit)

    if not results:
        return f"No results found for '{query}'."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['title']}**")
        lines.append(f"   {r['url']}")
        lines.append(f"   {r['snippet']}")
        lines.append("")

    return "\n".join(lines).strip()


def _parse_ddg_lite(html: str, limit: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo Lite HTML into structured results.

    DDG Lite uses a simple structure:
        <a rel="nofollow" href="...">Title</a>
        <span class="link-text">display URL</span>
        <td class="result-snippet">snippet text</td>

    Args:
        html: Raw HTML from DuckDuckGo Lite.
        limit: Maximum number of results to return.

    Returns:
        List of dicts with title, url, and snippet keys.
    """
    results: list[dict[str, str]] = []

    # Extract result blocks — each is a table row with result-link, link-text, result-snippet
    # Pattern: find title+URL pairs, then look ahead for snippets
    link_pattern = re.compile(
        r'<a\s+rel="nofollow"\s+(?:class="[^"]*"\s+)?href="([^"]+)"[^>]*>'
        r"(.*?)</a>",
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<td\s+class="result-snippet">(.*?)</td>',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (url, raw_title) in enumerate(links):
        if len(results) >= limit:
            break

        title = _clean_html(raw_title)

        # Try to find corresponding snippet
        snippet = ""
        if i < len(snippets):
            snippet = _clean_html(snippets[i])

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
        })

    return results


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode HTML entities from text.

    Args:
        text: Raw HTML text potentially containing tags and entities.

    Returns:
        Clean, readable text.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities (&amp;, &lt;, &#39;, etc.)
    text = unescape(text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


registry.register(
    name="web_search",
    schema=WEB_SEARCH_SCHEMA,
    handler=web_search,
)
