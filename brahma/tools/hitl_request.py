"""Bootstrap tool: hitl_request — request human-in-the-loop input."""

from __future__ import annotations

from brahma.tools.registry import registry

# ── Module-level HITL queue reference  ────────────────────────────────────
# Set by the server on startup via set_hitl_queue().  Tools that need HITL
# poll this reference; None means HITL is not configured.

_hitl_queue: object = None  # HITLQueue | None


def set_hitl_queue(queue: object) -> None:
    """Set the HITL queue reference (called by server.create_app)."""
    global _hitl_queue
    _hitl_queue = queue


HITL_REQUEST_SCHEMA = {
    "description": (
        "Request human input for a task you cannot complete autonomously. "
        "Blocks until the human responds via the Brahma Client UI, or the "
        "timeout expires. Use for: CAPTCHAs, auth flows, destructive action "
        "approval, ambiguous instructions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "request_type": {
                "type": "string",
                "enum": ["approval", "input", "choice", "captcha"],
                "description": (
                    "approval: yes/no confirmation for an action. "
                    "input: free-text response. "
                    "choice: pick from provided options. "
                    "captcha: human must solve a visual challenge."
                ),
            },
            "prompt": {
                "type": "string",
                "description": "The question or request to show the human.",
            },
            "task_summary": {
                "type": "string",
                "description": "Brief one-line summary of the overall task context.",
            },
            "choices": {
                "type": "string",
                "description": "Comma-separated options (only for request_type='choice').",
            },
            "timeout": {
                "type": "number",
                "description": "Seconds to wait for a human response (default: 300).",
            },
        },
        "required": ["request_type", "prompt"],
    },
}


def hitl_request(
    request_type: str,
    prompt: str,
    task_summary: str = "",
    choices: str = "",
    timeout: float = 300.0,
) -> str:
    """Request human input. Blocks until the human responds or timeout expires.

    Use this when you encounter a task you cannot complete autonomously:
    - CAPTCHA verification
    - Authentication flows
    - Approval for irreversible actions (publishing, payments, deletions)
    - Ambiguous instructions that need human clarification

    The request appears in the Brahma Client UI. A human reviews and responds.
    This tool blocks until a response arrives or the timeout is reached.
    """
    if _hitl_queue is None:
        return "ERROR: HITL not configured. Start the Brahma server to enable human-in-the-loop."

    choice_list = [c.strip() for c in choices.split(",") if c.strip()] if choices else []

    request_id = _hitl_queue.create(  # ty: ignore[unresolved-attribute]
        agent_id="brahma-agent",
        task_summary=task_summary or prompt[:100],
        prompt=prompt,
        request_type=request_type,
        choices=choice_list,
    )

    result = _hitl_queue.wait_for_response(  # ty: ignore[unresolved-attribute]
        request_id, timeout=timeout
    )

    if result["status"] == "resolved":
        return f"HUMAN RESPONSE: {result['response']}"
    elif result["status"] == "cancelled":
        return "HITL_CANCELLED: The request was cancelled."
    else:
        return f"HITL_TIMEOUT: No human response within {timeout}s."


# Check function: HITL is available only when the server has wired a queue.
def _hitl_available() -> bool:
    return _hitl_queue is not None


registry.register(
    name="hitl_request",
    schema=HITL_REQUEST_SCHEMA,
    handler=hitl_request,
    check_fn=_hitl_available,
)
