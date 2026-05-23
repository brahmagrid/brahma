"""Brahma — HITL (Human-in-the-Loop) system.

HITL requests are stored in an in-memory queue. Agents create requests,
poll for responses, and continue once a human responds via the client.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class HITLStatus(StrEnum):
    """Status of a HITL request."""

    PENDING = "pending"
    RESPONDED = "responded"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass
class HITLRequest:
    """A single human-in-the-loop interaction."""

    id: str
    agent_id: str
    task_summary: str
    prompt: str
    request_type: str  # "approval", "input", "choice", "captcha"
    choices: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    status: HITLStatus = HITLStatus.PENDING
    response: str | None = None
    created_at: str = ""
    responded_at: str | None = None

    def __post_init__(self) -> None:
        """Set created_at timestamp if not already set."""
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict:
        """Serialize the request to a JSON-compatible dict."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "task_summary": self.task_summary,
            "prompt": self.prompt,
            "request_type": self.request_type,
            "choices": self.choices,
            "status": self.status.value,
            "response": self.response,
            "created_at": self.created_at,
            "responded_at": self.responded_at,
        }


class HITLQueue:
    """Thread-safe queue for HITL requests."""

    def __init__(self) -> None:
        """Initialize an empty queue."""
        self._lock = threading.Lock()
        self._requests: dict[str, HITLRequest] = {}
        self._responses: dict[str, str] = {}  # request_id → response text

    def create(
        self,
        agent_id: str,
        task_summary: str,
        prompt: str,
        request_type: str = "approval",
        choices: list[str] | None = None,
        context: dict[str, object] | None = None,
    ) -> str:
        """Create a HITL request. Returns the request_id."""
        request_id = f"hitl-{uuid.uuid4().hex[:8]}"
        req = HITLRequest(
            id=request_id,
            agent_id=agent_id,
            task_summary=task_summary,
            prompt=prompt,
            request_type=request_type,
            choices=choices or [],
            context=context or {},
        )
        with self._lock:
            self._requests[request_id] = req
        return request_id

    def respond(self, request_id: str, response: str) -> bool:
        """Record a human response. Returns True if the request exists."""
        with self._lock:
            req = self._requests.get(request_id)
            if not req or req.status != HITLStatus.PENDING:
                return False
            req.status = HITLStatus.RESPONDED
            req.response = response
            req.responded_at = datetime.now(UTC).isoformat()
            self._responses[request_id] = response
        return True

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request."""
        with self._lock:
            req = self._requests.get(request_id)
            if not req or req.status != HITLStatus.PENDING:
                return False
            req.status = HITLStatus.CANCELLED
        return True

    def get(self, request_id: str) -> HITLRequest | None:
        """Retrieve a HITL request by ID, or None if not found."""
        with self._lock:
            return self._requests.get(request_id)

    def list_pending(self) -> list[dict]:
        """List all pending requests (for the client to display)."""
        with self._lock:
            return [
                req.to_dict() for req in self._requests.values() if req.status == HITLStatus.PENDING
            ]

    def list_all(self) -> list[dict]:
        """List all requests (for debugging)."""
        with self._lock:
            return [req.to_dict() for req in self._requests.values()]

    def wait_for_response(
        self,
        request_id: str,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> dict:
        """Block until a human responds or timeout expires.

        Called by the agent's hitl_request tool. Polls the queue every
        poll_interval seconds until a response arrives or timeout is reached.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                req = self._requests.get(request_id)
                if req and req.status == HITLStatus.RESPONDED:
                    return {
                        "status": "resolved",
                        "response": req.response,
                        "request_id": request_id,
                    }
                if req and req.status == HITLStatus.CANCELLED:
                    return {
                        "status": "cancelled",
                        "response": None,
                        "request_id": request_id,
                    }
            time.sleep(poll_interval)

        # Timeout
        with self._lock:
            req = self._requests.get(request_id)
            if req and req.status == HITLStatus.PENDING:
                req.status = HITLStatus.TIMED_OUT
        return {
            "status": "timed_out",
            "response": None,
            "request_id": request_id,
        }
