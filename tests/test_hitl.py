"""
Tests for Brahma HITL (Human-in-the-Loop) system.

Covers HITLRequest, HITLQueue, request lifecycle, and wait_for_response.
"""

from __future__ import annotations

import time
from threading import Thread

import pytest

from brahma.hitl import HITLQueue, HITLRequest, HITLStatus

# ═══════════════════════════════════════════════════════════════════════════
# HITLRequest
# ═══════════════════════════════════════════════════════════════════════════


class TestHITLRequest:
    """Tests for the HITLRequest dataclass."""

    def test_created_with_defaults(self) -> None:
        """A new request has sensible defaults."""
        req = HITLRequest(
            id="hitl-abc123",
            agent_id="brahma-001",
            task_summary="Approve deployment",
            prompt="Should we deploy to production?",
            request_type="approval",
        )
        assert req.id == "hitl-abc123"
        assert req.agent_id == "brahma-001"
        assert req.request_type == "approval"
        assert req.status == HITLStatus.PENDING
        assert req.response is None
        assert req.responded_at is None
        assert req.choices == []
        assert req.context == {}

    def test_created_at_is_set_automatically(self) -> None:
        """created_at is populated in __post_init__ if empty."""
        req = HITLRequest(
            id="hitl-001",
            agent_id="brahma-001",
            task_summary="Test",
            prompt="Question?",
            request_type="input",
        )
        assert req.created_at != ""
        assert "T" in req.created_at  # ISO 8601 format

    def test_created_at_respected_if_provided(self) -> None:
        """If created_at is provided, it is not overwritten."""
        req = HITLRequest(
            id="hitl-001",
            agent_id="brahma-001",
            task_summary="Test",
            prompt="Q",
            request_type="input",
            created_at="2025-01-01T00:00:00+00:00",
        )
        assert req.created_at == "2025-01-01T00:00:00+00:00"

    def test_choices_are_stored(self) -> None:
        """Choices list is stored."""
        req = HITLRequest(
            id="hitl-001",
            agent_id="brahma-001",
            task_summary="Pick one",
            prompt="Choose an option",
            request_type="choice",
            choices=["Option A", "Option B"],
        )
        assert req.choices == ["Option A", "Option B"]

    def test_to_dict_serialization(self) -> None:
        """to_dict produces a JSON-compatible dictionary."""
        req = HITLRequest(
            id="hitl-test",
            agent_id="brahma-001",
            task_summary="Summary",
            prompt="Do you approve?",
            request_type="approval",
            choices=["yes", "no"],
        )
        d = req.to_dict()
        assert d["id"] == "hitl-test"
        assert d["agent_id"] == "brahma-001"
        assert d["task_summary"] == "Summary"
        assert d["prompt"] == "Do you approve?"
        assert d["request_type"] == "approval"
        assert d["choices"] == ["yes", "no"]
        assert d["status"] == "pending"
        assert d["response"] is None
        assert d["created_at"] is not None
        assert d["responded_at"] is None

    def test_to_dict_after_response(self) -> None:
        """to_dict reflects status and response after resolution."""
        req = HITLRequest(
            id="hitl-001",
            agent_id="brahma-001",
            task_summary="Test",
            prompt="Q",
            request_type="input",
        )
        req.status = HITLStatus.RESPONDED
        req.response = "my answer"
        req.responded_at = "2025-06-01T12:00:00+00:00"
        d = req.to_dict()
        assert d["status"] == "responded"
        assert d["response"] == "my answer"
        assert d["responded_at"] == "2025-06-01T12:00:00+00:00"


# ═══════════════════════════════════════════════════════════════════════════
# HITLQueue
# ═══════════════════════════════════════════════════════════════════════════


class TestHITLQueue:
    """Tests for the HITLQueue thread-safe queue."""

    @pytest.fixture
    def queue(self) -> HITLQueue:
        """Fresh HITLQueue for each test."""
        return HITLQueue()

    def test_create_returns_request_id(self, queue: HITLQueue) -> None:
        """create() returns a hitl- prefixed ID."""
        req_id = queue.create(
            agent_id="brahma-001",
            task_summary="Test task",
            prompt="Please confirm",
        )
        assert req_id.startswith("hitl-")
        assert len(req_id) == 13  # "hitl-" + 8 hex chars

    def test_unique_ids(self, queue: HITLQueue) -> None:
        """Each request gets a unique ID."""
        id1 = queue.create(agent_id="a", task_summary="1", prompt="?")
        id2 = queue.create(agent_id="a", task_summary="2", prompt="?")
        assert id1 != id2

    def test_get_returns_request(self, queue: HITLQueue) -> None:
        """get() retrieves a created request."""
        req_id = queue.create(
            agent_id="brahma-001",
            task_summary="Test",
            prompt="Question?",
            request_type="input",
        )
        req = queue.get(req_id)
        assert req is not None
        assert req.id == req_id
        assert req.agent_id == "brahma-001"
        assert req.prompt == "Question?"
        assert req.request_type == "input"
        assert req.status == HITLStatus.PENDING

    def test_get_nonexistent_returns_none(self, queue: HITLQueue) -> None:
        """get() on unknown ID returns None."""
        assert queue.get("nonexistent-id") is None

    def test_list_pending_empty(self, queue: HITLQueue) -> None:
        """list_pending() returns empty list when no requests."""
        assert queue.list_pending() == []

    def test_list_pending_with_requests(self, queue: HITLQueue) -> None:
        """list_pending() returns only pending requests."""
        id1 = queue.create(agent_id="a", task_summary="T1", prompt="?")
        id2 = queue.create(agent_id="a", task_summary="T2", prompt="?")
        queue.respond(id2, "ok")

        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0]["id"] == id1

    def test_list_all(self, queue: HITLQueue) -> None:
        """list_all() returns all requests regardless of status."""
        id1 = queue.create(agent_id="a", task_summary="1", prompt="?")
        queue.create(agent_id="a", task_summary="2", prompt="?")
        all_reqs = queue.list_all()
        assert len(all_reqs) == 2
        ids = {r["id"] for r in all_reqs}
        assert id1 in ids

    def test_respond_success(self, queue: HITLQueue) -> None:
        """respond() marks a request as RESPONDED and stores the response."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        ok = queue.respond(req_id, "yes")
        assert ok is True

        req = queue.get(req_id)
        assert req is not None
        assert req.status == HITLStatus.RESPONDED
        assert req.response == "yes"
        assert req.responded_at is not None

    def test_respond_nonexistent(self, queue: HITLQueue) -> None:
        """respond() on unknown ID returns False."""
        assert queue.respond("no-such-id", "x") is False

    def test_respond_already_resolved(self, queue: HITLQueue) -> None:
        """respond() on already-responded request returns False."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        queue.respond(req_id, "first")
        assert queue.respond(req_id, "second") is False

    def test_cancel_success(self, queue: HITLQueue) -> None:
        """cancel() marks a request as CANCELLED."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        ok = queue.cancel(req_id)
        assert ok is True

        req = queue.get(req_id)
        assert req is not None
        assert req.status == HITLStatus.CANCELLED

    def test_cancel_nonexistent(self, queue: HITLQueue) -> None:
        """cancel() on unknown ID returns False."""
        assert queue.cancel("no-such-id") is False

    def test_cancel_already_resolved(self, queue: HITLQueue) -> None:
        """cancel() on already-responded request returns False."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        queue.respond(req_id, "ok")
        assert queue.cancel(req_id) is False

    def test_cancelled_not_in_pending(self, queue: HITLQueue) -> None:
        """Cancelled requests don't appear in list_pending()."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        queue.cancel(req_id)
        assert queue.list_pending() == []

    def test_create_with_choices(self, queue: HITLQueue) -> None:
        """Choices are stored with the request."""
        req_id = queue.create(
            agent_id="a",
            task_summary="Pick",
            prompt="Choose:",
            request_type="choice",
            choices=["A", "B", "C"],
        )
        req = queue.get(req_id)
        assert req is not None
        assert req.choices == ["A", "B", "C"]

    def test_create_with_context(self, queue: HITLQueue) -> None:
        """Context dict is stored with the request."""
        req_id = queue.create(
            agent_id="a",
            task_summary="T",
            prompt="?",
            context={"key": "value", "nested": {"a": 1}},
        )
        req = queue.get(req_id)
        assert req is not None
        assert req.context == {"key": "value", "nested": {"a": 1}}

    def test_default_request_type_is_approval(self, queue: HITLQueue) -> None:
        """Default request_type is 'approval'."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        req = queue.get(req_id)
        assert req is not None
        assert req.request_type == "approval"

    def test_choices_default_to_empty_list(self, queue: HITLQueue) -> None:
        """When choices is None, it defaults to empty list."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        req = queue.get(req_id)
        assert req is not None
        assert req.choices == []

    def test_context_default_is_empty_dict(self, queue: HITLQueue) -> None:
        """When context is not provided, it defaults to empty dict."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        req = queue.get(req_id)
        assert req is not None
        assert req.context == {}


# ═══════════════════════════════════════════════════════════════════════════
# HITLQueue — wait_for_response
# ═══════════════════════════════════════════════════════════════════════════


class TestWaitForResponse:
    """Tests for HITLQueue.wait_for_response()."""

    @pytest.fixture
    def queue(self) -> HITLQueue:
        """Fresh HITLQueue for each test."""
        return HITLQueue()

    def test_returns_resolved_when_responded(self, queue: HITLQueue) -> None:
        """Returns resolved status when a response arrives."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")

        def respond_after_delay() -> None:
            time.sleep(0.05)
            queue.respond(req_id, "approved")

        Thread(target=respond_after_delay, daemon=True).start()

        result = queue.wait_for_response(req_id, timeout=5.0, poll_interval=0.01)
        assert result["status"] == "resolved"
        assert result["response"] == "approved"
        assert result["request_id"] == req_id

    def test_returns_cancelled_when_cancelled(self, queue: HITLQueue) -> None:
        """Returns cancelled status when the request is cancelled."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")

        def cancel_after_delay() -> None:
            time.sleep(0.05)
            queue.cancel(req_id)

        Thread(target=cancel_after_delay, daemon=True).start()

        result = queue.wait_for_response(req_id, timeout=5.0, poll_interval=0.01)
        assert result["status"] == "cancelled"
        assert result["response"] is None

    def test_returns_timed_out_on_timeout(self, queue: HITLQueue) -> None:
        """Returns timed_out when no response arrives before timeout."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        result = queue.wait_for_response(req_id, timeout=0.1, poll_interval=0.05)
        assert result["status"] == "timed_out"
        assert result["response"] is None

        # Request should now be marked as timed out
        req = queue.get(req_id)
        assert req is not None
        assert req.status == HITLStatus.TIMED_OUT

    def test_already_responded_returns_immediately(self, queue: HITLQueue) -> None:
        """Returns immediately if the request was already responded to."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        queue.respond(req_id, "done")

        result = queue.wait_for_response(req_id, timeout=5.0)
        assert result["status"] == "resolved"
        assert result["response"] == "done"

    def test_already_cancelled_returns_immediately(self, queue: HITLQueue) -> None:
        """Returns immediately if the request was already cancelled."""
        req_id = queue.create(agent_id="a", task_summary="T", prompt="?")
        queue.cancel(req_id)

        result = queue.wait_for_response(req_id, timeout=5.0)
        assert result["status"] == "cancelled"


# ═══════════════════════════════════════════════════════════════════════════
# HITLStatus enum
# ═══════════════════════════════════════════════════════════════════════════


class TestHITLStatus:
    """Tests for the HITLStatus StrEnum."""

    def test_all_statuses_defined(self) -> None:
        """All expected statuses are available."""
        assert HITLStatus.PENDING == "pending"
        assert HITLStatus.RESPONDED == "responded"
        assert HITLStatus.TIMED_OUT == "timed_out"
        assert HITLStatus.CANCELLED == "cancelled"

    def test_string_comparison(self) -> None:
        """StrEnum values compare with strings."""
        assert HITLStatus.PENDING == "pending"
        assert HITLStatus.PENDING == "pending"
