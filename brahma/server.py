"""
Brahma — REST API server.

JSON-over-HTTP interface for the Brahma agent. Every agent (god or child)
exposes the same API. Agent delegation = HTTP calls between agents.

Endpoints:
  POST /run              Execute a task synchronously
  POST /spawn            Create a child agent
  POST /agent/{id}/run   Run task on a child agent
  GET  /agents           List all agents
  GET  /health           Health check
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from brahma.agent import Agent
from brahma.bootstrap import BOOTSTRAP_PROMPT
from brahma.hitl import HITLQueue
from brahma.tools import bootstrap_tools, set_hitl_queue

logger = logging.getLogger(__name__)

# ── Data Models ────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    """Request body for POST /run and POST /agent/{id}/run."""

    task: str
    model: str | None = None
    max_turns: int = 50


class RunResponse(BaseModel):
    """Response body for task execution results."""

    result: str
    turns: int
    tokens: dict[str, int]
    model: str


class SpawnRequest(BaseModel):
    """Request body for POST /spawn."""

    tools: list[str] | None = None
    model: str | None = None
    system_prompt: str | None = None


class SpawnResponse(BaseModel):
    """Response body for agent creation."""

    agent_id: str


class AgentsResponse(BaseModel):
    """Response body for GET /agents."""

    agents: list[dict]


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    active_agents: int


class HITLCreateRequest(BaseModel):
    """Request body for POST /hitl/request."""

    agent_id: str = "brahma-agent"
    task_summary: str = ""
    prompt: str
    request_type: str = "approval"
    choices: list[str] | None = None


class HITLRespondRequest(BaseModel):
    """Request body for POST /hitl/{id}/respond."""

    response: str


# ── Agent Registry ─────────────────────────────────────────────────────


@dataclass
class AgentEntry:
    """Internal container tracking a registered agent and its metadata."""

    agent: Agent
    model: str
    tools: list[str]
    created_at: str


class AgentRegistry:
    """Thread-safe registry of all agents in this process."""

    def __init__(self) -> None:
        """Initialize an empty registry with a thread lock."""
        self._lock = threading.Lock()
        self._agents: dict[str, AgentEntry] = {}

    def create(
        self,
        model: str,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Register a new agent and return its ID."""
        agent_id = f"brahma-{uuid.uuid4().hex[:8]}"
        tool_names = tools or list(bootstrap_tools()._tools.keys())

        # Build a tool registry with the requested subset
        full_registry = bootstrap_tools()
        child_registry = full_registry  # For now, inherit all. TODO: subset.

        agent = Agent(
            system_prompt=system_prompt or BOOTSTRAP_PROMPT,
            model=model,
            tools=child_registry,
        )

        with self._lock:
            self._agents[agent_id] = AgentEntry(
                agent=agent,
                model=model,
                tools=tool_names,
                created_at=_now(),
            )

        return agent_id

    def get(self, agent_id: str) -> AgentEntry:
        """Retrieve an agent by ID. Raises HTTPException(404) if not found."""
        with self._lock:
            entry = self._agents.get(agent_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        return entry

    def list_agents(self) -> list[dict]:
        """Return a summary of all registered agents."""
        with self._lock:
            return [
                {
                    "id": aid,
                    "model": entry.model,
                    "tools": entry.tools,
                    "created_at": entry.created_at,
                    "turns_run": entry.agent.turn_count,
                }
                for aid, entry in self._agents.items()
            ]

    def remove(self, agent_id: str) -> None:
        """Remove an agent from the registry (no-op if not found)."""
        with self._lock:
            self._agents.pop(agent_id, None)

    @property
    def count(self) -> int:
        """Number of currently registered agents."""
        with self._lock:
            return len(self._agents)


# ── App Factory ────────────────────────────────────────────────────────


def create_app(default_model: str = "deepseek-v4-pro") -> FastAPI:
    """
    Build the Brahma REST application.

    Args:
        default_model: The model used when no model is specified in requests.

    Returns:
        A configured FastAPI application instance.
    """
    app = FastAPI(
        title="Brahma — Bootstrap Agents",
        version="0.1.0",
        description="JSON-over-HTTP API for self-extending agent runtime.",
    )
    registry = AgentRegistry()
    hitl_queue = HITLQueue()
    set_hitl_queue(hitl_queue)  # Wire into tools module for hitl_request tool

    # ── Root agent (god agent) ──────────────────────────────────────
    god_id = registry.create(model=default_model)

    # ── Endpoints ───────────────────────────────────────────────────

    @app.post("/run", response_model=RunResponse)
    def run_task(req: RunRequest) -> RunResponse:
        """
        Execute a task synchronously on the god agent.

        Blocks until the task completes or max_turns is reached.
        """
        entry = registry.get(god_id)
        agent = entry.agent

        if req.model:
            agent.model = req.model

        result = agent.run(req.task)

        return RunResponse(
            result=result,
            turns=agent.turn_count,
            tokens=agent._token_usage,
            model=agent.model,
        )

    @app.post("/spawn", response_model=SpawnResponse)
    def spawn_agent(req: SpawnRequest) -> SpawnResponse:
        """
        Create a child agent with optional tool subset and custom prompt.

        Returns the agent_id for subsequent /agent/{id}/run calls.
        """
        agent_id = registry.create(
            model=req.model or default_model,
            tools=req.tools,
            system_prompt=req.system_prompt,
        )
        return SpawnResponse(agent_id=agent_id)

    @app.post("/agent/{agent_id}/run", response_model=RunResponse)
    def run_child_task(agent_id: str, req: RunRequest) -> RunResponse:
        """Execute a task on a specific child agent."""
        entry = registry.get(agent_id)
        agent = entry.agent

        if req.model:
            agent.model = req.model

        result = agent.run(req.task)

        return RunResponse(
            result=result,
            turns=agent.turn_count,
            tokens=agent._token_usage,
            model=agent.model,
        )

    @app.get("/agents", response_model=AgentsResponse)
    def list_agents_endpoint() -> AgentsResponse:
        """List all agents (god + children)."""
        return AgentsResponse(agents=registry.list_agents())

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check."""
        return HealthResponse(active_agents=registry.count)

    @app.get("/context")
    def context_budget() -> dict:
        """Return the god agent's context budget (used, max, percentage)."""
        entry = registry.get(god_id)
        return entry.agent.context_budget

    @app.get("/agent/{agent_id}/context")
    def child_context_budget(agent_id: str) -> dict:
        """Return a child agent's context budget."""
        entry = registry.get(agent_id)
        return entry.agent.context_budget

    # ── HITL Endpoints ──────────────────────────────────────────────

    @app.post("/hitl/request")
    def create_hitl_request(req: HITLCreateRequest) -> dict:
        """
        Create a HITL request. Used by agents when they need human input.

        Returns the request_id for polling.
        """
        request_id = hitl_queue.create(
            agent_id=req.agent_id,
            task_summary=req.task_summary,
            prompt=req.prompt,
            request_type=req.request_type,
            choices=req.choices,
        )
        return {"request_id": request_id, "status": "pending"}

    @app.get("/hitl/pending")
    def list_pending_hitl() -> dict:
        """List all pending HITL requests (for the client to poll)."""
        return {"requests": hitl_queue.list_pending()}

    @app.get("/hitl/{request_id}")
    def get_hitl_request(request_id: str) -> dict:
        """Get details of a specific HITL request."""
        req = hitl_queue.get(request_id)
        if not req:
            raise HTTPException(status_code=404, detail=f"HITL request '{request_id}' not found")
        return req.to_dict()

    @app.post("/hitl/{request_id}/respond")
    def respond_hitl(request_id: str, req: HITLRespondRequest) -> dict:
        """
        Respond to a HITL request.

        Called by the client when a human submits their response.
        """
        ok = hitl_queue.respond(request_id, req.response)
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"HITL request '{request_id}' not found or already resolved.",
            )
        return {"status": "resolved", "request_id": request_id}

    @app.post("/hitl/{request_id}/cancel")
    def cancel_hitl(request_id: str) -> dict:
        """Cancel a pending HITL request."""
        ok = hitl_queue.cancel(request_id)
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"HITL request '{request_id}' not found or already resolved.",
            )
        return {"status": "cancelled", "request_id": request_id}

    # Store registry reference for testing
    app.state.registry = registry
    app.state.hitl_queue = hitl_queue
    app.state.god_id = god_id

    return app


# ── Helpers ────────────────────────────────────────────────────────────


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


# ── Entry Point ────────────────────────────────────────────────────────


def main() -> None:
    """Start the Brahma REST server."""
    model = os.getenv("BRAHMA_MODEL", "deepseek-v4-pro")
    host = os.getenv("BRAHMA_HOST", "0.0.0.0")
    port = int(os.getenv("BRAHMA_PORT", "8420"))
    log_level = os.getenv("BRAHMA_LOG_LEVEL", "info").lower()

    # Configure Python logging so agent turn-level logs are visible.
    log_format = "%(asctime)s [%(levelname)-5s] %(name)s — %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        datefmt="%H:%M:%S",
    )

    # Also write logs to a file.
    log_file = os.getenv("BRAHMA_LOG_FILE", "logs/brahma.log")
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    app = create_app(default_model=model)

    logger.info("Brahma — Bootstrap Agents")
    logger.info("Model: %s", model)
    logger.info("Log level: %s", log_level)
    logger.info("Log file: %s", os.path.abspath(log_file))
    logger.info("Listening: http://%s:%s", host, port)
    logger.info("Docs:     http://%s:%s/docs", host, port)

    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
