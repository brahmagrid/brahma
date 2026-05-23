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

import os
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from brahma import Agent, bootstrap_tools, BOOTSTRAP_PROMPT


# ── Data Models ────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    task: str
    model: str | None = None
    max_turns: int = 50


class RunResponse(BaseModel):
    result: str
    turns: int
    tokens: dict[str, int]
    model: str


class SpawnRequest(BaseModel):
    tools: list[str] | None = None
    model: str | None = None
    system_prompt: str | None = None


class SpawnResponse(BaseModel):
    agent_id: str


class AgentsResponse(BaseModel):
    agents: list[dict]


class HealthResponse(BaseModel):
    status: str = "ok"
    active_agents: int


# ── Agent Registry ─────────────────────────────────────────────────────


@dataclass
class AgentEntry:
    agent: Agent
    model: str
    tools: list[str]
    created_at: str


class AgentRegistry:
    """Thread-safe registry of all agents in this process."""

    def __init__(self):
        self._lock = threading.Lock()
        self._agents: dict[str, AgentEntry] = {}

    def create(
        self,
        model: str,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> str:
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
        with self._lock:
            entry = self._agents.get(agent_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        return entry

    def list(self) -> list[dict]:
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
        with self._lock:
            self._agents.pop(agent_id, None)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._agents)


# ── App Factory ────────────────────────────────────────────────────────


def create_app(default_model: str = "deepseek:deepseek-chat") -> FastAPI:
    """
    Build the Brahma REST application.

    default_model: The model used when no model is specified in requests.
    """
    app = FastAPI(
        title="Brahma — Bootstrap Agents",
        version="0.1.0",
        description="JSON-over-HTTP API for self-extending agent runtime.",
    )
    registry = AgentRegistry()

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
        model = req.model or default_model

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
        """
        Execute a task on a specific child agent.
        """
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
    def list_agents() -> AgentsResponse:
        """List all agents (god + children)."""
        return AgentsResponse(agents=registry.list())

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check."""
        return HealthResponse(active_agents=registry.count)

    # Store registry reference for testing
    app.state.registry = registry
    app.state.god_id = god_id

    return app


# ── Helpers ────────────────────────────────────────────────────────────


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── Entry Point ────────────────────────────────────────────────────────


def main():
    """Start the Brahma REST server."""
    model = os.getenv("BRAHMA_MODEL", "deepseek:deepseek-chat")
    host = os.getenv("BRAHMA_HOST", "0.0.0.0")
    port = int(os.getenv("BRAHMA_PORT", "8420"))

    app = create_app(default_model=model)

    print(f"\n  Brahma — Bootstrap Agents")
    print(f"  Model: {model}")
    print(f"  Listening: http://{host}:{port}")
    print(f"  Docs:     http://{host}:{port}/docs\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
