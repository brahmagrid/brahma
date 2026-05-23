"""
Tests for the Brahma REST API server.

Covers create_app, endpoint behavior, AgentRegistry, and data models.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from brahma.server import (
    AgentRegistry,
    HealthResponse,
    RunRequest,
    SpawnRequest,
    create_app,
)

# ═══════════════════════════════════════════════════════════════════════════
# create_app
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateApp:
    """Tests for the create_app factory."""

    @pytest.fixture
    def app(self) -> Generator[FastAPI]:
        """Create a fresh app with mocked Agent to avoid LLM calls."""
        with patch("brahma.server.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.turn_count = 0
            mock_agent._token_usage = {"input": 0, "output": 0, "total": 0}
            mock_agent.model = "deepseek:deepseek-chat"
            mock_agent.run.return_value = "mocked result"
            yield create_app()

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """TestClient wrapping the app."""
        return TestClient(app)

    def test_health_endpoint(self, client: TestClient) -> None:
        """GET /health returns status and active agent count."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["active_agents"] >= 1  # God agent created

    def test_root_not_found(self, client: TestClient) -> None:
        """Root path returns 404 (no route defined)."""
        response = client.get("/")
        assert response.status_code == 404

    def test_docs_available(self, client: TestClient) -> None:
        """Swagger docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client: TestClient) -> None:
        """OpenAPI JSON schema is generated."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Brahma — Bootstrap Agents"
        assert "/run" in schema["paths"]
        assert "/spawn" in schema["paths"]
        assert "/agents" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_app_state_has_registry(self, app: FastAPI) -> None:
        """App state stores registry and god_id references."""
        assert hasattr(app.state, "registry")
        assert hasattr(app.state, "god_id")
        assert isinstance(app.state.registry, AgentRegistry)


# ═══════════════════════════════════════════════════════════════════════════
# /run endpoint
# ═══════════════════════════════════════════════════════════════════════════


class TestRunEndpoint:
    """Tests for POST /run."""

    @pytest.fixture
    def client(self) -> Generator[TestClient]:
        """Client with mocked Agent."""
        with patch("brahma.server.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.turn_count = 3
            mock_agent._token_usage = {
                "input": 100,
                "output": 50,
                "total": 150,
            }
            mock_agent.model = "test:mock-model"
            mock_agent.run.return_value = "Task completed successfully."
            app = create_app()
            yield TestClient(app)

    def test_run_returns_result(self, client: TestClient) -> None:
        """POST /run executes a task and returns structured response."""
        response = client.post(
            "/run",
            json={"task": "Write a prime checker"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "Task completed successfully."
        assert data["turns"] == 3
        assert data["tokens"]["total"] == 150
        assert "model" in data

    def test_run_with_model_override(self, client: TestClient) -> None:
        """Model can be overridden in the request."""
        response = client.post(
            "/run",
            json={
                "task": "test",
                "model": "anthropic:claude-sonnet-4",
            },
        )
        assert response.status_code == 200

    def test_run_with_max_turns(self, client: TestClient) -> None:
        """max_turns can be specified."""
        response = client.post(
            "/run",
            json={"task": "test", "max_turns": 10},
        )
        assert response.status_code == 200

    def test_run_missing_task_returns_422(self, client: TestClient) -> None:
        """Missing required 'task' field returns validation error."""
        response = client.post("/run", json={})
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# /spawn endpoint
# ═══════════════════════════════════════════════════════════════════════════


class TestSpawnEndpoint:
    """Tests for POST /spawn."""

    @pytest.fixture
    def client(self) -> Generator[TestClient]:
        """Client with mocked Agent."""
        with patch("brahma.server.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.turn_count = 0
            mock_agent._token_usage = {"input": 0, "output": 0, "total": 0}
            mock_agent.model = "deepseek:deepseek-chat"
            mock_agent.run.return_value = ""
            app = create_app()
            yield TestClient(app)

    def test_spawn_with_defaults(self, client: TestClient) -> None:
        """POST /spawn with no body creates an agent."""
        response = client.post("/spawn", json={})
        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data
        assert data["agent_id"].startswith("brahma-")

    def test_spawn_with_tools(self, client: TestClient) -> None:
        """Tools can be specified when spawning."""
        response = client.post(
            "/spawn",
            json={"tools": ["read_file", "write_file"]},
        )
        assert response.status_code == 200
        assert "agent_id" in response.json()

    def test_spawn_with_model(self, client: TestClient) -> None:
        """Model can be specified when spawning."""
        response = client.post(
            "/spawn",
            json={"model": "openai:gpt-4o"},
        )
        assert response.status_code == 200

    def test_spawn_with_custom_prompt(self, client: TestClient) -> None:
        """Custom system prompt can be specified."""
        response = client.post(
            "/spawn",
            json={"system_prompt": "You are a math expert."},
        )
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# /agents endpoint
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentsEndpoint:
    """Tests for GET /agents."""

    @pytest.fixture
    def client(self) -> Generator[TestClient]:
        """Client with mocked Agent."""
        with patch("brahma.server.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.turn_count = 0
            mock_agent._token_usage = {"input": 0, "output": 0, "total": 0}
            mock_agent.model = "test:model"
            mock_agent.run.return_value = ""
            app = create_app()
            yield TestClient(app)

    def test_lists_all_agents(self, client: TestClient) -> None:
        """GET /agents returns all registered agents."""
        # God agent is created automatically
        response = client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) >= 1

    def test_includes_child_agents_after_spawn(
        self, client: TestClient
    ) -> None:
        """Spawned agents appear in the list."""
        client.post("/spawn", json={})
        response = client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) >= 2  # god + child


# ═══════════════════════════════════════════════════════════════════════════
# /agent/{id}/run endpoint
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentRunEndpoint:
    """Tests for POST /agent/{id}/run."""

    @pytest.fixture
    def client(self) -> Generator[TestClient]:
        """Client with mocked Agent."""
        with patch("brahma.server.Agent", autospec=True) as mock_agent_cls:
            mock_agent = mock_agent_cls.return_value
            mock_agent.turn_count = 1
            mock_agent._token_usage = {"input": 5, "output": 2, "total": 7}
            mock_agent.model = "test:model"
            mock_agent.run.return_value = "child result"
            app = create_app()
            yield TestClient(app)

    def test_run_on_child_agent(self, client: TestClient) -> None:
        """Can run a task on a spawned child agent."""
        spawn_resp = client.post("/spawn", json={})
        agent_id = spawn_resp.json()["agent_id"]

        run_resp = client.post(
            f"/agent/{agent_id}/run",
            json={"task": "Do something"},
        )
        assert run_resp.status_code == 200
        assert run_resp.json()["result"] == "child result"

    def test_run_on_nonexistent_agent(self, client: TestClient) -> None:
        """Running on a nonexistent agent returns 404."""
        run_resp = client.post(
            "/agent/nonexistent-123/run",
            json={"task": "test"},
        )
        assert run_resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# AgentRegistry
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentRegistry:
    """Tests for the AgentRegistry class directly."""

    @pytest.fixture
    def registry(self) -> Generator[AgentRegistry]:
        """Fresh registry with mocked Agent."""
        with patch("brahma.server.Agent", autospec=True) as mock_cls:
            mock_agent = mock_cls.return_value
            mock_agent.turn_count = 0
            mock_agent.model = "test:model"
            yield AgentRegistry()

    def test_create_returns_agent_id(self, registry: AgentRegistry) -> None:
        """create() returns a brahma- prefixed ID."""
        agent_id = registry.create(model="test:model")
        assert agent_id.startswith("brahma-")
        assert len(agent_id) == 15  # "brahma-" + 8 hex chars

    def test_unique_ids(self, registry: AgentRegistry) -> None:
        """Each agent gets a unique ID."""
        id1 = registry.create(model="a")
        id2 = registry.create(model="b")
        assert id1 != id2

    def test_get_returns_entry(self, registry: AgentRegistry) -> None:
        """get() returns the AgentEntry for a valid ID."""
        agent_id = registry.create(model="test:model")
        entry = registry.get(agent_id)
        assert entry.model == "test:model"
        assert entry.agent is not None
        assert isinstance(entry.tools, list)

    def test_get_nonexistent_raises_404(
        self, registry: AgentRegistry
    ) -> None:
        """get() on nonexistent ID raises HTTP 404."""
        with pytest.raises(HTTPException) as exc_info:
            registry.get("nonexistent-id")
        assert exc_info.value.status_code == 404

    def test_list_agents_returns_summaries(
        self, registry: AgentRegistry
    ) -> None:
        """list_agents() returns dict summaries."""
        registry.create(model="a")
        registry.create(model="b")
        agents = registry.list_agents()
        assert len(agents) == 2
        assert "id" in agents[0]
        assert "model" in agents[0]
        assert "tools" in agents[0]
        assert "created_at" in agents[0]

    def test_remove(self, registry: AgentRegistry) -> None:
        """remove() deletes an agent."""
        agent_id = registry.create(model="test:model")
        assert registry.count == 1
        registry.remove(agent_id)
        assert registry.count == 0

    def test_remove_nonexistent_noop(
        self, registry: AgentRegistry
    ) -> None:
        """remove() on nonexistent ID is a no-op."""
        registry.remove("no-such-id")
        # Should not raise

    def test_count_property(self, registry: AgentRegistry) -> None:
        """Count reflects number of registered agents."""
        assert registry.count == 0
        registry.create(model="a")
        assert registry.count == 1
        registry.create(model="b")
        assert registry.count == 2


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════


class TestServerDataModels:
    """Tests for Pydantic data models."""

    def test_run_request_serialization(self) -> None:
        """RunRequest serializes correctly."""
        req = RunRequest(task="hello", model="test:model", max_turns=10)
        assert req.model_dump() == {
            "task": "hello",
            "model": "test:model",
            "max_turns": 10,
        }

    def test_run_request_defaults(self) -> None:
        """RunRequest has sensible defaults."""
        req = RunRequest(task="test")
        assert req.model is None
        assert req.max_turns == 50

    def test_spawn_request_defaults(self) -> None:
        """SpawnRequest has all-None defaults for optional fields."""
        req = SpawnRequest()
        assert req.tools is None
        assert req.model is None
        assert req.system_prompt is None

    def test_health_response_defaults(self) -> None:
        """HealthResponse has default status."""
        hr = HealthResponse(active_agents=3)
        assert hr.status == "ok"
        assert hr.active_agents == 3
