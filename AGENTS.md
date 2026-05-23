# Brahma — Bootstrap Agent Runtime

> AGENTS.md — Agent harness context. Read this first.
> Project root: `/home/ahallur/Workspace/AgentSpace/brahma`
> Domain: brahmagrid.io | God agent: Brahma

## What This Is

Brahma is a recursive, self-organizing agent runtime. It starts with 7 bootstrap tools
and generates all other capabilities at runtime through a generate-validate-use-save loop.

Traditional agentic tools (Claude Code, Hermes, OpenCode) pre-load 40+ tools at startup.
Brahma loads 7 and JIT-generates everything else.

## Project Structure

```
brahma/                         ← YOU ARE HERE (cwd)
├── AGENTS.md                   ← This file
├── CLAUDE.md                   ← Claude Code / OpenCode fallback
├── README.md                   ← Human-facing overview
├── pyproject.toml              ← uv-managed, entry: brahma.server:main
├── .env.example
├── .gitignore
│
├── brahma/                     ← Python package
│   ├── __init__.py             # Public API: Agent, bootstrap_tools, create_app, BOOTSTRAP_PROMPT
│   ├── agent.py                # Core agent loop (~150 lines)
│   ├── tools.py                # 7 bootstrap tools (read_file, write_file, search_files, terminal, delegate_task, skill_manage, web_fetch)
│   ├── models.py               # Provider routing (Anthropic, OpenAI, DeepSeek, OpenRouter)
│   ├── bootstrap.py            # BOOTSTRAP_PROMPT — the meta-capability system prompt
│   └── server.py               # FastAPI REST server (5 endpoints, port 8420)
│
├── tests/
│   └── __init__.py
│
└── ../                           # Parent dir (AgentSpace/) has design docs:
    ├── bootstrap-agents-paper.md
    ├── bootstrap-agents-architecture.md
    ├── bootstrap-agents-implementation.md
    └── brahmagrid-project-tracker.md
```

## The Bootstrap Principle

When a task requires a capability the agent doesn't have:
1. **GENERATE** — Write code + tests + documentation via write_file
2. **VALIDATE** — Run tests. Use a different model to adversarially try to break it.
3. **USE** — Execute the original task with the new capability.
4. **SAVE** — Persist to `~/.brahma/skills/` via skill_manage for future reuse.

## REST API (port 8420)

```
POST /run              {"task": "..."}           → {"result": "...", "turns": N, "tokens": {...}, "model": "..."}
POST /spawn            {"tools": [...], "model": "..."}  → {"agent_id": "brahma-xxx"}
POST /agent/{id}/run   {"task": "..."}           → {"result": "...", ...}
GET  /agents                                     → {"agents": [{"id": "...", "model": "...", "tools": [...]}]}
GET  /health                                     → {"status": "ok", "active_agents": N}
GET  /docs                                       → Swagger UI
```

Agent delegation = HTTP calls between agents. Every agent (god or child) exposes the same API.

## Commands

```bash
# All commands run from this directory (brahma/)

uv sync              # Install dependencies
uv run brahma        # Start REST server on port 8420
uv run pytest        # Run tests
uv add <package>     # Add dependency

# Test endpoints
curl http://localhost:8420/health
curl -X POST http://localhost:8420/spawn -H "Content-Type: application/json" -d '{}'
curl http://localhost:8420/agents
```

## Environment

- **Python:** 3.14+ via uv (not pip)
- **OS:** Linux
- **API keys:** Set in `.env` (copy from .env.example)
- **Provider format:** `provider:model_name` (e.g., `deepseek:deepseek-chat`, `anthropic:claude-sonnet-4-20250514`)

## Type Discipline (NON-NEGOTIABLE)

1. **All classes must have type annotations** — every `__init__` parameter, every method
   signature, every return type.
2. **All public functions and classes must have docstrings** — at minimum a one-line
   summary; for non-trivial logic, include `Args:`/`Returns:`/`Raises:`.
3. **When a function signature changes, update the docstring immediately** — stale
   docs are worse than no docs.
4. This applies to ALL code in this repo: `brahma/`, `tests/`, scripts, everything.

## Import Rules

1. **All imports go at the top of the module** — no exceptions for stdlib or
   always-available packages. `import datetime` at the top, not inside a function.
2. **Deferred (in-function) imports are allowed only when:**
   - **Circular import avoidance** — when two modules would otherwise import each
     other at module level (see `_delegate_task` in tools.py).
   - **Heavy/optional dependencies** — third-party packages that are expensive to
     load and may not be used in every session (see `import httpx` in `_web_fetch`).
   - **Conditional/platform imports** — `if sys.platform == "win32": ...`
3. If you add a deferred import, add a comment explaining *why* it's deferred.

## Agent Rules

1. Use `uv` for all Python operations — never `pip`
2. The REST API is the only interface — no CLI
3. Keep the agent loop and tools lean — Brahma is minimal by design
4. Python imports: `from brahma import Agent` (the inner `brahma/brahma/` package)
5. Never edit `.venv/` — it's uv-managed
6. Design docs reference: `../bootstrap-agents-*.md` (parent directory)
