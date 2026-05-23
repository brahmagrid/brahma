# CLAUDE.md — Brahma

> Claude Code / OpenCode fallback. Full context in AGENTS.md.

## Project

Brahma — recursive, self-organizing agent runtime. 7 bootstrap tools, JIT-generates
everything else. REST API via FastAPI + uvicorn. Domain: brahmagrid.io.

## Setup & Run

```bash
uv sync
uv run brahma    # Starts on port 8420
```

## Key Files

- `brahma/agent.py` — Core agent loop
- `brahma/tools.py` — 7 bootstrap tools
- `brahma/server.py` — REST API
- `brahma/models.py` — Provider routing
- `brahma/bootstrap.py` — System prompt

## API (port 8420)

```
POST /run              → Run task on god agent
POST /spawn            → Create child agent
POST /agent/{id}/run   → Run task on child
GET  /agents           → List all agents
GET  /health           → Health check
GET  /docs             → Swagger
```

## Rules

- `uv` for Python (not pip)
- REST API only — no CLI
- Lean agent loop — minimal bootstrap philosophy
