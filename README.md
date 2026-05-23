# Brahma — Bootstrap Agents

> **Recursive Self-Organization for Long-Horizon Autonomous Task Execution**
>
> Domain: brahmagrid.io

Brahma is a minimal, self-extending agent runtime with a JSON-over-HTTP interface.
It starts with 8 bootstrap tools and generates all other capabilities at runtime.

## Architecture

```
POST /run              → Execute task on god agent
POST /spawn            → Create child agent
POST /agent/{id}/run   → Execute task on child agent
GET  /agents           → List all agents
GET  /health           → Health check
```

Agent delegation = HTTP calls between agents. Every agent (god or child)
exposes the same REST interface.

## Quick Start

```bash
cd brahma
cp .env.example .env
# Edit .env with your API key

uv sync
uv run brahma    # Starts on http://0.0.0.0:8420
```

## API Examples

```bash
# Health check
curl http://localhost:8420/health

# Spawn a child agent
curl -X POST http://localhost:8420/spawn \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek:deepseek-chat"}'

# List all agents
curl http://localhost:8420/agents

# Run a task (blocks until complete)
curl -X POST http://localhost:8420/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Write a Python function to check if a number is prime. Include tests."}'

# Run task on a child agent
curl -X POST http://localhost:8420/agent/brahma-abc123/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Query the database..."}'
```

## Interactive Docs

Once running, open http://localhost:8420/docs for the Swagger UI.

## Supported Providers

| Provider | Env Var | Model string |
|----------|---------|-------------|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek:deepseek-chat` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic:claude-sonnet-4-20250514` |
| OpenAI | `OPENAI_API_KEY` | `openai:gpt-4o` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter:anthropic/claude-sonnet-4` |

## Environment Variables

| Var | Default | Description |
|-----|---------|-------------|
| `BRAHMA_MODEL` | `deepseek:deepseek-chat` | Default model |
| `BRAHMA_HOST` | `0.0.0.0` | Listen address |
| `BRAHMA_PORT` | `8420` | Listen port |

## Project Structure

```
brahma/
├── brahma/
│   ├── __init__.py      # Public API
│   ├── agent.py         # Agent loop (~150 lines)
│   ├── tools.py         # 8 bootstrap tools
│   ├── models.py        # Provider routing
│   ├── bootstrap.py     # System prompt
│   ├── hitl.py           # HITL queue system
│   └── server.py        # REST API server
├── tests/
├── pyproject.toml
└── README.md
```

## MVP Status

- [x] Core agent loop
- [x] 8 bootstrap tools
- [x] Multi-provider (Anthropic, OpenAI, DeepSeek, OpenRouter)
- [x] JSON REST API (FastAPI + uvicorn)
- [x] Agent spawning (/spawn + /agent/{id}/run)
- [x] HITL client (Electron + Vite + React)
- [ ] Adversarial cross-model validation
- [ ] Context budget monitoring
- [ ] Auto-splitting on context threshold
- [ ] Tiered memory (hot/warm/cool/cold)
- [ ] Layered OverlayFS isolation
- [ ] Recursive cost governance
