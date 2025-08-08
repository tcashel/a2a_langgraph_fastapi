# A2A x LangGraph x FastAPI — Mini Repo (Echo + Math)

Two conversational agents, mounted as A2A JSON-RPC services with **sync** and **true streaming** replies, plus a Python SDK smoke test.

- Agents: `EchoAgent` and `MathAgent` (LangGraph `create_react_agent`)
- Transport: **A2A JSON-RPC** (0.2.5 card format), served via FastAPI/Starlette
- Streaming: forwards **assistant message chunks** (not artifacts)
- OpenAI model via `OPENAI_API_KEY` (e.g., `gpt-4o-mini`)

## Quickstart

```bash
# 1) Python 3.10+ and uv installed
uv venv
uv pip install -e .

# 2) Export your OpenAI key
export OPENAI_API_KEY=sk-...

# 3) Run the server
uv run serve
# Server at http://localhost:8000

## Endpoints
Platform index (both agents): GET /.well-known/agents.json

Echo Agent:

Card: GET /agents/echo/.well-known/agent-card.json

JSON-RPC: POST /agents/echo/a2a/v1/jsonrpc

Math Agent:

Card: GET /agents/math/.well-known/agent-card.json

JSON-RPC: POST /agents/math/a2a/v1/jsonrpc
