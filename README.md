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
```

## Beautiful CLI Testing Interface

The smoke test now features a beautiful CLI interface with rich formatting and interactive options:

```bash
# Test all agents with all test types
uv run smoke_test.py

# Test specific agent with specific test
uv run smoke_test.py --test context --agent math

# Test basic communication only
uv run smoke_test.py --test basic --agent echo

# Test conversation history
uv run smoke_test.py --test history --agent math

# Test A2A Protocol conversation flow
uv run smoke_test.py --test context --agent echo
```

### **Available Test Types:**

- **`basic`**: Simple communication tests
- **`history`**: Conversation memory tests  
- **`context`**: A2A Protocol conversation flow tests
- **`all`**: Run all test types (default)

### **Available Agents:**

- **`echo`**: Echo Agent (repeats and responds)
- **`math`**: Math Agent (mathematical reasoning)
- **`all`**: Test both agents (default)

### **Features:**

- ✅ **Rich formatting** with colors and panels
- ✅ **Server status checking** with progress indicators
- ✅ **Conversation continuity** testing
- ✅ **A2A Protocol compliance** verification
- ✅ **Error handling** with graceful failures
- ✅ **Modular test selection** for focused testing

## A2A ID Mapping and Conversation Continuity

This implementation properly maps A2A Protocol IDs to LangGraph's conversation system for seamless multi-turn conversations.

### **A2A ID Hierarchy:**

| A2A Protocol | LangGraph Equivalent | Purpose | Persistence |
|--------------|---------------------|---------|-------------|
| `contextId` | `thread_id` | **Conversation continuity** | Spans entire conversation |
| `taskId` | `N/A` | **Task-specific state** | Individual task lifecycle |
| `messageId` | `message_id` | **Unique message identifier** | Single message |

### **Conversation Flow:**

1. **First Message**: Client sends message without IDs
   ```json
   {
     "message": {
       "role": "user",
       "parts": [{"kind": "text", "text": "My name is Bob"}],
       "messageId": "msg-123"
     }
   }
   ```

2. **Server Response**: Server generates and returns both IDs
   ```json
   {
     "contextId": "ctx-conversation-abc",
     "taskId": "task-123",
     "message": {...}
   }
   ```

3. **Subsequent Messages**: Client uses server-generated `contextId`
   ```json
   {
     "message": {
       "role": "user", 
       "parts": [{"kind": "text", "text": "What's my name?"}],
       "contextId": "ctx-conversation-abc",  // Server-generated
       "messageId": "msg-456"
     }
   }
   ```

### **Key Benefits:**

- ✅ **True conversation history** across multiple messages
- ✅ **Server-generated IDs** following A2A Protocol
- ✅ **Immutable tasks** - each message gets new `taskId`
- ✅ **Persistent conversations** - same `contextId` throughout
- ✅ **LangGraph integration** - uses `MemorySaver` for in-memory state

### **Testing Conversation History:**

Run the smoke test to see conversation continuity in action:

```bash
uv run smoke_test.py
```

This demonstrates how the agent remembers information shared earlier in the conversation using the same `contextId`.

## Endpoints
Platform index (both agents): GET /.well-known/agents.json

Echo Agent:

Card: GET /agents/echo/.well-known/agent-card.json

JSON-RPC: POST /agents/echo/a2a/v1/jsonrpc

Math Agent:

Card: GET /agents/math/.well-known/agent-card.json

JSON-RPC: POST /agents/math/a2a/v1/jsonrpc

## Example curls

### sync

```bash
curl -s http://localhost:8000/agents/echo/a2a/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":"1",
    "method":"message/send",
    "params":{
      "message":{
        "role":"user",
        "parts":[{"kind":"text","text":"Say hello to A2A!"}]
      },
      "configuration":{"blocking":true}
    }
  }' | jq
```

### streaming

```bash
# Server-Sent Events (A2A streaming). Use a tool like curl to watch SSE lines.
curl -N http://localhost:8000/agents/echo/a2a/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":"2",
    "method":"message/stream",
    "params":{
      "message":{
        "role":"user",
        "parts":[{"kind":"text","text":"Stream hello, please."}]
      },
      "configuration":{"blocking":false}
    }
  }'

```
