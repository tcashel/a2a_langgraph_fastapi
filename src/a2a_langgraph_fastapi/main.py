import os
from fastapi import FastAPI

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import InMemoryQueueManager
from a2a.server.tasks import InMemoryPushNotificationConfigStore

from .cards import build_echo_card, build_math_card
from .agents import build_echo_agent, build_math_agent
from .executor import LangGraphAgentExecutor
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Create shared task store for conversation continuity across agents
SHARED_TASK_STORE = InMemoryTaskStore()
SHARED_QUEUE_MANAGER = InMemoryQueueManager()
SHARED_PUSH_CONFIG_STORE = InMemoryPushNotificationConfigStore()


def _mount_agent(app: FastAPI, mount_path: str, card_builder, agent_builder) -> str:
    """
    Build AgentCard, LangGraph agent, wrap in A2A request handler and mount.
    Returns the fully-qualified card URL for platform index.
    """
    card = card_builder(BASE_URL)
    agent_graph = agent_builder()
    executor = LangGraphAgentExecutor(agent_graph)

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=SHARED_TASK_STORE,  # Use shared task store
        queue_manager=SHARED_QUEUE_MANAGER,  # Use shared queue manager
        push_config_store=SHARED_PUSH_CONFIG_STORE,  # Use shared push config store
    )

    starlette_app = A2AFastAPIApplication(agent_card=card, http_handler=handler).build()
    app.mount(mount_path, starlette_app)

    return f"{card.url}/.well-known/agent-card.json"


def build_app() -> FastAPI:
    app = FastAPI(title="A2A Multi-Agent Platform")

    # Mount Echo and Math agents
    echo_card_url = _mount_agent(app, "/agents/echo", build_echo_card, build_echo_agent)
    math_card_url = _mount_agent(app, "/agents/math", build_math_card, build_math_agent)

    # Platform index: advertise both cards
    @app.get("/.well-known/agents.json")
    async def agents_index():
        return {"agents": [echo_card_url, math_card_url]}

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        build_app(),
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )
