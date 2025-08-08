import os
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

SPEC_VERSION = "0.2.5"  # card format/spec version (documented in README)

def _base_card(name: str, url: str, description: str) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        url=url,  # full base URL of this agent (mount point)
        preferredTransport="JSONRPC",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=False,
        ),
        skills=[
            AgentSkill(
                id="chat",
                name="Chat",
                description="Conversational chat over text.",
                tags=["chat"],
                examples=["hello", "what can you do?"],
                inputModes=["text"],
                outputModes=["text"],
            )
        ],
        # NOTE: The spec version is 0.2.5; some SDKs include this implicitly in the served card.
        # We keep it in README and ensure JSON-RPC paths align with 0.2.5 default routes.
    )

def build_echo_card(base_url: str) -> AgentCard:
    return _base_card(
        "EchoAgent",
        f"{base_url}/agents/echo",
        "Repeats your message and explains that it echoed it.",
    )

def build_math_card(base_url: str) -> AgentCard:
    return _base_card(
        "MathAgent",
        f"{base_url}/agents/math",
        "Finds numbers in your message and returns their sum with explanation.",
    )
