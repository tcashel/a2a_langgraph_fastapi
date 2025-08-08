"""
Minimal Python SDK client that:
  1) Resolves cards for /agents/echo and /agents/math
  2) Runs sync and streaming tests for both
"""

import asyncio
import httpx

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import MessageSendParams, MessageSendConfiguration
from a2a.client import create_text_message_object


BASE = "http://localhost:8000"
ECHO = f"{BASE}/agents/echo"
MATH = f"{BASE}/agents/math"


async def resolve_card(agent_base: str):
    # The resolver fetches /.well-known/agent-card.json by default
    resolver = A2ACardResolver(base_url=agent_base)
    return await resolver.get_agent_card()  # -> AgentCard
    # SDK exposes resolver/client and send_message(_streaming) per docs. :contentReference[oaicite:2]{index=2}


async def run_sync_test(agent_url: str, text: str):
    card = await resolve_card(agent_url)
    client = A2AClient(agent_card=card)

    msg = create_text_message_object(text)
    params = MessageSendParams(
        message=msg,
        configuration=MessageSendConfiguration(blocking=True),
    )
    result = await client.send_message(params)
    # The SDK packs a Task with history; final assistant text is in result / history.
    # For demo, print the top-level message-like result if present.
    print(f"\n=== SYNC @ {agent_url} ===")
    print(result)


async def run_stream_test(agent_url: str, text: str):
    card = await resolve_card(agent_url)
    client = A2AClient(agent_card=card)

    msg = create_text_message_object(text)
    params = MessageSendParams(
        message=msg,
        configuration=MessageSendConfiguration(blocking=False),
    )

    print(f"\n=== STREAMING @ {agent_url} ===")
    async for event in client.send_message_streaming(params):
        # Each event is a JSON-RPC chunk; print the text parts if present.
        print(event)


async def main():
    await run_sync_test(ECHO, "Hello Echo — please repeat me.")
    await run_stream_test(ECHO, "Stream this slowly if you can.")

    await run_sync_test(MATH, "Sum 3.5 and 6 and 10, thanks.")
    await run_stream_test(MATH, "Please add 1, 2, 3, 4. And stream it.")


if __name__ == "__main__":
    asyncio.run(main())
