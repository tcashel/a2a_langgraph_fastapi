"""
Minimal Python SDK client that:
  1) Resolves cards for /agents/echo and /agents/math
  2) Runs sync and streaming tests for both
"""

import asyncio
import httpx

from a2a.client import ClientFactory, A2ACardResolver, ClientConfig
from a2a.types import MessageSendParams, MessageSendConfiguration
from a2a.client import create_text_message_object


BASE = "http://localhost:8000"
ECHO = f"{BASE}/agents/echo"
MATH = f"{BASE}/agents/math"


async def resolve_card(agent_base: str, httpx_client: httpx.AsyncClient):
    # The resolver fetches /.well-known/agent-card.json by default
    resolver = A2ACardResolver(base_url=agent_base, httpx_client=httpx_client)
    return await resolver.get_agent_card()  # -> AgentCard
    # SDK exposes resolver/client and send_message(_streaming) per docs. :contentReference[oaicite:2]{index=2}


async def run_sync_test(agent_url: str, text: str, httpx_client: httpx.AsyncClient):
    card = await resolve_card(agent_url, httpx_client)
    
    # Create client config and factory - disable streaming for sync test
    config = ClientConfig(streaming=False, httpx_client=httpx_client)
    factory = ClientFactory(config)
    client = factory.create(card)

    msg = create_text_message_object(content=text)
    
    print(f"\n=== SYNC @ {agent_url} ===")
    async for event in client.send_message(msg):
        # Handle the event - could be a Task, Update, or Message
        print(f"Event: {event}")


async def run_stream_test(agent_url: str, text: str, httpx_client: httpx.AsyncClient):
    card = await resolve_card(agent_url, httpx_client)
    
    # Create client config and factory - enable streaming for stream test
    config = ClientConfig(streaming=True, httpx_client=httpx_client)
    factory = ClientFactory(config)
    client = factory.create(card)

    msg = create_text_message_object(content=text)

    print(f"\n=== STREAMING @ {agent_url} ===")
    async for event in client.send_message(msg):
        # Each event is a Task, Update, or Message
        print(f"Event: {event}")


async def main():
    async with httpx.AsyncClient(follow_redirects=True) as httpx_client:
        await run_sync_test(ECHO, "Hello Echo — please repeat me.", httpx_client)
        await run_stream_test(ECHO, "Stream this slowly if you can.", httpx_client)

        await run_sync_test(MATH, "Sum 3.5 and 6 and 10, thanks.", httpx_client)
        await run_stream_test(MATH, "Please add 1, 2, 3, 4. And stream it.", httpx_client)


if __name__ == "__main__":
    asyncio.run(main())
