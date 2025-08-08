"""
Minimal Python SDK client that:
  1) Resolves cards for /agents/echo and /agents/math
  2) Runs sync and streaming tests for both
  3) Demonstrates multi-turn conversations
"""

import asyncio
import httpx
import uuid

from a2a.client import ClientFactory, A2ACardResolver, ClientConfig
from a2a.types import MessageSendParams, MessageSendConfiguration, Message, Role, Part, TextPart, AgentCard
from a2a.client import create_text_message_object
from a2a.utils import new_task, new_agent_text_message


BASE = "http://localhost:8000"
ECHO = f"{BASE}/agents/echo"
MATH = f"{BASE}/agents/math"
AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, any]:
    """Helper function to create the payload for sending a task with proper A2A structure."""
    payload: dict[str, any] = {
        'message': {
            'role': 'user',
            'parts': [{'type': 'text', 'text': text}],
            'messageId': str(uuid.uuid4()),
        },
    }

    if task_id:
        payload['message']['taskId'] = task_id

    if context_id:
        payload['message']['contextId'] = context_id
    
    return payload


class A2ASimpleClient:
    """A2A Simple client to call A2A servers with conversation support."""

    def __init__(self, default_timeout: float = 240.0):
        self._agent_info_cache: dict[str, dict[str, any] | None] = {}  # Cache for agent metadata
        self.default_timeout = default_timeout
        self._conversation_task_ids: dict[str, str] = {}  # Cache task IDs per agent URL
        self._conversation_context_ids: dict[str, str] = {}  # Cache context IDs per agent URL

    async def create_task(self, agent_url: str, message: str, use_conversation: bool = True) -> str:
        """Send a message following the official A2A SDK pattern with proper conversation support."""
        # Configure httpx client with timeout
        timeout_config = httpx.Timeout(
            timeout=self.default_timeout,
            connect=10.0,
            read=self.default_timeout,
            write=10.0,
            pool=5.0,
        )

        async with httpx.AsyncClient(timeout=timeout_config, follow_redirects=True) as httpx_client:
            # Check if we have cached agent card data
            if (
                agent_url in self._agent_info_cache
                and self._agent_info_cache[agent_url] is not None
            ):
                agent_card_data = self._agent_info_cache[agent_url]
            else:
                # Fetch the agent card
                agent_card_response = await httpx_client.get(
                    f'{agent_url}{AGENT_CARD_WELL_KNOWN_PATH}'
                )
                agent_card_data = self._agent_info_cache[agent_url] = (
                    agent_card_response.json()
                )

            # Create AgentCard from data
            agent_card = AgentCard(**agent_card_data)

            # Create A2A client with the agent card
            config = ClientConfig(
                httpx_client=httpx_client,
                streaming=False,  # Use non-streaming mode
            )

            factory = ClientFactory(config)
            client = factory.create(agent_card)

            # Create the message object following A2A protocol
            if use_conversation and agent_url in self._conversation_context_ids:
                # Subsequent message: Use server-generated contextId for conversation continuity
                context_id = self._conversation_context_ids[agent_url]
                # Don't include taskId for conversation continuity - tasks are immutable
                # Only use taskId if we need to reference a specific task
                
                message_obj = Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.user,
                    parts=[Part(root=TextPart(kind="text", text=message))],
                    contextId=context_id,  # Server-generated contextId for conversation continuity
                    # taskId omitted - tasks are immutable once completed
                )
                print(f"DEBUG: Using server-generated context_id={context_id} for conversation continuity")
            else:
                # First message: No contextId or taskId (server will generate them)
                message_obj = create_text_message_object(content=message)
                print(f"DEBUG: First message - server will generate contextId and taskId")

            # Send the message and collect responses
            responses = []
            async for response in client.send_message(message_obj):
                responses.append(response)

            # Handle the response - it's a Message object directly
            if responses and len(responses) > 0:
                message = responses[0]  # First response is a Message object
                
                # Debug: Print the full response structure
                print(f"DEBUG: Response type: {type(message)}")
                print(f"DEBUG: Response: {message}")
                
                # Extract server-generated contextId and taskId for conversation continuity
                if use_conversation:
                    # Server generates contextId and taskId in first response
                    if hasattr(message, 'contextId') and message.contextId:
                        self._conversation_context_ids[agent_url] = message.contextId
                        print(f"DEBUG: Server generated contextId: {message.contextId}")
                    elif hasattr(message, 'context_id') and message.context_id:
                        self._conversation_context_ids[agent_url] = message.context_id
                        print(f"DEBUG: Server generated context_id: {message.context_id}")
                    
                    if hasattr(message, 'taskId') and message.taskId:
                        self._conversation_task_ids[agent_url] = message.taskId
                        print(f"DEBUG: Server generated taskId: {message.taskId}")
                    elif hasattr(message, 'task_id') and message.task_id:
                        self._conversation_task_ids[agent_url] = message.task_id
                        print(f"DEBUG: Server generated task_id: {message.task_id}")
                
                # Extract text from the message parts
                try:
                    if message.parts and len(message.parts) > 0:
                        part = message.parts[0]
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            return part.root.text
                    return str(message)
                except (AttributeError, IndexError):
                    return str(message)

            return 'No response received'

    def start_new_conversation(self, agent_url: str):
        """Start a new conversation by clearing the task ID and context ID for this agent."""
        if agent_url in self._conversation_task_ids:
            del self._conversation_task_ids[agent_url]
        if agent_url in self._conversation_context_ids:
            del self._conversation_context_ids[agent_url]

    def get_current_task_id(self, agent_url: str) -> str | None:
        """Get the current task ID for this agent's conversation."""
        return self._conversation_task_ids.get(agent_url)

    def get_current_context_id(self, agent_url: str) -> str | None:
        """Get the current context ID for this agent's conversation."""
        return self._conversation_context_ids.get(agent_url)


async def run_simple_client_test(agent_url: str):
    """Test using the simple A2A client pattern from the quickstart notebook"""
    client = A2ASimpleClient()
    
    print(f"\n=== SIMPLE A2A CLIENT TEST @ {agent_url} ===")
    print("Testing using the proper A2A client pattern from the quickstart notebook.")
    print()
    
    # Test basic message
    print("User: Hello! What's your name?")
    response1 = await client.create_task(agent_url, "Hello! What's your name?")
    print(f"Agent: {response1}")
    
    # Test follow-up message
    print("\nUser: Can you help me with math?")
    response2 = await client.create_task(agent_url, "Can you help me with math?")
    print(f"Agent: {response2}")
    
    # Test math question
    print("\nUser: What's 15 + 25?")
    response3 = await client.create_task(agent_url, "What's 15 + 25?")
    print(f"Agent: {response3}")
    
    # Test another math question
    print("\nUser: What's 100 - 30?")
    response4 = await client.create_task(agent_url, "What's 100 - 30?")
    print(f"Agent: {response4}")


async def run_conversation_history_simple_test(agent_url: str):
    """Test conversation history using the simple client pattern"""
    client = A2ASimpleClient()
    
    print(f"\n=== CONVERSATION HISTORY SIMPLE TEST @ {agent_url} ===")
    print("Testing conversation history using the simple client pattern.")
    print()
    
    # First message - tell the agent your name
    print("User: My name is Bob.")
    response1 = await client.create_task(agent_url, "My name is Bob.")
    print(f"Agent: {response1}")
    
    # Second message - ask the agent to remember your name
    print("\nUser: What's my name?")
    response2 = await client.create_task(agent_url, "What's my name?")
    print(f"Agent: {response2}")
    
    # Third message - tell the agent something else
    print("\nUser: I like pizza.")
    response3 = await client.create_task(agent_url, "I like pizza.")
    print(f"Agent: {response3}")
    
    # Fourth message - ask about both pieces of information
    print("\nUser: What's my name and what do I like?")
    response4 = await client.create_task(agent_url, "What's my name and what do I like?")
    print(f"Agent: {response4}")


async def run_conversation_with_task_id_test(agent_url: str):
    """Test conversation history using server-generated context ID for continuity"""
    client = A2ASimpleClient()
    
    print(f"\n=== CONVERSATION WITH SERVER-GENERATED CONTEXT ID TEST @ {agent_url} ===")
    print("Testing conversation history using server-generated context ID for continuity.")
    print("Following A2A Protocol: First message has no IDs, server generates them.")
    print()
    
    # First message - tell the agent your name (server generates contextId and taskId)
    print("User: My name is Bob.")
    response1 = await client.create_task(agent_url, "My name is Bob.", use_conversation=True)
    print(f"Agent: {response1}")
    print(f"Server-generated Context ID: {client.get_current_context_id(agent_url)}")
    print(f"Server-generated Task ID: {client.get_current_task_id(agent_url)}")
    
    # Second message - ask the agent to remember your name (uses server-generated contextId)
    print("\nUser: What's my name?")
    response2 = await client.create_task(agent_url, "What's my name?", use_conversation=True)
    print(f"Agent: {response2}")
    print(f"Using server-generated Context ID: {client.get_current_context_id(agent_url)}")
    print(f"Using server-generated Task ID: {client.get_current_task_id(agent_url)}")
    
    # Third message - tell the agent something else (uses server-generated contextId)
    print("\nUser: I like pizza.")
    response3 = await client.create_task(agent_url, "I like pizza.", use_conversation=True)
    print(f"Agent: {response3}")
    print(f"Using server-generated Context ID: {client.get_current_context_id(agent_url)}")
    print(f"Using server-generated Task ID: {client.get_current_task_id(agent_url)}")
    
    # Fourth message - ask about both pieces of information (uses server-generated contextId)
    print("\nUser: What's my name and what do I like?")
    response4 = await client.create_task(agent_url, "What's my name and what do I like?", use_conversation=True)
    print(f"Agent: {response4}")
    print(f"Using server-generated Context ID: {client.get_current_context_id(agent_url)}")
    print(f"Using server-generated Task ID: {client.get_current_task_id(agent_url)}")


async def main():
    # Test conversation with task ID (debug version)
    await run_conversation_with_task_id_test(MATH)


if __name__ == "__main__":
    asyncio.run(main())
