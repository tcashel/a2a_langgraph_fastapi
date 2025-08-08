#!/usr/bin/env python3
"""
A2A x LangGraph x FastAPI Smoke Test

A beautiful CLI interface for testing A2A agents with conversation continuity.
"""

import asyncio
import httpx
import uuid
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.text import Text

from a2a.client import ClientFactory, A2ACardResolver, ClientConfig
from a2a.types import MessageSendParams, MessageSendConfiguration, Message, Role, Part, TextPart, AgentCard
from a2a.client import create_text_message_object
from a2a.utils import new_task, new_agent_text_message


# Initialize Typer app and Rich console
app = typer.Typer(help="A2A x LangGraph x FastAPI Smoke Test")
console = Console()

# Configuration
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

    if context_id:
        payload['message']['contextId'] = context_id

    if task_id:
        payload['message']['taskId'] = task_id
    
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
            else:
                # First message: No contextId or taskId (server will generate them)
                message_obj = create_text_message_object(content=message)

            # Send the message and collect responses
            responses = []
            async for response in client.send_message(message_obj):
                responses.append(response)

            # Handle the response - it's a Message object directly
            if responses and len(responses) > 0:
                message = responses[0]  # First response is a Message object
                
                # Extract server-generated contextId and taskId for conversation continuity
                if use_conversation:
                    # Server generates contextId and taskId in first response
                    if hasattr(message, 'contextId') and message.contextId:
                        self._conversation_context_ids[agent_url] = message.contextId
                    elif hasattr(message, 'context_id') and message.context_id:
                        self._conversation_context_ids[agent_url] = message.context_id
                    
                    if hasattr(message, 'taskId') and message.taskId:
                        self._conversation_task_ids[agent_url] = message.taskId
                    elif hasattr(message, 'task_id') and message.task_id:
                        self._conversation_task_ids[agent_url] = message.task_id
                
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


async def test_basic_communication(agent_url: str, agent_name: str):
    """Test basic communication with an agent."""
    client = A2ASimpleClient()
    
    console.print(Panel(f"[bold blue]Testing Basic Communication: {agent_name}[/bold blue]"))
    
    # Test basic message
    console.print(f"[yellow]User:[/yellow] Hello! What's your name?")
    response1 = await client.create_task(agent_url, "Hello! What's your name?")
    console.print(f"[green]{agent_name}:[/green] {response1}")
    
    # Test follow-up message
    console.print(f"\n[yellow]User:[/yellow] Can you help me with math?")
    response2 = await client.create_task(agent_url, "Can you help me with math?")
    console.print(f"[green]{agent_name}:[/green] {response2}")
    
    # Test math question
    console.print(f"\n[yellow]User:[/yellow] What's 15 + 25?")
    response3 = await client.create_task(agent_url, "What's 15 + 25?")
    console.print(f"[green]{agent_name}:[/green] {response3}")
    
    # Test another math question
    console.print(f"\n[yellow]User:[/yellow] What's 100 - 30?")
    response4 = await client.create_task(agent_url, "What's 100 - 30?")
    console.print(f"[green]{agent_name}:[/green] {response4}")


async def test_conversation_history(agent_url: str, agent_name: str):
    """Test conversation history by telling the agent information and asking it to remember later."""
    client = A2ASimpleClient()
    
    console.print(Panel(f"[bold blue]Testing Conversation History: {agent_name}[/bold blue]"))
    console.print("[dim]Testing if the agent can remember information shared earlier in the conversation.[/dim]")
    
    # First message - tell the agent your name
    console.print(f"\n[yellow]User:[/yellow] My name is Bob.")
    response1 = await client.create_task(agent_url, "My name is Bob.")
    console.print(f"[green]{agent_name}:[/green] {response1}")
    
    # Second message - ask the agent to remember your name
    console.print(f"\n[yellow]User:[/yellow] What's my name?")
    response2 = await client.create_task(agent_url, "What's my name?")
    console.print(f"[green]{agent_name}:[/green] {response2}")
    
    # Third message - tell the agent something else
    console.print(f"\n[yellow]User:[/yellow] I like pizza.")
    response3 = await client.create_task(agent_url, "I like pizza.")
    console.print(f"[green]{agent_name}:[/green] {response3}")
    
    # Fourth message - ask about both pieces of information
    console.print(f"\n[yellow]User:[/yellow] What's my name and what do I like?")
    response4 = await client.create_task(agent_url, "What's my name and what do I like?")
    console.print(f"[green]{agent_name}:[/green] {response4}")


async def test_conversation_with_context_id(agent_url: str, agent_name: str):
    """Test conversation history using server-generated context ID for continuity."""
    client = A2ASimpleClient()
    
    console.print(Panel(f"[bold blue]Testing Conversation with Context ID: {agent_name}[/bold blue]"))
    console.print("[dim]Following A2A Protocol: First message has no IDs, server generates them.[/dim]")
    
    # First message - tell the agent your name (server generates contextId and taskId)
    console.print(f"\n[yellow]User:[/yellow] My name is Bob.")
    response1 = await client.create_task(agent_url, "My name is Bob.", use_conversation=True)
    console.print(f"[green]{agent_name}:[/green] {response1}")
    console.print(f"[dim]Context ID: {client.get_current_context_id(agent_url)}[/dim]")
    console.print(f"[dim]Task ID: {client.get_current_task_id(agent_url)}[/dim]")
    
    # Second message - ask the agent to remember your name (uses server-generated contextId)
    console.print(f"\n[yellow]User:[/yellow] What's my name?")
    response2 = await client.create_task(agent_url, "What's my name?", use_conversation=True)
    console.print(f"[green]{agent_name}:[/green] {response2}")
    console.print(f"[dim]Using Context ID: {client.get_current_context_id(agent_url)}[/dim]")
    
    # Third message - tell the agent something else (uses server-generated contextId)
    console.print(f"\n[yellow]User:[/yellow] I like pizza.")
    response3 = await client.create_task(agent_url, "I like pizza.", use_conversation=True)
    console.print(f"[green]{agent_name}:[/green] {response3}")
    console.print(f"[dim]Using Context ID: {client.get_current_context_id(agent_url)}[/dim]")
    
    # Fourth message - ask about both pieces of information (uses server-generated contextId)
    console.print(f"\n[yellow]User:[/yellow] What's my name and what do I like?")
    response4 = await client.create_task(agent_url, "What's my name and what do I like?", use_conversation=True)
    console.print(f"[green]{agent_name}:[/green] {response4}")
    console.print(f"[dim]Using Context ID: {client.get_current_context_id(agent_url)}[/dim]")


async def check_server_status():
    """Check if the server is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BASE}/.well-known/agents.json")
            if response.status_code == 200:
                return True
    except Exception:
        pass
    return False


@app.command()
def main(
    test_type: str = typer.Option(
        "all",
        "--test",
        "-t",
        help="Type of test to run: basic, history, context, or all"
    ),
    agent: str = typer.Option(
        "all",
        "--agent",
        "-a", 
        help="Agent to test: echo, math, or all"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    )
):
    """A2A x LangGraph x FastAPI Smoke Test - Beautiful CLI Interface"""
    
    console.print(Panel.fit(
        "[bold blue]A2A x LangGraph x FastAPI[/bold blue]\n[dim]Smoke Test Suite[/dim]",
        border_style="blue"
    ))
    
    # Check server status
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Checking server status...", total=None)
        server_running = asyncio.run(check_server_status())
    
    if not server_running:
        console.print("[red]❌ Server is not running![/red]")
        console.print("Please start the server with: [bold]uv run serve[/bold]")
        raise typer.Exit(1)
    
    console.print("[green]✅ Server is running![/green]\n")
    
    # Define test functions
    tests = {
        "basic": test_basic_communication,
        "history": test_conversation_history, 
        "context": test_conversation_with_context_id
    }
    
    agents = {
        "echo": (ECHO, "Echo Agent"),
        "math": (MATH, "Math Agent")
    }
    
    # Determine which tests to run
    if test_type == "all":
        tests_to_run = list(tests.keys())
    else:
        tests_to_run = [test_type]
    
    # Determine which agents to test
    if agent == "all":
        agents_to_test = list(agents.keys())
    else:
        agents_to_test = [agent]
    
    # Run tests
    for test_name in tests_to_run:
        if test_name not in tests:
            console.print(f"[red]❌ Unknown test type: {test_name}[/red]")
            continue
            
        for agent_name in agents_to_test:
            if agent_name not in agents:
                console.print(f"[red]❌ Unknown agent: {agent_name}[/red]")
                continue
                
            agent_url, agent_display_name = agents[agent_name]
            
            try:
                asyncio.run(tests[test_name](agent_url, agent_display_name))
                console.print("\n" + "="*80 + "\n")
            except Exception as e:
                console.print(f"[red]❌ Error testing {agent_display_name}: {str(e)}[/red]")
    
    console.print(Panel("[bold green]✅ All tests completed![/bold green]", border_style="green"))


if __name__ == "__main__":
    app()
