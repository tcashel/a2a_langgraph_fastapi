from typing import Any, AsyncIterator, Optional, Union
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import get_message_text, new_agent_text_message
from a2a.types import TaskState, TaskStatusUpdateEvent

# LangGraph objects are Runnable graphs that expose ainvoke/astream
# We'll accept anything that implements `ainvoke()` and `astream(..., stream_mode="messages")`.


class LangGraphAgentExecutor(AgentExecutor):
    """
    Wraps a LangGraph `create_react_agent` graph and exposes it to A2A.

    Sync path:
      - await agent.ainvoke(...) and send a single final assistant message.

    Streaming path:
      - iterate `agent.astream(..., stream_mode="messages")` and forward each
        assistant chunk as an A2A assistant message (no artifacts).
      - completion is signaled by returning from `execute()`.
    """

    def __init__(self, agent_graph: Any) -> None:
        self.agent = agent_graph

    async def _final_text_from_result(self, result: Any) -> str:
        """
        LangGraph prebuilt agent returns a dict state with 'messages'.
        Grab the last assistant message's content.
        """
        try:
            messages = result.get("messages") or []
            if not messages:
                return "No response."
            last = messages[-1]
            # Last can be a BaseMessage or dict; both provide `.content`
            content = getattr(last, "content", None)
            if isinstance(content, list):
                # LangChain sometimes uses list-of-parts; join text portions
                parts = [p.get("text") if isinstance(p, dict) else str(p) for p in content]
                return "".join(parts)
            return content or str(last)
        except Exception:
            return "No response."
        
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        The default implementation just marks the task as cancelled.
        If you keep LangGraph's run_id you can also call graph.interrupt(run_id).
        """
        await event_queue.enqueue_event(TaskStatusUpdateEvent(status=TaskState.cancelled, final=True))

    async def _stream_langgraph_messages(
        self,
        user_text: str,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Yields assistant message chunks from LangGraph streaming.
        """
        # Use contextId for conversation continuity (thread_id), taskId for task state (checkpoint_id)
        thread_id = context_id if context_id else str(uuid.uuid4())
        checkpoint_id = task_id if task_id else str(uuid.uuid4())
        
        config = {
            "configurable": {
                "thread_id": thread_id,  # Use contextId for conversation continuity
                "checkpoint_ns": "a2a_conversation",
                "checkpoint_id": checkpoint_id  # Use taskId for task-specific state
            }
        }
        inputs = {"messages": [("user", user_text)]}
        
        async for chunk in self.agent.astream(inputs, config=config, stream_mode="messages"):
            # `chunk` is usually a BaseMessage with .content (string or list)
            content = getattr(chunk, "content", None)
            if isinstance(content, list):
                piece = "".join([p.get("text") if isinstance(p, dict) else str(p) for p in content])
            else:
                piece = content if content is not None else str(chunk)
            if piece:
                yield piece

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_text = get_message_text(context.message) or ""
        blocking = True
        try:
            cfg = getattr(context, "configuration", None)
            if cfg is not None and getattr(cfg, "blocking", True) is False:
                blocking = False
        except Exception:
            pass

        # Get the A2A context ID and task ID for conversation continuity
        context_id = None
        task_id = None
        
        # Extract contextId and taskId from the request context
        if hasattr(context, 'context_id') and context.context_id:
            context_id = context.context_id
        elif hasattr(context, 'message') and context.message and hasattr(context.message, 'contextId'):
            context_id = context.message.contextId
        
        if hasattr(context, 'task_id') and context.task_id:
            task_id = context.task_id
        elif hasattr(context, 'task') and context.task and hasattr(context.task, 'id'):
            task_id = context.task.id
        
        # Debug: Print ID information
        print(f"DEBUG: context.context_id = {getattr(context, 'context_id', None)}")
        print(f"DEBUG: context.task_id = {getattr(context, 'task_id', None)}")
        print(f"DEBUG: Using context_id = {context_id}, task_id = {task_id}")

        # Use contextId for conversation continuity (thread_id), taskId for task state (checkpoint_id)
        thread_id = context_id if context_id else str(uuid.uuid4())
        checkpoint_id = task_id if task_id else str(uuid.uuid4())

        if blocking:
            # SYNC: single final message
            config = {
                "configurable": {
                    "thread_id": thread_id,  # Use contextId for conversation continuity
                    "checkpoint_ns": "a2a_conversation",
                    "checkpoint_id": checkpoint_id  # Use taskId for task-specific state
                }
            }
            inputs = {"messages": [("user", user_text)]}
            result = await self.agent.ainvoke(inputs, config=config)
            final_text = await self._final_text_from_result(result)
            # Include both contextId and taskId in the response message
            await event_queue.enqueue_event(new_agent_text_message(final_text, context_id=context_id, task_id=task_id))
            return

        # STREAMING: forward assistant chunks as they arrive
        async for piece in self._stream_langgraph_messages(user_text, task_id, context_id):
            # Include both contextId and taskId in each streaming message
            await event_queue.enqueue_event(new_agent_text_message(piece, context_id=context_id, task_id=task_id))
        # Returning ends the streaming task; DefaultRequestHandler will finalize.

