from typing import Any, AsyncIterator, Optional, Union

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import get_message_text, new_agent_text_message

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

    async def _stream_langgraph_messages(
        self,
        user_text: str,
    ) -> AsyncIterator[str]:
        """
        Yields assistant message chunks from LangGraph streaming.
        """
        inputs = {"messages": [("user", user_text)]}
        async for chunk in self.agent.astream(inputs, stream_mode="messages"):
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

        if blocking:
            # SYNC: single final message
            result = await self.agent.ainvoke({"messages": [("user", user_text)]})
            final_text = await self._final_text_from_result(result)
            event_queue.enqueue_event(new_agent_text_message(final_text))
            return

        # STREAMING: forward assistant chunks as they arrive
        async for piece in self._stream_langgraph_messages(user_text):
            event_queue.enqueue_event(new_agent_text_message(piece))
        # Returning ends the streaming task; DefaultRequestHandler will finalize.
