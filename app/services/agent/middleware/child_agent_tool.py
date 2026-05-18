import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import cast

from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware.types import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from ._constants import INTERRUPT_CANCEL_MESSAGE, ChildAgentCancelled


def _create_child_agent_middleware() -> AgentMiddleware:
    """
    Build the wrap-tool-call middleware that intercepts every child-agent tool call
    made by the supervisor.  It has two responsibilities:

    1. **Artifact propagation** — child agents return ``(str, dict)`` with a rich
       artifact (mcp_responses, mcp_data, interrupt_info).  LangGraph's ToolNode
       stores the artifact on the ToolMessage but does *not* copy it into
       ``additional_kwargs``, so downstream code that reads ``additional_kwargs``
       would miss it.  This middleware does that copy after each successful call.

    2. **Cancellation short-circuit** — if the user rejected a human-in-the-loop
       prompt inside a child agent, ``_invoke`` raises ``ChildAgentCancelled``.
       The middleware catches it, constructs a well-formed ToolMessage, and returns
       ``Command(goto="__end__")`` to terminate the supervisor graph immediately
       without invoking the LLM node again.
    """

    @wrap_tool_call  # type: ignore[misc]
    async def handle_child_agent_tool_call(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Intercept a single child-agent tool call: propagate artifacts and handle cancellation."""
        name = request.tool_call["name"]
        try:
            logging.debug(f"Supervisor is invoking tool '{name}'")
            result = await handler(request)

            artifact = getattr(result, 'artifact', None) or {}
            if artifact and isinstance(result, ToolMessage):
                extra = {}
                mcp_responses = artifact.get("mcp_responses", [])
                if mcp_responses:
                    extra["mcp_response"] = "\n".join(mcp_responses)
                mcp_data = artifact.get("mcp_data")
                if mcp_data:
                    extra["mcp_data"] = mcp_data
                interrupt_info = artifact.get("interrupt_info")
                if interrupt_info:
                    extra.update(interrupt_info)
                    extra["created_at"] = datetime.now().isoformat()
                if extra:
                    result.additional_kwargs = {
                        **result.additional_kwargs,
                        **extra,
                    }

            return result
        except ChildAgentCancelled as exc:
            additional_kwargs = exc.interrupt_info or {}
            additional_kwargs["created_at"] = datetime.now().isoformat()
            tool_message = ToolMessage(
                content=INTERRUPT_CANCEL_MESSAGE,
                name=name,
                tool_call_id=request.tool_call["id"],
                additional_kwargs=additional_kwargs,
            )
            return Command(goto="__end__", update={"messages": [tool_message]})
        except Exception as e:
            raise

    return cast(AgentMiddleware, handle_child_agent_tool_call)
