
"""
Supervisor agent implementation that coordinates multiple child agents as tools.

This module provides a supervisor agent that wraps each child agent as a callable tool,
allowing the LLM to decide which agent(s) to invoke and coordinate their results
to handle complex, multi-step user requests.

Unlike the parent agent (which routes a request to a single child), the supervisor can
call multiple agents in sequence and synthesize their outputs into a unified response.
"""

import logging

from collections.abc import Callable
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import Any, CompiledStateGraph, Checkpointer
from langchain.agents.middleware import wrap_tool_call, before_model, AgentState
from langchain.messages import AIMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
import langgraph.types
from langgraph.types import Command
from langchain_core.callbacks.manager import dispatch_custom_event
from dataclasses import dataclass
from .loader import AgentConfig
from .base import INTERRUPT_CANCEL_MESSAGE
from langchain.messages import AIMessage
from langgraph.runtime import Runtime


class ChildAgentCancelled(Exception):
    """Raised when a child agent's tool execution is cancelled by the user."""
    pass

@dataclass
class ChildAgent:
    """
    Represents a specialized child agent that can handle specific types of requests.

    Attributes:
        config: Agent configuration with name, description, and other metadata
        agent: The compiled LangGraph agent that handles the actual work
    """
    config: AgentConfig
    agent: CompiledStateGraph


SUPERVISOR_PROMPT = (
    "You are a supervisor agent that coordinates multiple specialized agents to handle "
    "complex user requests. Each agent is exposed as a tool you can call.\n\n"
    "INSTRUCTIONS:\n"
    "1. Analyze the user's request and determine which agent(s) are needed.\n"
    "2. Break down multi-step requests into individual agent calls.\n"
    "3. When a request spans multiple domains, invoke the relevant agents in sequence.\n"
    "4. Synthesize the results from all agent calls into a coherent final response.\n"
    "5. If a single agent suffices, call only that one — do not invoke agents unnecessarily.\n"
    "6. Never instruct the user to use kubectl, the Rancher UI, or any external tool directly.\n"
    "All Kubernetes and Rancher-related operations must be handled by the rancher agent.\n"
)


def _create_agent_tool(child_agent: ChildAgent) -> BaseTool:
    """
    Wrap a child agent's compiled graph as a LangChain tool.

    The returned tool accepts a ``query`` string, invokes the child agent's graph,
    and returns the last AI message content.

    When the child agent triggers a human-in-the-loop interrupt, the resulting
    ``GraphInterrupt`` propagates to the supervisor, which saves its own state and
    forwards the interrupt to the client.  On resume the supervisor re-enters its
    tool node and re-calls ``_invoke``.  At that point the child's checkpointed
    state already contains the pending interrupt, so ``_invoke`` detects it and
    resumes the child graph with the user's response value instead of starting a
    fresh invocation.

    Args:
        child_agent: The child agent to expose as a tool.

    Returns:
        A ``StructuredTool`` that delegates to the child agent.
    """
    agent_name = child_agent.config.name
    agent_description = child_agent.config.description or f"Specialized agent '{agent_name}'"
    compiled_graph = child_agent.agent

    async def _invoke(query: str) -> str:
        """Send *query* to the child agent and return its textual response."""
        # Derive a child-specific thread_id from the parent's thread_id so the
        # child graph can checkpoint its state (required for interrupt/resume).
        parent_config = ensure_config()
        parent_thread_id = parent_config.get("configurable", {}).get("thread_id", "")
        child_thread_id = f"{parent_thread_id}::{agent_name}" if parent_thread_id else agent_name

        # Forward configurable keys the child graph needs (e.g. request_id)
        # while suppressing callbacks to prevent event leakage.
        parent_configurable = parent_config.get("configurable", {})
        child_configurable = {"thread_id": child_thread_id}
        for key in ("request_id", "request_metadata", "user_id"):
            if key in parent_configurable:
                child_configurable[key] = parent_configurable[key]

        child_config = {
            "configurable": child_configurable,
            "callbacks": [],
        }

        # Check if the child has a pending interrupt (i.e. we are resuming after
        # a human-in-the-loop confirmation).  If so, consume the supervisor's
        # resume value via interrupt() and forward it to the child graph.
        child_state = await compiled_graph.aget_state(config=child_config)
        if child_state and child_state.interrupts:
            logging.debug(
                f"Child agent '{agent_name}' has pending interrupt, "
                "consuming supervisor resume value and forwarding to child"
            )
            # interrupt() is recognised by the supervisor's LangGraph runtime:
            # on the first execution path the GraphInterrupt from the nested
            # graph already propagated up, so the supervisor recorded it.  On
            # resume the runtime delivers the user's Command(resume=…) value
            # through this interrupt() call.
            resume_value = langgraph.types.interrupt(child_state.interrupts[0].value)
            result = await compiled_graph.ainvoke(
                Command(resume=resume_value),
                config=child_config,
            )

            # If the user declined, the child graph ended with a cancel
            # ToolMessage.  Stop immediately — nothing left to do.
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content == INTERRUPT_CANCEL_MESSAGE:
                    logging.debug(f"Child agent '{agent_name}' was cancelled by the user")
                    raise ChildAgentCancelled(agent_name)
        else:
            # Normal invocation.  If the child triggers interrupt(), the
            # resulting GraphInterrupt propagates to the supervisor, which
            # saves its own state and forwards the interrupt to the client.
            result = await compiled_graph.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config=child_config,
            )

        messages = result.get("messages", [])
        # Walk backwards to find the last AI message with content
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                return msg.content
        return "No response from agent."

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=agent_name,
        description=agent_description,
    )

@before_model(can_jump_to=["end"])
def human_in_the_loop_loop(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if state["messages"][-1].content == INTERRUPT_CANCEL_MESSAGE: # TODO check if ToolMessage? 
        return {
            "messages": [AIMessage("Previous tool cancel by the user.")],
            "jump_to": "end"
        }
    return None

@wrap_tool_call
async def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    dispatch_custom_event("subagent_call", f"Supervisor is calling agent '{request.tool_call['name']}' with: {request.tool_call['args']}\n",)
    try:
        result = await handler(request)
        return result
    except ChildAgentCancelled:
        # Create a proper ToolMessage so the supervisor's state stays clean,
        # then use Command to route directly to END — skipping the LLM call.
        tool_message = ToolMessage(
            content=INTERRUPT_CANCEL_MESSAGE,
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )
        return Command(goto="__end__", update={"messages": [tool_message]})
    except Exception as e:
        raise


def create_supervisor_agent(
    llm: BaseChatModel,
    child_agents: list[ChildAgent],
    checkpointer: Checkpointer,
) -> CompiledStateGraph:
    """
    Creates a supervisor agent that coordinates multiple child agents as tools.

    Each child agent (loaded from AIAgentConfig CRDs) is wrapped as a LangChain tool.
    The supervisor uses the LLM to analyze user requests, decide which agent tool(s)
    to call, and synthesize their outputs into a coherent response.

    Args:
        llm: The language model instance to use for the supervisor agent.
        child_agents: List of child agents read from AIAgentConfig CRDs.
        checkpointer: Checkpointer for persisting agent state.

    Returns:
        A compiled LangGraph StateGraph ready to be invoked.
    """
    agent_tools = [_create_agent_tool(child) for child in child_agents]
    logging.info(
        "Supervisor agent created with %d agent tool(s): %s",
        len(agent_tools),
        [t.name for t in agent_tools],
    )

    return create_agent(
        llm,
        tools=agent_tools,
        system_prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer,
        name="supervisor",
        middleware=[monitor_tool, human_in_the_loop_loop],
    )