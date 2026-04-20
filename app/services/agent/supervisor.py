"""
Supervisor agent implementation that coordinates multiple child agents as tools.

This module provides a supervisor agent that wraps each child agent as a callable tool,
allowing the LLM to decide which agent(s) to invoke and coordinate their results
to handle complex, multi-step user requests.

Unlike the parent agent (which routes a request to a single child), the supervisor can
call multiple agents in sequence and synthesize their outputs into a unified response.
"""

import logging

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph, Checkpointer

from .parent import ChildAgent

SUPERVISOR_PROMPT = (
    "You are a supervisor agent that coordinates multiple specialized agents to handle "
    "complex user requests. Each agent is exposed as a tool you can call.\n\n"
    "INSTRUCTIONS:\n"
    "1. Analyze the user's request and determine which agent(s) are needed.\n"
    "2. Break down multi-step requests into individual agent calls.\n"
    "3. When a request spans multiple domains, invoke the relevant agents in sequence.\n"
    "4. Synthesize the results from all agent calls into a coherent final response.\n"
    "5. If a single agent suffices, call only that one — do not invoke agents unnecessarily.\n"
)


def _create_agent_tool(child_agent: ChildAgent) -> BaseTool:
    """
    Wrap a child agent's compiled graph as a LangChain tool.

    The returned tool accepts a ``query`` string, invokes the child agent's graph,
    and returns the last AI message content.

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
        # Pass empty callbacks to prevent child agent events (custom events, LLM
        # streaming tokens, etc.) from propagating through the supervisor's callback
        # context and leaking to the websocket.
        result = await compiled_graph.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"callbacks": []},
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
    )
