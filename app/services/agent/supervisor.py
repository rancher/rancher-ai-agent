
"""
Supervisor agent implementation that coordinates multiple child agents as tools.

This module provides a supervisor agent that wraps each child agent as a callable tool,
allowing the LLM to decide which agent(s) to invoke and coordinate their results
to handle complex, multi-step user requests.

"""

import json
import logging
import yaml

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import Any, CompiledStateGraph, Checkpointer
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import  HumanMessage, ToolMessage
import langgraph.types
from langgraph.types import Command
from langchain_core.callbacks.manager import dispatch_custom_event
from dataclasses import dataclass
from .loader import AgentConfig
from .system_prompts import SUPERVISOR_PROMPT
from .middleware import (
    INTERRUPT_CANCEL_MESSAGE,
    MessagesHistoryMiddleware,
    cancel_human_validation_middleware,
    inject_additional_kwargs_middleware,
    ui_tools_middleware
)


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


class _AgentCallCounter:
    """Tracks consecutive calls to the same child agent."""

    def __init__(self):
        self.last_agent: str | None = None
        self.count: int = 0

    def record(self, agent_name: str) -> int:
        if agent_name == self.last_agent:
            self.count += 1
        else:
            self.last_agent = agent_name
            self.count = 1
        return self.count


class SupervisorGraph:
    """
    Typed wrapper around the compiled supervisor agent graph.

    Provides properly typed attributes for supervisor-specific metadata
    (child_agents) while delegating all CompiledStateGraph
    methods to the underlying graph.

    Attributes:
        child_agents: Mapping of agent names to their compiled graphs for direct routing.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        child_agents: dict[str, CompiledStateGraph],
    ):
        self._graph = graph
        self.child_agents = child_agents

    def __getattr__(self, name):
        return getattr(self._graph, name)


# Config keys forwarded from supervisor to child so the child can access request context.
_FORWARDED_CONFIG_KEYS = ("request_id", "request_metadata", "user_id")

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
    call_counter = _AgentCallCounter()
    agent_tools = [_create_agent_tool(child, call_counter) for child in child_agents]
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
        middleware=[
            MessagesHistoryMiddleware(),
            cancel_human_validation_middleware(),
            inject_additional_kwargs_middleware(),
            ui_tools_middleware(llm),
            SummarizationMiddleware(model=llm, trigger=[("messages", 30), ("tokens", 30000)], keep=("messages", 15)),
        ],
    )

def _build_child_config(agent_name: str) -> RunnableConfig:
    """
    Build a LangGraph run-config for a child agent derived from the current supervisor config.

    - Creates a namespaced thread_id (``<parent>::child::<agent_name>``) so the child graph
      can checkpoint its own state independently from the supervisor.
    - Forwards a fixed set of configurable keys (request_id, user_id, …) from the
      supervisor so the child has access to request-level context.
    - Clears callbacks to prevent event leakage from child to supervisor.
    """
    parent_configurable = ensure_config().get("configurable", {})
    parent_thread_id = parent_configurable.get("thread_id", "")
    if not parent_thread_id:
        raise ValueError("thread_id is required in configurable but was not provided")

    child_configurable: dict = {
        "thread_id": f"{parent_thread_id}::child::{agent_name}",
        **{k: parent_configurable[k] for k in _FORWARDED_CONFIG_KEYS if k in parent_configurable},
    }
    return RunnableConfig(configurable=child_configurable, callbacks=[])


def _extract_last_message(result: dict) -> str:
    """Return the content of the last non-empty message in *result*, or a fallback string."""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            return msg.content
    return "No response from agent."


def _extract_tool_metadata(result: dict) -> dict:
    """Extract mcp_response, mcp_data, interrupt_info, and created_at from result messages.

    All fields are extracted only from messages appearing after the last HumanMessage.
    """
    mcp_response = None
    mcp_data = None
    interrupt_info = None
    created_at = None
    ui_tools = None

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, HumanMessage):
            break
        kwargs = getattr(msg, "additional_kwargs", {})
        if created_at is None:
            created_at = kwargs.get("created_at") or created_at
        if isinstance(msg, ToolMessage):
            if mcp_response is None:
                candidate = kwargs.get("mcp_response")
                if candidate:
                    mcp_response = candidate
            if mcp_data is None:
                data = kwargs.get("mcp_data")
                if data:
                    mcp_data = _convert_tool_message_to_context(data)
            if interrupt_info is None and "interrupt_message" in kwargs:
                interrupt_info = {k: kwargs[k] for k in ("interrupt_message", "confirmation") if k in kwargs}
            if ui_tools is None and "ui_tools" in kwargs:
                ui_tools = kwargs.get("ui_tools")

    return {
        "mcp_response": mcp_response,
        "mcp_data": mcp_data,
        "interrupt_info": interrupt_info,
        "created_at": created_at,
        "ui_tools": ui_tools,
    }

def _convert_tool_message_to_context(content: str) -> str:
    """
    Converts a tool message content to formatted context (YAML format).
    
    Args:
        content: The raw tool message content (usually JSON)
        
    Returns:
        Formatted content string with MCP result payloads label
    """
    try:
        
        parsed = json.loads(content)
        # Handle both single objects and arrays of objects
        if isinstance(parsed, list):
            # Convert array of objects to YAML with document separators
            yaml_parts = []
            for item in parsed:
                yaml_parts.append(yaml.dump(item, default_flow_style=False, sort_keys=False))
            content = "---\n".join(yaml_parts)
            logging.debug(f"Converted ToolMessage array content to YAML format ({len(parsed)} items)")
        elif isinstance(parsed, dict):
            # Convert single object to YAML
            content = yaml.dump(parsed, default_flow_style=False, sort_keys=False)
            logging.debug(f"Converted ToolMessage dict content to YAML format")
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, use original content
        pass
    
    return f"\n[MCP result payloads]: {content}"


def _create_agent_tool(child_agent: ChildAgent, call_counter: _AgentCallCounter) -> BaseTool:
    """
    Wrap a child agent's compiled graph as a LangChain tool.

    The tool accepts a ``query`` string and returns the last AI message content
    along with all MCP responses as an artifact.

    Interrupt / resume flow:
    - If the child graph has a pending interrupt (human-in-the-loop), the supervisor
      pauses via ``interrupt()`` to collect the user's decision, then resumes the child.
    - After any invocation, if the child raised a new interrupt it is re-triggered at
      the supervisor level so the client receives the confirmation prompt.
    """
    agent_name = child_agent.config.name
    compiled_graph = child_agent.agent

    async def _invoke(query: str) -> tuple[str, dict]:
        child_config = _build_child_config(agent_name)
        child_state = await compiled_graph.aget_state(config=child_config)
        interrupt_ui_tools: list[dict] = []

        if child_state and child_state.interrupts:
            interrupt_value = child_state.interrupts[0].value
            if isinstance(interrupt_value, dict):
                interrupt_ui_tools = interrupt_value.get("ui_tools", [])

            resume_value = langgraph.types.interrupt(interrupt_value)
            result = await compiled_graph.ainvoke(Command(resume=resume_value), config=child_config)
            # If the user declined, return the cancel message so the supervisor's
            # cancel_check_middleware ends the graph.
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "content") and msg.content == INTERRUPT_CANCEL_MESSAGE:
                    logging.debug(f"Child agent '{agent_name}' was cancelled by the user")
                    metadata = {**_extract_tool_metadata(result), "mcp_response": None, "mcp_data": None}
                    if not metadata.get("ui_tools") and interrupt_ui_tools:
                        metadata["ui_tools"] = interrupt_ui_tools
                    return INTERRUPT_CANCEL_MESSAGE, metadata
        else:
            child_config["tags"] = ["no-stream"]
            _dispatch_subagent_event("processing-subagent-start", agent_name, query)
            result = await compiled_graph.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config=child_config,
            )
            _dispatch_subagent_event("processing-subagent-end", agent_name, query)

        metadata = _extract_tool_metadata(result)
        if not metadata.get("ui_tools") and interrupt_ui_tools:
            metadata["ui_tools"] = interrupt_ui_tools

        # The child is called via ainvoke() (not as a subgraph), so a GraphInterrupt
        # raised inside it is suppressed and ainvoke() returns normally.  Re-trigger any
        # new interrupt at the supervisor level so the client receives the prompt.
        child_state = await compiled_graph.aget_state(config=child_config)
        if child_state and child_state.interrupts:
            logging.debug(f"Child agent '{agent_name}' raised a new interrupt — re-triggering at supervisor level")
            langgraph.types.interrupt(child_state.interrupts[0].value)

        # send recommendation event if the same agent is selected 3 times in a row
        agent_selected_count = call_counter.record(agent_name)
        if agent_selected_count >= 3:
            recommended_field = f', "recommended": "{agent_name}"'
            dispatch_custom_event(
                "subagent_choice_event",
                _build_agent_metadata(agent_name, "auto", recommended_field),
            )
            call_counter.count = 0

        return _extract_last_message(result), metadata

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=agent_name,
        description=child_agent.config.description or f"Specialized agent '{agent_name}'",
        response_format="content_and_artifact",
    )

def _dispatch_subagent_event(tag: str, name: str, query: str | None = None) -> None:
    """Dispatch a subagent lifecycle event with a valid JSON payload."""
    if query is not None:
        payload: dict = {"name": name}
        payload["query"] = query
        dispatch_custom_event("subagent_call", f"<{tag}>{json.dumps(payload)}</{tag}>")

def _build_agent_metadata(agent_name: str, selection_mode: str, extra_metadata: str = "") -> str:
    """Build a structured agent metadata string for custom events."""
    return (
        f'<agent-metadata>{{"agentName": "{agent_name}", '
        f'"selectionMode": "{selection_mode}"{extra_metadata}}}</agent-metadata>'
    )
