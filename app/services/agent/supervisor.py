
"""
Supervisor agent implementation that coordinates multiple child agents as tools.

This module provides a supervisor agent that wraps each child agent as a callable tool,
allowing the LLM to decide which agent(s) to invoke and coordinate their results
to handle complex, multi-step user requests.

"""

import json
import logging
import yaml

from datetime import datetime
from collections.abc import Callable
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import Any, CompiledStateGraph, Checkpointer
from langchain.agents.middleware import wrap_tool_call, SummarizationMiddleware
from langchain.messages import AIMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
import langgraph.types
from langgraph.types import Command
from langchain_core.callbacks.manager import dispatch_custom_event
from dataclasses import dataclass
from .loader import AgentConfig
from .system_prompts import SUPERVISOR_PROMPT
from .middleware import (
    INTERRUPT_CANCEL_MESSAGE,
    MessagesHistoryMiddleware,
    create_cancel_check_middleware,
    create_inject_request_id_middleware,
    create_ui_tools_middleware,
)


class ChildAgentCancelled(Exception):
    """Raised when a child agent's tool execution is cancelled by the user."""

    def __init__(self, agent_name: str, interrupt_info: dict | None = None):
        super().__init__(agent_name)
        self.agent_name = agent_name
        self.interrupt_info = interrupt_info

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
        middleware=[
            MessagesHistoryMiddleware(),
            _create_subagent_event_middleware(),
            create_cancel_check_middleware(),
            create_inject_request_id_middleware(),
            create_ui_tools_middleware(llm),
            SummarizationMiddleware(model=llm, trigger=[("messages", 4), ("tokens", 6000)], keep=("messages", 4)),
        ],
    )

def _build_child_config(agent_name: str) -> dict:
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
    return {"configurable": child_configurable, "callbacks": []}


def _extract_last_message(result: dict) -> str:
    """Return the content of the last non-empty message in *result*, or a fallback string."""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            return msg.content
    return "No response from agent."


def _extract_all_mcp_responses(result: dict) -> list[str]:
    """Return all ``mcp_response`` values found in the child's ToolMessages."""
    responses = []
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            mcp = getattr(msg, "additional_kwargs", {}).get("mcp_response")
            if mcp:
                responses.append(mcp)
    return responses


def _extract_last_mcp_data(result: dict) -> str | None:
    """Return the last ``mcp_data`` (full MCP server response) from the child's ToolMessages."""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, ToolMessage):
            data = getattr(msg, "additional_kwargs", {}).get("mcp_data")
            if data:
                return _convert_tool_message_to_context(data)
    return None


def _extract_last_interrupt_info(result: dict) -> dict | None:
    """Return interrupt metadata from the last ToolMessage that has it."""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, ToolMessage):
            kwargs = getattr(msg, "additional_kwargs", {})
            if "interrupt_message" in kwargs:
                return {
                    k: kwargs[k]
                    for k in ("interrupt_message", "confirmation")
                    if k in kwargs
                }
    return None

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


async def _resume_child_from_interrupt(
    compiled_graph: CompiledStateGraph,
    child_config: dict,
    child_state: Any,
    agent_name: str,
) -> dict:
    """
    Resume a child graph that has a pending human-in-the-loop interrupt.

    Calls ``langgraph.types.interrupt()`` at the supervisor level so the runtime
    delivers the user's ``Command(resume=…)`` value here, then forwards it to the
    child graph.  Raises ``ChildAgentCancelled`` if the user rejected the action.
    """
    logging.debug(f"Child agent '{agent_name}' has a pending interrupt — forwarding resume value from supervisor")
    resume_value = langgraph.types.interrupt(child_state.interrupts[0].value)
    result = await compiled_graph.ainvoke(Command(resume=resume_value), config=child_config)

    # If the user declined, the child ends with a INTERRUPT_CANCEL_MESSAGE ToolMessage.
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content == INTERRUPT_CANCEL_MESSAGE:
            logging.debug(f"Child agent '{agent_name}' was cancelled by the user")
            interrupt_info = _extract_last_interrupt_info(result)
            raise ChildAgentCancelled(agent_name, interrupt_info=interrupt_info)

    return result


def _create_agent_tool(child_agent: ChildAgent) -> BaseTool:
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

        if child_state and child_state.interrupts:
            result = await _resume_child_from_interrupt(compiled_graph, child_config, child_state, agent_name)
        else:
            child_config["tags"] = ["no-stream"]
            result = await compiled_graph.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config=child_config,
            )

        mcp_responses = _extract_all_mcp_responses(result)
        mcp_data = _extract_last_mcp_data(result)
        interrupt_info = _extract_last_interrupt_info(result)

        # The child is called via ainvoke() (not as a subgraph), so a GraphInterrupt
        # raised inside it is suppressed and ainvoke() returns normally.  Re-trigger any
        # new interrupt at the supervisor level so the client receives the prompt.
        child_state = await compiled_graph.aget_state(config=child_config)
        if child_state and child_state.interrupts:
            logging.debug(f"Child agent '{agent_name}' raised a new interrupt — re-triggering at supervisor level")
            langgraph.types.interrupt(child_state.interrupts[0].value)

        return _extract_last_message(result), {
            "mcp_responses": mcp_responses,
            "mcp_data": mcp_data,
            "interrupt_info": interrupt_info,
        }

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=agent_name,
        description=child_agent.config.description or f"Specialized agent '{agent_name}'",
        response_format="content_and_artifact",
    )

# =============================================================================
# Middleware
# =============================================================================


def _dispatch_subagent_event(tag: str, name: str, query: str | None = None) -> None:
    """Dispatch a subagent lifecycle event with a valid JSON payload."""
    if query is not None:
        payload: dict = {"name": name}
        payload["query"] = query
        dispatch_custom_event("subagent_call", f"<{tag}>{json.dumps(payload)}</{tag}>")


def _create_subagent_event_middleware():
    """Wrap-tool-call middleware: log and dispatch events for supervisor tool calls."""

    @wrap_tool_call
    async def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name = request.tool_call["name"]
        query = request.tool_call["args"].get("query")
        _dispatch_subagent_event("processing-subagent-start", name, query)
        try:
            logging.debug(f"Supervisor is invoking tool '{name}'")
            result = await handler(request)
            _dispatch_subagent_event("processing-subagent-end", name, query)

            # Propagate the child's MCP and interrupt data onto the supervisor ToolMessage.
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
            _dispatch_subagent_event("processing-subagent-end", name, query)
            # Create a proper ToolMessage so the supervisor's state stays clean,
            # then use Command to route directly to END — skipping the LLM call.
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

    return monitor_tool
