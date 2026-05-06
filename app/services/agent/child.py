"""
Agent builder for creating LangGraph agents with tool execution and human-in-the-loop validation.

Provides create_child_agent which constructs agents for use under a supervisor (multi-agent)
or standalone (single-agent) setups using the langchain create_agent factory with middleware.
Each agent has an LLM-driven reasoning loop with tool execution, human validation gates,
and automatic retry on malformed tool calls.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

import langgraph.types
import yaml
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    SummarizationMiddleware,
    after_model,
    wrap_tool_call,
)
from langchain.messages import AIMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.graph.state import Checkpointer, CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from ..ui_tools.loader import load_ui_tools_from_configmap
from ..ui_tools.selector import create_ui_tools_selector, filter_tool
from .loader import AgentConfig
from .middleware import (
    INTERRUPT_CANCEL_MESSAGE,
    create_cancel_check_middleware,
    create_inject_request_id_middleware,
)

INTERRUPT_PREVIOUS_TOOL_FAILED_MESSAGE = "tool execution cancelled because previous tool call failed"


def create_child_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    checkpointer: Checkpointer,
    agent_config: AgentConfig,
) -> CompiledStateGraph:
    """Create and compile a child agent graph using langchain create_agent with middleware.

    The agent uses the same create_agent factory as the supervisor, with middleware
    implementing: human-in-the-loop validation, metadata injection, UI tools dispatch,
    and tool execution error handling.
    """
    planning_tools = [t for t in tools if t.name.endswith("Plan")]
    execution_tools = [t for t in tools if not t.name.endswith("Plan")]
    planning_tools_by_name = {t.name: t for t in planning_tools}

    middleware = [
        _create_tool_execution_middleware(llm, planning_tools_by_name, agent_config),
        create_cancel_check_middleware(),
        create_inject_request_id_middleware(),
        _create_inject_selected_agent_middleware(agent_config),
        _create_ui_tools_middleware(llm, agent_config),
        SummarizationMiddleware(model=llm, trigger=[("messages", 30), ("tokens", 6000)]),
    ]

    return create_agent(
        llm,
        tools=execution_tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=middleware,
    )


# =============================================================================
# Middleware factories
# =============================================================================


def _create_inject_selected_agent_middleware(agent_config: AgentConfig):
    """After-model middleware: inject selected_agent into the last AIMessage."""

    @after_model
    def inject_selected_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        last_message.additional_kwargs["selected_agent"] = state.get("selected_agent", {})

        return {"messages": [last_message]}

    return inject_selected_agent


def _create_ui_tools_middleware(llm: BaseChatModel, agent_config: AgentConfig):
    """After-model middleware: dispatch UI tools when agent produces a final answer (no tool calls)."""

    @after_model
    def ui_tools_dispatch(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1] if state["messages"] else None
        if not isinstance(last_message, AIMessage):
            return None

        # Only run when the model is NOT calling tools (i.e. producing a final answer)
        if getattr(last_message, "tool_calls", None):
            return None

        config = get_config()
        ui_tools_list = _dispatch_ui_tools_event(llm, agent_config, state, config)

        if ui_tools_list:
            request_id = config.get("configurable", {}).get("request_id")
            if request_id:
                last_message.additional_kwargs["ui_tools"] = ui_tools_list
                return {"messages": [last_message]}

        return None

    return ui_tools_dispatch


def _create_tool_execution_middleware(
    llm: BaseChatModel,
    planning_tools_by_name: dict[str, BaseTool],
    agent_config: AgentConfig,
):
    """Wrap-tool-call middleware: human validation, MCP response processing, error handling."""

    @wrap_tool_call
    async def tool_execution(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        config = get_config()
        request_id = config.get("configurable", {}).get("request_id", "")
        state = request.state
        tool_call = request.tool_call

        additional_kwargs: dict = {
            "request_id": request_id,
            "selected_agent": state.get("selected_agent", {}),
        }

        # Human validation / interrupt
        human_validation_tools = getattr(agent_config, "human_validation_tools", [])
        interrupt_message = await _should_interrupt(human_validation_tools, tool_call, planning_tools_by_name)

        if interrupt_message:
            logging.info(f"Confirmation interrupt triggered for tool '{tool_call['name']}'")

            ui_tools_list: list[dict] = []
            try:
                ui_tools_list = _build_interrupt_ui_tools(interrupt_message, state, config)
            except Exception as e:
                logging.debug(
                    f"Could not extract precomputed fields from interrupt message "
                    f"and dispatch UI tools: {e}"
                )

            response = langgraph.types.interrupt(interrupt_message)
            if response != "yes":
                additional_kwargs["interrupt_message"] = interrupt_message
                additional_kwargs["confirmation"] = False
                additional_kwargs["ui_tools"] = ui_tools_list
                return ToolMessage(
                    content=INTERRUPT_CANCEL_MESSAGE,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs=additional_kwargs,
                )

            additional_kwargs["interrupt_message"] = interrupt_message
            additional_kwargs["confirmation"] = True
            additional_kwargs["ui_tools"] = ui_tools_list

            selected_agent = state.get("selected_agent", {})
            if selected_agent:
                dispatch_custom_event(
                    "subagent_choice_event",
                    _build_agent_metadata(selected_agent.get("name"), selected_agent.get("mode")),
                )

        # Execute the tool
        try:
            logging.debug("calling tool")
            result = await handler(request)
            logging.debug("tool call finished")

            # Process result for MCP responses
            if isinstance(result, ToolMessage):
                processed_content, mcp_response = _process_tool_result(result.content, state)
                result.content = processed_content
                if mcp_response:
                    additional_kwargs["mcp_response"] = mcp_response
                result.additional_kwargs = {**result.additional_kwargs, **additional_kwargs}

            return result
        except Exception as e:
            logging.error(f"unexpected error during tool call: {e}")
            return ToolMessage(
                content=f"unexpected error during tool call: {e}",
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                additional_kwargs=additional_kwargs,
            )

    return tool_execution


# =============================================================================
# Helper functions
# =============================================================================


async def _should_interrupt(
    human_validation_tools: list[str],
    tool_call: dict,
    planning_tools_by_name: dict[str, BaseTool],
) -> str:
    """Return a confirmation prompt if *tool_call* requires human validation, else ``''``."""
    for tool_name in human_validation_tools:
        if tool_name == tool_call["name"]:
            plan_tool_name = tool_call["name"] + "Plan"
            plan_tool = planning_tools_by_name.get(plan_tool_name)
            if plan_tool is None:
                raise ValueError(
                    f"planning tool '{plan_tool_name}' not found for tool '{tool_call['name']}'"
                )
            plan_response = await plan_tool.ainvoke(tool_call["args"])

            # Normalize list response: [{"type": "text", "text": "..."}]
            if isinstance(plan_response, list) and plan_response:
                if isinstance(plan_response[0], dict) and "text" in plan_response[0]:
                    plan_response = plan_response[0]["text"]

            try:
                safe_response = json.dumps(json.loads(plan_response))
            except (json.JSONDecodeError, TypeError):
                safe_response = json.dumps(plan_response)
            return f"<confirmation-response>{safe_response}</confirmation-response>"
    return ""


def _build_interrupt_ui_tools(
    interrupt_message: str,
    state: dict,
    config: dict,
) -> list[dict]:
    """Build preprocessed UI tools from the interrupt payload and dispatch them."""
    ui_tools_list: list[dict] = []

    request_metadata = config.get("configurable", {}).get("request_metadata", {})
    ui_tools_config = request_metadata.get("ui_tools", {})
    name = ui_tools_config.get("name", "")
    if not name:
        return ui_tools_list

    data = json.loads(
        interrupt_message.strip("<confirmation-response></confirmation-response>")
    )
    if isinstance(data, list) and len(data) > 0:
        data = data[0]

    resource = data.get("resource", {})
    tool_input: dict = {
        "resourceKind": resource.get("kind"),
        "resourceName": resource.get("name"),
        "resourceNamespace": resource.get("namespace"),
    }
    ui_tool_name = "show-yaml"

    if data.get("type") == "create":
        tool_input["yaml"] = data.get("payload", {})
    else:
        ui_tool_name = "show-yaml-diff"
        tool_input["original"] = data.get("payload", {}).get("original")
        tool_input["patched"] = data.get("payload", {}).get("patched")

    tool_input = {k: v for k, v in tool_input.items() if v is not None}

    ui_tools_list = [{"toolName": ui_tool_name, "input": tool_input}]
    _dispatch_ui_tools(ui_tools_list)
    return ui_tools_list



def _dispatch_ui_tools_event(
    llm: BaseChatModel,
    agent_config: AgentConfig,
    state: dict,
    config: dict,
) -> list[dict]:
    """Select and dispatch UI tools based on the current conversation state.

    Returns:
        List of selected UI tools, or empty list if dispatch was skipped.
    """
    try:
        request_metadata = config.get("configurable", {}).get("request_metadata", {})
        ui_tools_config = request_metadata.get("ui_tools", {})

        logging.debug(f"_dispatch_ui_tools_event: config={ui_tools_config}")

        name = ui_tools_config.get("name", "")
        tool_filters = ui_tools_config.get("tools", [])

        if not name:
            logging.debug("UI tools config name is missing, skipping ui tools dispatch")
            return []

        if not tool_filters:
            logging.debug("UI tools list is empty, skipping ui tools dispatch")
            return []

        ui_tools_config_data = load_ui_tools_from_configmap(name)

        if not ui_tools_config_data or not ui_tools_config_data.config:
            logging.debug(f"UI tools config {name} not found, skipping ui tools dispatch")
            return []

        if not ui_tools_config_data.config.enabled:
            logging.debug(f"UI tools config {name} are disabled, skipping ui tools dispatch")
            return []

        filtered_tools = [t for t in ui_tools_config_data.tools if filter_tool(t, tool_filters)]
        logging.debug(
            f"Filtered UI tools: {[t.name for t in filtered_tools]} "
            f"based on filters: {tool_filters}"
        )

        if not filtered_tools:
            logging.debug("No UI tools available after filtering, skipping ui tools dispatch")
            return []

        user_message, ai_message, mcp_response, mcp_data = _extract_context_for_tool_selection(
            state, config
        )

        system_prompt = ui_tools_config_data.config.system_prompt
        max_tools = ui_tools_config_data.config.max_tools

        selector = create_ui_tools_selector(llm, system_prompt=system_prompt, max_tools=max_tools)

        dispatch_custom_event("notify_processing", "<processing-ui-tools/>")

        ui_tools_list = selector.select_tools(
            agent_config=agent_config,
            context=user_message + ai_message,
            mcp_response=mcp_response + mcp_data,
            available_tools=filtered_tools,
        )

        _dispatch_ui_tools(ui_tools_list)
        return ui_tools_list
    except Exception as e:
        logging.error(f"Error dispatching UI tools event: {e}", exc_info=True)
        return []


def _extract_context_for_tool_selection(
    state: dict,
    config: dict,
) -> tuple[str, str, str, str]:
    """Extract context from the conversation for UI tool selection.

    Returns:
        ``(user_message, ai_message, mcp_response, mcp_data)``
    """
    request_id = config.get("configurable", {}).get("request_id", "")
    user_message = ""
    ai_message = ""
    mcp_data = ""

    for msg in reversed(state.get("messages", [])[-10:]):
        additional_kwargs = msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}

        if request_id != additional_kwargs.get("request_id", ""):
            break

        if user_message and ai_message and mcp_data:
            break

        if hasattr(msg, "text") and isinstance(msg.text, str):
            if isinstance(msg, HumanMessage) and not user_message:
                if "request_metadata" in additional_kwargs:
                    rm = additional_kwargs["request_metadata"]
                    user_input = rm.get("user_input", "")
                    context = rm.get("context", {})
                    user_message += f"\n[User Message]: {user_input} - ui context: {context}"
                if not user_message:
                    user_message += f"\n[User Message]: {msg.text}"
            elif isinstance(msg, AIMessage) and not ai_message:
                ai_message += f"\n[Assistant Message]: {msg.text}"
            elif isinstance(msg, ToolMessage) and not mcp_data:
                mcp_data = _convert_tool_message_to_context(msg.content)

    mcp_response = _extract_mcp_responses(state)
    return user_message, ai_message, mcp_response, mcp_data


def _convert_tool_message_to_context(content: str) -> str:
    """Convert tool message content to YAML-formatted context."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            yaml_parts = [yaml.dump(item, default_flow_style=False, sort_keys=False) for item in parsed]
            content = "---\n".join(yaml_parts)
        elif isinstance(parsed, dict):
            content = yaml.dump(parsed, default_flow_style=False, sort_keys=False)
    except (json.JSONDecodeError, TypeError):
        pass
    return f"\n[MCP result payloads]: {content}"


def _extract_mcp_responses(state: dict) -> str:
    """Extract MCP response resources from all messages."""
    mcp_response = ""
    for msg in reversed(state.get("messages", [])):
        additional_kwargs = msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}
        if "mcp_response" in additional_kwargs:
            mcp_response += f"\n{additional_kwargs['mcp_response'].strip('<mcp-response></mcp-response>')}"
    if mcp_response:
        mcp_response = "\n[MCP result resources]: " + mcp_response
    return mcp_response


def _dispatch_ui_tools(tools: list[dict]) -> None:
    """Dispatch a ``ui_tools`` custom event with the given tools list."""
    try:
        ui_tools_json = json.dumps(tools)
        ui_tools_event = f"<ui-tools>{ui_tools_json}</ui-tools>"
        dispatch_custom_event("ui_tools", ui_tools_event)
        logging.debug(f"Dispatched {len(tools)} UI tool(s): {[t['toolName'] for t in tools]}")
    except Exception as e:
        logging.error(f"Error dispatching UI tools: {e}", exc_info=True)


def _build_agent_metadata(agent_name: str, selection_mode: str, extra_metadata: str = "") -> str:
    """Build a structured agent metadata string for custom events."""
    return (
        f'<agent-metadata>{{"agentName": "{agent_name}", '
        f'"selectionMode": "{selection_mode}"{extra_metadata}}}</agent-metadata>'
    )


def _process_tool_result(tool_result: str | list, state: dict) -> tuple[str, str | None]:
    """Process the raw tool result, extracting UI context and doc links if present.

    Returns:
        ``(processed_result, mcp_response)`` where *mcp_response* is ``None`` if no uiContext.
    """
    mcp_response = None
    try:
        # Handle list format: [{"type": "text", "text": "..."}]
        if isinstance(tool_result, list) and tool_result:
            if isinstance(tool_result[0], dict) and "text" in tool_result[0]:
                tool_result = tool_result[0]["text"]

        json_result = json.loads(tool_result)

        if "uiContext" in json_result:
            mcp_response = f"<mcp-response>{json.dumps(json_result['uiContext'])}</mcp-response>"
            dispatch_custom_event("ui_context", mcp_response)
        llm_result = json_result.get("llm", json_result) if isinstance(json_result, dict) else json_result
        return convert_to_string_if_needed(llm_result), mcp_response
    except (json.JSONDecodeError, TypeError):
        return tool_result, mcp_response


def convert_to_string_if_needed(var):
    """Convert dicts and lists to JSON strings; pass through everything else."""
    if isinstance(var, (dict, list)):
        return json.dumps(var)
    return var
