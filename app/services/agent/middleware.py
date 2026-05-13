"""
Shared middleware factories and constants used by both supervisor and child agents.
"""

import json
import logging

from datetime import datetime
from typing import Any

from langchain.agents.middleware import AgentState, after_model, before_model, after_agent
from langchain.messages import AIMessage, ToolMessage
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from ..ui_tools.loader import load_ui_tools_from_configmap
from ..ui_tools.selector import create_ui_tools_selector, filter_tool
from .loader import AgentConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

INTERRUPT_CANCEL_MESSAGE = "tool execution cancelled by the user"


# ---------------------------------------------------------------------------
# Shared middleware factories
# ---------------------------------------------------------------------------


def create_inject_request_id_middleware():
    """After-model middleware: inject request_id and created_at into the last AIMessage."""

    @after_model
    def inject_request_id(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        config = get_config()
        request_id = config.get("configurable", {}).get("request_id")
        if not request_id or not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        last_message.additional_kwargs["request_id"] = request_id
        last_message.additional_kwargs["created_at"] = datetime.now().isoformat()

        return {"messages": [last_message]}

    return inject_request_id


def create_cancel_check_middleware():
    """Before-model middleware: skip LLM call if the last tool was cancelled."""

    @before_model(can_jump_to=["end"])
    def cancel_check(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if isinstance(last, ToolMessage) and last.content == INTERRUPT_CANCEL_MESSAGE:
            return {
                "messages": [AIMessage("Previous tool canceled by the user.")],
                "jump_to": "end",
            }
        return None

    return cancel_check


# ---------------------------------------------------------------------------
# UI tools middleware
# ---------------------------------------------------------------------------


def create_ui_tools_middleware(llm: BaseChatModel, only_when_direct: bool = False):
    """After-agent middleware: dispatch UI tools when agent produces a final answer (no tool calls).

    Args:
        llm: The language model used for UI tools selection.
        only_when_direct: When True, the middleware only executes if the agent is
            the direct target of the request (config["configurable"]["agent"] is set).
            This is used by child agents to avoid dispatching UI tools when invoked
            by the supervisor (the supervisor has its own ui_tools middleware).
            When False (default), the middleware always executes — used by the supervisor.
    """

    @after_agent
    def ui_tools_dispatch(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1] if state["messages"] else None
        if not isinstance(last_message, AIMessage):
            return None

        config = get_config()

        # When only_when_direct is set, skip if the child is called via supervisor
        if only_when_direct and not config.get("configurable", {}).get("agent"):
            return None

        ui_tools_list = _dispatch_ui_tools_event(llm, state, config)

        if ui_tools_list:
            request_id = config.get("configurable", {}).get("request_id")
            if request_id:
                last_message.additional_kwargs["ui_tools"] = ui_tools_list
                return {"messages": [last_message]}

        return None

    return ui_tools_dispatch


def _dispatch_ui_tools(tools: list[dict]) -> None:
    """Dispatch a ``ui_tools`` custom event with the given tools list."""
    try:
        ui_tools_json = json.dumps(tools)
        ui_tools_event = f"<ui-tools>{ui_tools_json}</ui-tools>"
        dispatch_custom_event("ui_tools", ui_tools_event)
        logging.debug(f"Dispatched {len(tools)} UI tool(s): {[t['toolName'] for t in tools]}")
    except Exception as e:
        logging.error(f"Error dispatching UI tools: {e}", exc_info=True)


def _dispatch_ui_tools_event(
    llm: BaseChatModel,
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

        system_prompt = ui_tools_config_data.config.system_prompt
        max_tools = ui_tools_config_data.config.max_tools

        selector = create_ui_tools_selector(llm, system_prompt=system_prompt, max_tools=max_tools)

        dispatch_custom_event("notify_processing", "<processing-ui-tools/>")

        context = _collect_context_until_human(state)

        mcp_response = _collect_all_mcp_responses(state)
        mcp_data = _find_last_mcp_data(state)

        ui_tools_list = selector.select_tools(
            context=context,
            mcp_response=mcp_response,
            mcp_data=mcp_data,
            available_tools=filtered_tools,
        )

        _dispatch_ui_tools(ui_tools_list)
        return ui_tools_list
    except Exception as e:
        logging.error(f"Error dispatching UI tools event: {e}", exc_info=True)
        return []


def _collect_all_mcp_responses(state: dict) -> str | None:
    """Collect all ``mcp_response`` values from ToolMessages since the last HumanMessage."""
    responses = []
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            mcp = getattr(msg, "additional_kwargs", {}).get("mcp_response")
            if mcp:
                responses.append(mcp)
    responses.reverse()
    return "\n".join(responses) if responses else None


def _find_last_mcp_data(state: dict) -> str | None:
    """Return the last ``mcp_data`` (full MCP server response) from ToolMessages since the last HumanMessage."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            data = getattr(msg, "additional_kwargs", {}).get("mcp_data")
            if data:
                return data
    return None


def _collect_context_until_human(state: dict) -> str:
    """Collect all messages from the end of the conversation back to (and including) the last HumanMessage."""
    parts = []
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            additional_kwargs = msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}
            if "request_metadata" in additional_kwargs:
                rm = additional_kwargs["request_metadata"]
                user_input = rm.get("user_input", "")
                context = rm.get("context", {})
                parts.append(f"[User Message]: {user_input} - ui context: {context}")
            else:
                parts.append(f"[User Message]: {getattr(msg, 'text', str(msg.content))}")
            break
        elif isinstance(msg, AIMessage):
            parts.append(f"[Assistant Message]: {getattr(msg, 'text', str(msg.content))}")
        elif isinstance(msg, ToolMessage):
            content = _extract_tool_text(msg.content)
            parts.append(f"[Tool Result ({msg.name})]: {content}")

    parts.reverse()
    return "\n".join(parts)


def _extract_tool_text(content: str) -> str:
    """Extract only text content from tool results, stripping reasoning and metadata."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            texts = []
            for item in parsed:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    texts.append(item["text"])
            if texts:
                return "\n".join(texts)
        elif isinstance(parsed, dict):
            if "text" in parsed:
                return parsed["text"]
            return json.dumps(parsed)
    except (json.JSONDecodeError, TypeError):
        pass
    return content
