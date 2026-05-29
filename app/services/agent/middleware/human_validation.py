import json
import logging
from collections.abc import Callable
from datetime import datetime

import langgraph.types
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.types import Command

from ._constants import INTERRUPT_CANCEL_MESSAGE
from ..loader import AgentConfig
from .ui_tools import _dispatch_ui_tools


def human_validation_middleware(
    planning_tools_by_name: dict[str, BaseTool],
    agent_config: AgentConfig,
):
    """``@wrap_tool_call`` middleware that gates tool execution behind human confirmation.

    For each tool call the middleware:

    1. Checks whether the tool is listed in ``agent_config.human_validation_tools``.
    2. If so, invokes the corresponding planning tool (``<tool_name>Plan``) to produce
       a preview of the intended change (diff / manifest), then calls
       ``langgraph.types.interrupt()`` to pause the graph and surface the prompt to
       the client.
    3. On resume, if the user answered anything other than ``"yes"`` the tool call is
       skipped and a ``ToolMessage`` with ``INTERRUPT_CANCEL_MESSAGE`` is returned so
       that ``cancel_human_validation_middleware`` can end the graph gracefully.
    4. On confirmation the real tool is executed and any associated UI tools (e.g.
       YAML diff viewers) are dispatched via ``_dispatch_ui_tools``.
    """

    @wrap_tool_call
    async def human_validation(
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

        human_validation_tools = getattr(agent_config, "human_validation_tools", [])
        interrupt_message = await _should_interrupt(human_validation_tools, tool_call, planning_tools_by_name)

        if interrupt_message:
            logging.debug(f"Confirmation interrupt triggered for tool '{tool_call['name']}'")
            ui_tools_list: list[dict] = []
            try:
                ui_tools_list = _build_interrupt_ui_tools(interrupt_message, state, config)
            except Exception as e:
                logging.debug(
                    f"Could not extract precomputed fields from interrupt message "
                    f"and dispatch UI tools: {e}"
                )
            additional_kwargs["ui_tools"] = ui_tools_list
            
            response = langgraph.types.interrupt({
                "message": interrupt_message,
                "ui_tools": ui_tools_list,
            })
            additional_kwargs["interrupt_message"] = interrupt_message
            additional_kwargs["created_at"] = datetime.now().isoformat()

            if response != "yes":
                additional_kwargs["confirmation"] = False
                return ToolMessage(
                    content=INTERRUPT_CANCEL_MESSAGE,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs=additional_kwargs,
                )

            additional_kwargs["confirmation"] = True

        try:
            logging.debug("calling tool")
            result = await handler(request)
            logging.debug("tool call finished")

            if isinstance(result, ToolMessage):
                processed_content, mcp_response, mcp_data = _process_tool_result(result.content, state)
                result.content = processed_content
                if mcp_response:
                    additional_kwargs["mcp_response"] = mcp_response
                if mcp_data:
                    additional_kwargs["mcp_data"] = mcp_data
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

    return human_validation


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

def _process_tool_result(tool_result: str | list, state: dict) -> tuple[str, str | None, str | None]:
    """Process the raw tool result, extracting UI context and doc links if present.

    Returns:
        ``(processed_result, mcp_response, mcp_data)`` where *mcp_response* is the
        uiContext wrapper (``None`` if absent) and *mcp_data* is the full raw JSON
        returned by the MCP server (``None`` if the result is not valid JSON).
    """
    mcp_response = None
    mcp_data = None
    try:
        if isinstance(tool_result, list) and tool_result:
            if isinstance(tool_result[0], dict) and "text" in tool_result[0]:
                tool_result = tool_result[0]["text"]

        json_result = json.loads(tool_result)

        if isinstance(json_result, dict) and "llm" in json_result:
            mcp_data = convert_to_string_if_needed(json_result["llm"])
        else:
            mcp_data = tool_result if isinstance(tool_result, str) else json.dumps(json_result)

        if "uiContext" in json_result:
            mcp_response = f"<mcp-response>{json.dumps(json_result['uiContext'])}</mcp-response>"
            dispatch_custom_event("ui_context", mcp_response)
        llm_result = json_result.get("llm", json_result) if isinstance(json_result, dict) else json_result
        return convert_to_string_if_needed(llm_result), mcp_response, mcp_data
    except (json.JSONDecodeError, TypeError):
        return tool_result, mcp_response, mcp_data


def convert_to_string_if_needed(var):
    """Convert dicts and lists to JSON strings; pass through everything else."""
    if isinstance(var, (dict, list)):
        return json.dumps(var)
    return var
