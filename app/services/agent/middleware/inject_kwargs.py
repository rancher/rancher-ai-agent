from datetime import datetime
from typing import Any

from langchain.agents.middleware import AgentState, after_model
from langchain.messages import AIMessage, ToolMessage
from langchain_core.messages import HumanMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime


def inject_additional_kwargs_middleware():
    """After-model middleware: inject request_id, created_at, and tool metadata into the last AIMessage."""

    @after_model
    def inject_additional_kwargs(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        config = get_config()
        request_id = config.get("configurable", {}).get("request_id")
        if not request_id or not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        last_message.additional_kwargs["request_id"] = request_id
        last_message.additional_kwargs["created_at"] = datetime.now().isoformat()

        mcp_responses = []
        ui_tools = []
        for msg in reversed(state["messages"][:-1]):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, ToolMessage):
                mcp_resp = getattr(msg, "additional_kwargs", {}).get("mcp_response", "")
                if mcp_resp:
                    mcp_responses.append(mcp_resp)
                tool_ui_tools = getattr(msg, "additional_kwargs", {}).get("ui_tools", [])
                if tool_ui_tools:
                    ui_tools.extend(tool_ui_tools)
        if mcp_responses:
            mcp_responses.reverse()
            last_message.additional_kwargs["mcp_response"] = "\n".join(mcp_responses)
        if ui_tools:
            ui_tools.reverse()
            last_message.additional_kwargs["ui_tools"] = ui_tools

        return {"messages": [last_message]}

    return inject_additional_kwargs
