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
        last_message.additional_kwargs["request_metadata"] = config.get("configurable", {}).get("request_metadata", {})

        for msg in reversed(state["messages"][:-1]):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, ToolMessage):
                kwargs = getattr(msg, "additional_kwargs", {})
                last_message.additional_kwargs["interrupt_message"] = kwargs.get("interrupt_message", "") or msg.name
                
                # artifact is the second element of a (content, artifact) tuple returned by
                # tools with response_format="content_and_artifact". LangGraph stores it on
                # the ToolMessage but does NOT copy it into additional_kwargs, so supervisor
                # child-agent tools (which return rich metadata like mcp_responses and
                # ui_tools in the artifact dict) need to be read from here directly.
                artifact = getattr(msg, "artifact", None) or {}
                mcp_resp = kwargs.get("mcp_response", "")
                if not mcp_resp and artifact:
                    last_message.additional_kwargs["mcp_response"] = artifact.get("mcp_response", "")
                if mcp_resp:
                    last_message.additional_kwargs["mcp_response"] = mcp_resp
                break

        return {"messages": [last_message]}

    return inject_additional_kwargs
