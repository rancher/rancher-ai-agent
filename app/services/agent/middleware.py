"""
Shared middleware factories and constants used by both supervisor and child agents.
"""

from datetime import datetime
from typing import Any

from langchain.agents.middleware import AgentState, after_model, before_model
from langchain.messages import AIMessage, ToolMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

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
