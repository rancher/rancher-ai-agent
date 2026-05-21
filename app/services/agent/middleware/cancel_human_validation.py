from typing import Any

from langchain.agents.middleware import AgentState, before_model
from langchain.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from ._constants import INTERRUPT_CANCEL_MESSAGE
from ....constants import INTERRUPT_CANCEL_REPLY


def cancel_human_validation_middleware():
    """Before-model middleware: skip LLM call if the last tool was cancelled."""

    @before_model(can_jump_to=["end"])
    def cancel_human_validation(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if isinstance(last, ToolMessage) and last.content == INTERRUPT_CANCEL_MESSAGE:
            return {
                "messages": [AIMessage(INTERRUPT_CANCEL_REPLY)],
                "jump_to": "end",
            }
        return None

    return cancel_human_validation
