from typing import Any

from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import SystemMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from ..system_prompts import IDENTITY_PREAMBLE


def identity_preamble_middleware():
    """Before-model middleware: inject IDENTITY_PREAMBLE only when child is called directly."""

    @before_model
    def inject_identity_preamble(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        config = get_config()
        if not config.get("configurable", {}).get("agent"):
            return None

        return {"messages": [SystemMessage(content=IDENTITY_PREAMBLE, id="identity_preamble")]}

    return inject_identity_preamble
