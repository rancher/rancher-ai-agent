from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langgraph.config import get_config

from ..system_prompts import PLANNER_SUBTASK_INSTRUCTIONS


def planner_subtask_middleware():
    """Dynamic-prompt middleware: append PLANNER_SUBTASK_INSTRUCTIONS only when run as a planner subtask."""

    @dynamic_prompt
    def planner_subtask_prompt(request: ModelRequest) -> str:
        base = request.system_prompt or ""
        if not get_config().get("configurable", {}).get("planner_subtask"):
            return base
        return base + PLANNER_SUBTASK_INSTRUCTIONS

    return planner_subtask_prompt
