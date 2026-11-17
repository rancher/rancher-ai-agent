from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langgraph.config import get_config


def aitask_prompt_middleware(replace_with: str | None = None, append: str = ""):
    """
    Dynamic-prompt middleware for unattended AITask runs.

    When the run config has ``configurable["aitask"]`` set, the system prompt is
    replaced with ``replace_with`` (if given) or extended with ``append``.
    Otherwise the prompt is left unchanged.
    """

    @dynamic_prompt
    def aitask_prompt(request: ModelRequest) -> str:
        base = request.system_prompt or ""
        if not get_config().get("configurable", {}).get("aitask"):
            return base
        if replace_with is not None:
            return replace_with
        return base + append

    return aitask_prompt
