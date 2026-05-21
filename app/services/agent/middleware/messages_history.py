from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import AgentMiddleware, OmitFromSchema
from langchain.messages import AIMessage, ToolMessage
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import override

from ....constants import INTERRUPT_CANCEL_REPLY


def _select_history_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Return messages worth preserving in the full history.

    Includes:
    - HumanMessages and AIMessages with content (excluding summarization injections)
    - ToolMessages with interrupt/confirmation data (human-in-the-loop responses)
    """
    result = []
    for m in messages:
        if getattr(m, "additional_kwargs", {}).get("lc_source") == "summarization":
            continue
        if isinstance(m, (HumanMessage, AIMessage)) and m.content and m.content != INTERRUPT_CANCEL_REPLY:
            result.append(m)
        elif isinstance(m, ToolMessage) and "confirmation" in getattr(m, "additional_kwargs", {}):
            result.append(m)
        elif isinstance(m, ToolMessage) and "confirmation" in ((getattr(m, "artifact", None) or {}).get("interrupt_info") or {}):
            artifact = getattr(m, "artifact", None) or {}
            if artifact.get("created_at"):
                m.additional_kwargs["created_at"] = artifact["created_at"]
            interrupt_info = artifact["interrupt_info"]
            m.additional_kwargs["confirmation"] = interrupt_info["confirmation"]
            if "interrupt_message" in interrupt_info:
                m.additional_kwargs["interrupt_message"] = interrupt_info["interrupt_message"]
                result.append(m)
 
    return result


def _append_new_to_history(
    messages: list[AnyMessage],
    history: list[AnyMessage],
    update_existing: bool = False,
) -> list[AnyMessage] | None:
    """Append or update messages in history (compared by id).

    When update_existing is False (default), only new messages are appended.
    When update_existing is True, existing messages are replaced with the
    incoming version (capturing any mutations like ui_tools added after initial capture).

    Returns the updated history list, or None if there is nothing to change.
    The middleware state is replaced on every update, so we must carry the full
    history forward — otherwise old entries would be lost.
    """
    existing_ids = {getattr(m, "id", None): i for i, m in enumerate(history) if getattr(m, "id", None)}
    updated = list(history)
    changed = False
    for m in messages:
        msg_id = getattr(m, "id", None)
        if msg_id in existing_ids:
            if update_existing:
                updated[existing_ids[msg_id]] = m
                changed = True
        else:
            updated.append(m)
            changed = True
    return updated if changed else None


class _MessagesHistoryState(AgentState):
    """Extended state that keeps a full, unsummarized copy of all messages."""
    messages_history: NotRequired[Annotated[list[AnyMessage], add_messages, OmitFromSchema()]]


class MessagesHistoryMiddleware(AgentMiddleware):
    """Preserves the full message history in a separate state field.

    The SummarizationMiddleware removes old messages from the ``messages`` channel.
    This middleware copies messages into ``messages_history`` (which is never pruned)
    so that the complete conversation can be retrieved later.
    """

    state_schema = _MessagesHistoryState

    @override
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        candidates = _select_history_messages(state.get("messages", []))
        updated = _append_new_to_history(candidates, state.get("messages_history", []))
        return {"messages_history": updated} if updated is not None else None

    @override
    def after_agent(self, state, runtime) -> dict[str, Any] | None:
        """Final pass: capture any new ToolMessages and update existing messages
        with mutations (e.g. ui_tools) added by later after_agent hooks."""
        candidates = _select_history_messages(state.get("messages", []))
        updated = _append_new_to_history(candidates, state.get("messages_history", []), update_existing=True)
        return {"messages_history": updated} if updated is not None else None

    @override
    async def aafter_agent(self, state, runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
