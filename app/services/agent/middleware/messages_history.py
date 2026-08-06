from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import AgentMiddleware, OmitFromSchema
from langchain.messages import AIMessage, ToolMessage
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import override

from ....constants import INTERRUPT_CANCEL_REPLY


def merge_usage(existing: dict | None, new_entries: dict | None) -> dict:
    """Merge token-usage entries keyed by message id.

    ``create_agent`` registers middleware-declared channels as ``LastValue`` (the
    ``Annotated`` reducer is ignored), so every write site must read the existing
    ``token_usage`` and return the full merged dict — mirroring how
    MessagesHistoryMiddleware rebuilds ``messages_history``. Keying by message id
    keeps accumulation idempotent across retries/resumes.
    """
    return {**(existing or {}), **(new_entries or {})}


def usage_entry_from_metadata(um: dict | None, source: str) -> dict | None:
    """Build a ``token_usage`` entry from a raw ``usage_metadata`` dict.

    Returns None when there is no usage metadata. ``source`` tags where the tokens
    were spent ("agent", "ui-tools", "summarization") so the read layer can bucket
    usage per source.
    """
    if not um:
        return None
    details = um.get("input_token_details") or {}
    return {
        "source": source,
        "input_tokens": um.get("input_tokens", 0),
        "output_tokens": um.get("output_tokens", 0),
        "total_tokens": um.get("total_tokens", 0),
        "cache_read": details.get("cache_read", 0),
        "cache_creation": details.get("cache_creation", 0),
    }


def usage_entry(msg: AnyMessage, source: str) -> dict | None:
    """Build a ``token_usage`` entry from a message's ``usage_metadata``."""
    return usage_entry_from_metadata(getattr(msg, "usage_metadata", None), source)


def usage_entry_from_response(response: Any, source: str) -> dict | None:
    """Build a ``token_usage`` entry from a model response, with fallbacks.

    Prefers the normalized ``usage_metadata``. When it is absent (some providers
    only populate it on *streamed* calls, so a plain ``.ainvoke`` — as used by
    summarization — may lack it), falls back to the raw counts most providers put
    in ``response_metadata`` under ``token_usage`` / ``usage``
    (OpenAI-style ``prompt_tokens``/``completion_tokens`` or Anthropic-style
    ``input_tokens``/``output_tokens``).
    """
    entry = usage_entry_from_metadata(getattr(response, "usage_metadata", None), source)
    if entry:
        return entry

    meta = getattr(response, "response_metadata", None) or {}
    raw = meta.get("token_usage") or meta.get("usage") or {}
    if not raw:
        return None

    input_tokens = raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0
    output_tokens = raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0
    total_tokens = raw.get("total_tokens", input_tokens + output_tokens) or 0
    if not (input_tokens or output_tokens or total_tokens):
        return None
    return {
        "source": source,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read": 0,
        "cache_creation": 0,
    }


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


class _TokenUsageState(AgentState):
    """State channel accumulating per-model-call token usage, keyed by message id.

    Declared as a shared base so any middleware (agent turns, ui-tools,
    summarization) can write ``token_usage`` updates into the same channel. The
    channel is effectively LastValue under create_agent, so writers must merge with
    the existing value via ``merge_usage`` and return the full dict.
    """
    token_usage: NotRequired[Annotated[dict[str, dict], merge_usage, OmitFromSchema()]]


class _MessagesHistoryState(_TokenUsageState):
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

        result: dict[str, Any] = {}
        if updated is not None:
            result["messages_history"] = updated

        # Capture token usage from the model call that just produced the last
        # AIMessage. Recording here (before summarization can prune it) keeps the
        # count accurate even for tool-call turns that never enter messages_history.
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if (
            isinstance(last, AIMessage)
            and getattr(last, "id", None)
            and last.additional_kwargs.get("lc_source") != "summarization"
        ):
            entry = usage_entry(last, "agent")
            if entry:
                result["token_usage"] = merge_usage(state.get("token_usage"), {last.id: entry})

        return result or None

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
