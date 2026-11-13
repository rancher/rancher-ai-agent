"""Token-usage accounting shared by the agent middleware and the planner.

Every LLM call is recorded as an entry in the ``token_usage`` state channel, keyed by
the response message id so re-recording the same call (retries/resumes) is idempotent.
"""

import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated, Any, Iterator, NotRequired

from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AnyMessage
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook
from langgraph.config import get_config


def merge_usage(existing: dict | None, new_entries: dict | None) -> dict:
    """Merge token-usage entries keyed by message id.

    ``create_agent`` registers middleware-declared channels as ``LastValue`` (the
    ``Annotated`` reducer is ignored), so middleware write sites must read the existing
    ``token_usage`` and return the full merged dict.
    """
    return {**(existing or {}), **(new_entries or {})}


def _model_name(response: Any) -> str | None:
    meta = getattr(response, "response_metadata", None) or {}
    return meta.get("model_name") or meta.get("model") or meta.get("model_id")


def _requested_agent() -> str | None:
    """Agent the request was addressed to, when a child agent runs on the chat's own thread."""
    try:
        return get_config().get("configurable", {}).get("agent") or None
    except RuntimeError:
        return None


def _build_entry(
    source: str,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> dict:
    return {
        "source": source,
        "agent": _requested_agent(),
        # Read from env directly: importing services.llm here would be circular.
        "provider": os.environ.get("ACTIVE_LLM") or None,
        "model": model,
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "total_tokens": total_tokens or 0,
        "cache_read": cache_read or 0,
        "cache_creation": cache_creation or 0,
    }


def usage_entry_from_metadata(um: dict | None, source: str, model: str | None = None) -> dict | None:
    """Build a ``token_usage`` entry from a raw ``usage_metadata`` dict, or None if empty.

    ``source`` tags where the tokens were spent ("agent", "ui-tools", "summarization",
    "planner") so the read layer can bucket usage per source.
    """
    if not um:
        return None
    details = um.get("input_token_details") or {}
    return _build_entry(
        source,
        model,
        um.get("input_tokens", 0),
        um.get("output_tokens", 0),
        um.get("total_tokens", 0),
        details.get("cache_read", 0),
        details.get("cache_creation", 0),
    )


def usage_entry(msg: AnyMessage, source: str) -> dict | None:
    """Build a ``token_usage`` entry from a message's ``usage_metadata``."""
    return usage_entry_from_metadata(getattr(msg, "usage_metadata", None), source, _model_name(msg))


def usage_entry_from_response(response: Any, source: str) -> dict | None:
    """Build a ``token_usage`` entry from a model response, with fallbacks.

    Prefers the normalized ``usage_metadata``. When it is absent (some providers only
    populate it on streamed calls), falls back to the raw counts most providers put in
    ``response_metadata`` under ``token_usage`` / ``usage`` (OpenAI-style
    ``prompt_tokens``/``completion_tokens`` or Anthropic-style ``input_tokens``/``output_tokens``).
    """
    entry = usage_entry(response, source)
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
    # Anthropic-style cache counters, or OpenAI-style prompt_tokens_details.cached_tokens.
    cache_read = raw.get("cache_read_input_tokens") or (raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cache_creation = raw.get("cache_creation_input_tokens") or 0
    return _build_entry(
        source, _model_name(response), input_tokens, output_tokens, total_tokens, cache_read, cache_creation
    )


def usage_update_from_response(response: Any, source: str) -> dict | None:
    """Return a ``{message_id: entry}`` update for a model response, or None without usage."""
    entry = usage_entry_from_response(response, source) if response is not None else None
    if not entry:
        return None
    return {getattr(response, "id", None) or str(uuid.uuid4()): entry}


class _TokenUsageState(AgentState):
    """State channel accumulating per-model-call token usage, keyed by message id."""
    token_usage: NotRequired[Annotated[dict[str, dict], merge_usage, OmitFromSchema()]]


# --- Collecting usage of LLM calls made outside the agent's model node ---

_collected_usage: ContextVar[dict | None] = ContextVar("token_usage_collected", default=None)


@contextmanager
def collect_usage() -> Iterator[dict]:
    """Collect the entries passed to ``record_usage`` (in this context) into the yielded dict."""
    collected: dict = {}
    token = _collected_usage.set(collected)
    try:
        yield collected
    finally:
        _collected_usage.reset(token)


def record_usage(response: Any, source: str) -> None:
    """Record a model response's usage into the active ``collect_usage`` block, if any."""
    collected = _collected_usage.get()
    if collected is None:
        return
    if update := usage_update_from_response(response, source):
        collected.update(update)


class _UsageCallbackHandler(BaseCallbackHandler):
    """Records the usage of every chat model call that ends while it is active."""

    run_inline = True

    def __init__(self, collected: dict, source: str):
        self.collected = collected
        self.source = source

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                if update := usage_update_from_response(getattr(generation, "message", None), self.source):
                    self.collected.update(update)


_usage_callback: ContextVar[_UsageCallbackHandler | None] = ContextVar("token_usage_callback", default=None)
register_configure_hook(_usage_callback, inheritable=True)


@contextmanager
def capture_llm_usage(source: str) -> Iterator[dict]:
    """Capture the usage of every LLM call started in this context, without access to the responses.

    Used to account for calls made inside library code (e.g. LangChain's summarization)
    that only exposes the generated text.
    """
    collected: dict = {}
    token = _usage_callback.set(_UsageCallbackHandler(collected, source))
    try:
        yield collected
    finally:
        _usage_callback.reset(token)
