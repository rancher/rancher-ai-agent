"""Summarization middleware that also accounts for the tokens it spends.

The stock ``SummarizationMiddleware`` invokes the model to produce a summary but
keeps only ``response.text`` — the summary is re-emitted as a HumanMessage with no
``usage_metadata``, so the tokens spent summarizing are invisible to the rest of
the token-accounting pipeline (which reads ``usage_metadata`` off AIMessages).

This subclass mirrors the parent's ``before_model`` / ``abefore_model`` flow but
captures the summarization response's usage in local scope (race-safe across
concurrent requests sharing the middleware instance) and writes it into the shared
``token_usage`` channel tagged ``"summarization"``.

NOTE: This couples to LangChain's ``SummarizationMiddleware`` internals
(``_ensure_message_ids``, ``_should_summarize``, ``_determine_cutoff_index``,
``_partition_messages``, ``_build_new_messages``, ``_trim_messages_for_summary``).
LangChain is pinned in pyproject.toml; revisit on upgrade.
"""

## TODO! try to find a better alternative so we don't have to couple to langchain internals. Maybe a new hook in langchain's SummarizationMiddleware that returns the response object instead of just the text, so we can read usage_metadata from it.

import logging
import uuid
from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import RemoveMessage
from langchain_core.messages.utils import get_buffer_string
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing_extensions import override

from .messages_history import _TokenUsageState, merge_usage, usage_entry_from_response


class UsageTrackingSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware that records summarization token usage."""

    # Ensures the token_usage channel is registered even if this middleware is used
    # without MessagesHistoryMiddleware.
    state_schema = _TokenUsageState

    def _summary_usage_update(self, response: Any) -> dict | None:
        """Build a ``token_usage`` update for a summarization model response."""
        entry = usage_entry_from_response(response, "summarization")
        if not entry:
            logging.warning(
                "Summarization fired but the summary response carried no token usage "
                "(usage_metadata and response_metadata both empty); summarization "
                "tokens will not be counted for this call."
            )
            return None
        msg_id = getattr(response, "id", None) or str(uuid.uuid4())
        logging.debug("Captured summarization token usage: %s", entry)
        return {msg_id: entry}

    def _create_summary_with_usage(self, messages_to_summarize: list) -> tuple[str, dict | None]:
        """Sync variant of ``_acreate_summary`` that also returns a usage update."""
        if not messages_to_summarize:
            return "No previous conversation history.", None

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize.", None

        formatted_messages = get_buffer_string(trimmed_messages, format="xml")

        try:
            response = self.model.invoke(
                self.summary_prompt.format(messages=formatted_messages).rstrip(),
                config={"metadata": {"lc_source": "summarization"}},
            )
            return response.text.strip(), self._summary_usage_update(response)
        except Exception as e:
            return f"Error generating summary: {e!s}", None

    async def _acreate_summary_with_usage(self, messages_to_summarize: list) -> tuple[str, dict | None]:
        """Async variant of ``_acreate_summary`` that also returns a usage update."""
        if not messages_to_summarize:
            return "No previous conversation history.", None

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize.", None

        formatted_messages = get_buffer_string(trimmed_messages, format="xml")

        try:
            response = await self.model.ainvoke(
                self.summary_prompt.format(messages=formatted_messages).rstrip(),
                config={"metadata": {"lc_source": "summarization"}},
            )
            return response.text.strip(), self._summary_usage_update(response)
        except Exception as e:
            return f"Error generating summary: {e!s}", None

    @override
    def before_model(self, state, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        summary, usage_update = self._create_summary_with_usage(messages_to_summarize)
        new_messages = self._build_new_messages(summary)

        result: dict[str, Any] = {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }
        if usage_update:
            result["token_usage"] = merge_usage(state.get("token_usage"), usage_update)
        return result

    @override
    async def abefore_model(self, state, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        summary, usage_update = await self._acreate_summary_with_usage(messages_to_summarize)
        new_messages = self._build_new_messages(summary)

        result: dict[str, Any] = {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }
        if usage_update:
            result["token_usage"] = merge_usage(state.get("token_usage"), usage_update)
        return result
