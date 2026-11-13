"""Summarization middleware that also accounts for the tokens it spends.

The stock ``SummarizationMiddleware`` keeps only the summary text, so the tokens spent
summarizing never reach the ``token_usage`` channel. This subclass runs the parent's
unchanged flow while capturing the usage of the LLM calls it makes, and writes it into
``token_usage`` tagged ``"summarization"``.
"""

from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langgraph.runtime import Runtime
from typing_extensions import override

from .token_usage import _TokenUsageState, capture_llm_usage, merge_usage


class UsageTrackingSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware that records summarization token usage."""

    # Registers the token_usage channel even without MessagesHistoryMiddleware.
    state_schema = _TokenUsageState

    @staticmethod
    def _with_usage(state, result: dict[str, Any] | None, usage: dict) -> dict[str, Any] | None:
        if result is None or not usage:
            return result
        return {**result, "token_usage": merge_usage(state.get("token_usage"), usage)}

    @override
    def before_model(self, state, runtime: Runtime) -> dict[str, Any] | None:
        with capture_llm_usage("summarization") as usage:
            result = super().before_model(state, runtime)
        return self._with_usage(state, result, usage)

    @override
    async def abefore_model(self, state, runtime: Runtime) -> dict[str, Any] | None:
        with capture_llm_usage("summarization") as usage:
            result = await super().abefore_model(state, runtime)
        return self._with_usage(state, result, usage)
