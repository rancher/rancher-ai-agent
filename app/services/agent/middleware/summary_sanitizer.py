"""Sanitization helpers for the summarization model.

Bedrock's Converse API rejects any request whose messages contain ``toolUse`` /
``toolResult`` content blocks unless a ``toolConfig`` is also supplied. The
summarization step never needs tools, so instead of binding a dummy tool
(``llm.bind_tools(...)``) we render past tool calls and results as plain text
before the summary model ever sees them.

The wrapping is done with a ``RunnableLambda`` piped into the LLM, so we do not
depend on any private method of ``SummarizationMiddleware``. The lambda handles
both possible inputs the middleware may pass to the model:

* a ``list`` of message objects (older langchain: ``model.invoke(messages)``)
* an already-formatted ``str`` prompt (current langchain, which flattens the
  slice via ``get_buffer_string`` before calling the model)

Only the message-list case can carry tool blocks, so that is the only case we
sanitize; strings are passed through untouched.
"""

import json
from functools import partial

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable, RunnableLambda


def _text_of(content) -> str:
    """Flatten message content (str or list of blocks) to plain text.

    ``tool_use`` / ``tool_result`` blocks (Bedrock Converse style) are dropped
    here; callers render them separately as text.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


def sanitize_for_summary(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Return a NEW message list with tool calls/results rendered as plain text.

    Never mutates the inputs and must never be applied to the live ``messages``
    channel: the agent's own next model call still needs valid, paired
    ``tool_use`` / ``tool_result`` blocks. This is only for the copy handed to
    the summarization model.
    """
    out: list[AnyMessage] = []
    for m in messages:
        if isinstance(m, AIMessage):
            lines: list[str] = []
            text = _text_of(m.content)
            if text:
                lines.append(text)
            for tc in (m.tool_calls or []):
                args = json.dumps(tc.get("args", {}), default=str, ensure_ascii=False)
                lines.append(f"[called tool `{tc.get('name', 'tool')}` with {args}]")
            out.append(AIMessage(content="\n".join(lines) or "[made a tool call]"))
        elif isinstance(m, ToolMessage):
            name = m.name or "tool"
            out.append(HumanMessage(content=f"[result of `{name}`]: {_text_of(m.content)}"))
        else:
            out.append(m)  # HumanMessage / SystemMessage pass through unchanged
    return out


def _summary_token_counter(llm: BaseChatModel):
    """Reproduce ``SummarizationMiddleware``'s default token-counter tuning.

    When the middleware receives the default ``token_counter`` it introspects
    ``model._llm_type`` to tune the approximation. A wrapped ``Runnable`` has no
    ``_llm_type``, so we must pass an explicit counter instead — and we compute
    the same tuning here from the real LLM so counting behaviour is unchanged.
    """
    if getattr(llm, "_llm_type", "").startswith("anthropic-chat"):
        return partial(count_tokens_approximately, use_usage_metadata_scaling=True, chars_per_token=3.3)
    return partial(count_tokens_approximately, use_usage_metadata_scaling=True)


def build_summarization_model(llm: BaseChatModel) -> Runnable:
    """Wrap ``llm`` so tool calls/results are stripped to text before invocation.

    Pass the result as ``SummarizationMiddleware(model=...)`` together with
    ``token_counter=summary_token_counter(llm)`` (a wrapped Runnable has no
    ``_llm_type`` for the middleware's default counter to introspect).
    """
    def _prep(model_input):
        if isinstance(model_input, list):
            return sanitize_for_summary(model_input)
        return model_input  # already-formatted string / prompt: no tool blocks

    return RunnableLambda(_prep) | llm


# Public alias so call sites don't reach for the underscore-prefixed helper.
summary_token_counter = _summary_token_counter
