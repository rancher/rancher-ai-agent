import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain.messages import AIMessage, ToolMessage
from langchain_core.exceptions import ContextOverflowError
from langgraph.types import Command

# Below this size a tool response is not worth halving; it is removed instead.
MIN_TOOL_RESPONSE_CHARS = 500

TRUNCATION_NOTE_PREFIX = "\n\n[Tool response truncated from "
TRUNCATION_NOTE = (
    TRUNCATION_NOTE_PREFIX + "{original} to {kept} characters because it exceeded "
    "the model context window. Tell the user the result is incomplete and suggest narrowing "
    "the query (e.g. namespace, label selector or resource name).]"
)

TRUNCATION_REMOVED_PREFIX = "[Tool response of "
TRUNCATION_REMOVED_NOTE = (
    TRUNCATION_REMOVED_PREFIX + "{original} characters removed because it exceeded the model "
    "context window. Tell the user the result could not be shown and suggest narrowing the "
    "query (e.g. namespace, label selector or resource name).]"
)


def tool_response_overflow_middleware(max_retries: int = 3):
    """Wrap-model-call middleware: retry with shorter tool responses on context overflow.

    When the model call fails with ``ContextOverflowError`` (e.g. an MCP tool returned a huge
    payload), the largest ``ToolMessage`` is cut in half and the call is retried, up to
    ``max_retries`` times. If it still overflows, the content of that ToolMessage is replaced
    by a note and the call is retried one last time.

    On success the shortened ToolMessages are written back to the ``messages`` channel
    (same id, so the ``add_messages`` reducer replaces them in place); otherwise every later
    request would send the full response again and overflow too.
    """

    @wrap_model_call
    async def tool_response_overflow(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        try:
            return await handler(request)
        except ContextOverflowError as overflow:
            last_error = overflow

        truncated: dict[int, ToolMessage] = {}
        original_lengths: dict[int, int] = {}
        halvings = 0
        while True:
            index = _largest_tool_message(request.messages)
            if index is None:
                raise last_error

            message = request.messages[index]
            current_length = _content_length(message.content)
            if current_length == 0:
                raise last_error

            original_length = original_lengths.setdefault(index, current_length)
            removing = halvings >= max_retries or current_length <= MIN_TOOL_RESPONSE_CHARS
            if removing:
                logging.warning(
                    "Context overflow: removing '%s' tool response of %d characters and retrying",
                    message.name, current_length,
                )
                content = TRUNCATION_REMOVED_NOTE.format(original=original_length)
            else:
                halvings += 1
                kept = current_length // 2
                logging.warning(
                    "Context overflow: truncating '%s' tool response from %d to %d characters and retrying",
                    message.name, current_length, kept,
                )
                content = _truncate_content(message.content, kept, original_length)

            truncated[index] = message.model_copy(update={"content": content})
            messages = list(request.messages)
            messages[index] = truncated[index]
            request = request.override(messages=messages)

            try:
                response = await handler(request)
            except ContextOverflowError as overflow:
                if removing:
                    raise
                last_error = overflow
                continue

            return _with_state_update(response, list(truncated.values()))

    return tool_response_overflow


def _with_state_update(
    response: ModelResponse | AIMessage, truncated: list[ToolMessage]
) -> ModelResponse | ExtendedModelResponse:
    """Attach a command replacing the truncated ToolMessages in the agent state."""
    if isinstance(response, AIMessage):
        response = ModelResponse(result=[response])

    persisted = [m for m in truncated if m.id]
    if len(persisted) != len(truncated):
        logging.warning("Truncated tool response without id cannot be persisted in the agent state")
    if not persisted:
        return response

    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"messages": persisted}),
    )


def _largest_tool_message(messages: list) -> int | None:
    """Return the index of the largest ToolMessage, if any."""
    largest_index = None
    largest_length = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            length = _content_length(msg.content)
            if length > largest_length:
                largest_index, largest_length = i, length
    return largest_index


def _content_length(content: str | list) -> int:
    """Return the number of tool response characters, ignoring a previous truncation note."""
    return len(_content_text(content))


def _content_text(content: str | list) -> str:
    """Return the text of a message content (str or content blocks) without a truncation note."""
    if isinstance(content, str):
        text = content
    else:
        text = "".join(_block_text(block) for block in content)
    if text.startswith(TRUNCATION_REMOVED_PREFIX):
        return ""
    position = text.find(TRUNCATION_NOTE_PREFIX)
    return text[:position] if position != -1 else text


def _block_text(block: str | dict) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and block.get("type", "text") == "text":
        return block.get("text", "")
    return ""


def _truncate_content(content: str | list, max_chars: int, original_length: int) -> str:
    """Return the first ``max_chars`` text characters of ``content`` followed by a truncation note."""
    text = _content_text(content)[:max_chars]
    return text + TRUNCATION_NOTE.format(original=original_length, kept=len(text))
