import logging
from collections.abc import Awaitable, Callable

from botocore.exceptions import ClientError
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain.messages import AIMessage, HumanMessage, ToolMessage
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

# Bedrock rejects oversized request bodies with a plain ValidationException that
# langchain_aws does not map to ContextOverflowError.
_REQUEST_TOO_LARGE_MARKERS = ("length limit exceeded", "failed to buffer the request body")


def _is_request_too_large(error: Exception) -> bool:
    if not isinstance(error, ClientError):
        return False
    info = error.response.get("Error", {})
    if info.get("Code", "").lower() != "validationexception":
        return False
    message = str(info.get("Message", "")).lower()
    return any(marker in message for marker in _REQUEST_TOO_LARGE_MARKERS)


def tool_response_overflow_middleware(max_retries: int = 3):
    """Wrap-model-call middleware: retry with shorter tool responses on context overflow.

    When the model call fails with ``ContextOverflowError`` (e.g. an MCP tool returned a huge
    payload), the largest ``ToolMessage`` after the last ``HumanMessage`` is cut in half and the
    call is retried. Earlier ToolMessages are never touched, so the cached prompt prefix stays valid.
    Each ToolMessage is halved up to ``max_retries`` times, then its content is replaced by a note.
    The original error is raised when there is nothing left to shorten.

    On success the shortened ToolMessages are written back to the ``messages`` channel
    (same id, so the ``add_messages`` reducer replaces them in place); otherwise every later
    request would send the full response again and overflow too.
    """

    @wrap_model_call
    async def tool_response_overflow(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        halvings: dict[int, int] = {}
        original_lengths: dict[int, int] = {}
        truncated: dict[int, ToolMessage] = {}
        while True:
            try:
                response = await handler(request)
                # Persist the shortened copies (same id) so later calls and turns don't overflow again.
                return _with_state_update(response, list(truncated.values()))
            except (ContextOverflowError, ClientError) as error:
                if not isinstance(error, ContextOverflowError) and not _is_request_too_large(error):
                    raise
                index = _pick_tool_message(request.messages)
                if index is None:
                    raise

                message = request.messages[index]
                original = original_lengths.setdefault(index, _content_length(message.content))
                remove = halvings.get(index, 0) >= max_retries
                truncated[index] = _shorten(message, original, remove)
                halvings[index] = halvings.get(index, 0) + 1

                # Copy instead of mutating: the list may be shared with the graph state.
                messages = list(request.messages)
                messages[index] = truncated[index]
                request = request.override(messages=messages)

    return tool_response_overflow


def _pick_tool_message(messages: list) -> int | None:
    """Return the index of the largest non-empty ToolMessage after the last HumanMessage, if any."""
    largest_index = None
    largest_length = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            length = _content_length(msg.content)
            if length > largest_length:
                largest_index, largest_length = i, length
    return largest_index


def _shorten(message: ToolMessage, original_length: int, remove: bool) -> ToolMessage:
    """Return a copy of ``message`` with its content halved, or removed if ``remove`` or too small."""
    current_length = _content_length(message.content)
    if remove or current_length <= MIN_TOOL_RESPONSE_CHARS:
        logging.warning(
            "Context overflow: removing '%s' tool response of %d characters and retrying",
            message.name, current_length,
        )
        content = TRUNCATION_REMOVED_NOTE.format(original=original_length)
    else:
        kept = current_length // 2
        logging.warning(
            "Context overflow: truncating '%s' tool response from %d to %d characters and retrying",
            message.name, current_length, kept,
        )
        content = _truncate_content(message.content, kept, original_length)
    return message.model_copy(update={"content": content})


def _with_state_update(
    response: ModelResponse | AIMessage, truncated: list[ToolMessage]
) -> ModelResponse | AIMessage | ExtendedModelResponse:
    """Attach a command replacing the truncated ToolMessages in the agent state."""
    if not truncated:
        return response
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
