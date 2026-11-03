"""Unit tests for tool_response_overflow middleware."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.exceptions import ContextOverflowError

from app.services.agent.middleware.tool_response_overflow import (
    TRUNCATION_NOTE_PREFIX,
    TRUNCATION_REMOVED_PREFIX,
    _content_length,
    _truncate_content,
    tool_response_overflow_middleware,
)


def _ai_tool_call(call_id: str, name: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": {}}])


def _request(messages: list) -> ModelRequest:
    return ModelRequest(model=MagicMock(), messages=messages)


def _messages_with_tool_results(big: str, small: str = "small result") -> list:
    return [
        HumanMessage(content="old question", id="h-1"),
        _ai_tool_call("call-0", "getVersion"),
        ToolMessage(content="v2.13", name="getVersion", tool_call_id="call-0", id="t-0"),
        HumanMessage(content="list all pods", id="h-2"),
        _ai_tool_call("call-1", "listPods"),
        ToolMessage(content=big, name="listPods", tool_call_id="call-1", id="t-1"),
        _ai_tool_call("call-2", "getCluster"),
        ToolMessage(content=small, name="getCluster", tool_call_id="call-2", id="t-2"),
    ]


def _persisted_messages(result) -> list:
    assert isinstance(result, ExtendedModelResponse)
    return result.command.update["messages"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_content_length_handles_text_blocks():
    content = [{"type": "text", "text": "abc"}, {"type": "image", "url": "x"}, {"type": "text", "text": "de"}]
    assert _content_length(content) == 5


def test_content_length_ignores_removed_note():
    assert _content_length(f"{TRUNCATION_REMOVED_PREFIX}1000 characters removed ...]") == 0


def test_content_length_ignores_truncation_note():
    truncated = _truncate_content("a" * 1000, 400, 1000)
    assert _content_length(truncated) == 400


def test_truncate_content_from_text_blocks():
    content = [{"type": "text", "text": "a" * 600}, {"type": "text", "text": "b" * 600}]
    result = _truncate_content(content, 700, 1200)
    assert result.startswith("a" * 600 + "b" * 100 + TRUNCATION_NOTE_PREFIX)
    assert "from 1200 to 700 characters" in result


# ---------------------------------------------------------------------------
# middleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_overflow_returns_handler_result_unchanged():
    middleware = tool_response_overflow_middleware()
    request = _request(_messages_with_tool_results("y" * 4000))
    response = ModelResponse(result=[AIMessage(content="ok")])
    handler = AsyncMock(return_value=response)

    result = await middleware.awrap_model_call(request, handler)

    assert result is response
    handler.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_overflow_retries_with_largest_tool_message_halved_and_persists_it():
    middleware = tool_response_overflow_middleware()
    messages = _messages_with_tool_results("y" * 4000)
    response = ModelResponse(result=[AIMessage(content="ok")])
    handler = AsyncMock(side_effect=[ContextOverflowError("too long"), response])

    result = await middleware.awrap_model_call(_request(messages), handler)

    assert handler.await_count == 2
    retried = handler.await_args_list[1].args[0].messages
    assert retried[5].content.startswith("y" * 2000 + TRUNCATION_NOTE_PREFIX)
    assert "from 4000 to 2000 characters" in retried[5].content
    # Other tool results are untouched.
    assert retried[2].content == "v2.13"
    assert retried[7].content == "small result"
    # The original messages are not mutated.
    assert messages[5].content == "y" * 4000

    assert result.model_response is response
    persisted = _persisted_messages(result)
    assert len(persisted) == 1
    assert persisted[0].id == "t-1"
    assert persisted[0].tool_call_id == "call-1"
    assert persisted[0].content == retried[5].content


@pytest.mark.asyncio
async def test_overflow_keeps_halving_and_persists_only_last_copy():
    middleware = tool_response_overflow_middleware()
    handler = AsyncMock(side_effect=[ContextOverflowError("1"), ContextOverflowError("2"), AIMessage(content="ok")])

    result = await middleware.awrap_model_call(_request(_messages_with_tool_results("y" * 4000)), handler)

    retried = handler.await_args_list[2].args[0].messages[5].content
    assert retried.startswith("y" * 1000 + TRUNCATION_NOTE_PREFIX)
    assert "from 4000 to 1000 characters" in retried
    assert retried.count(TRUNCATION_NOTE_PREFIX) == 1

    persisted = _persisted_messages(result)
    assert [m.content for m in persisted] == [retried]


@pytest.mark.asyncio
async def test_bare_ai_message_response_is_wrapped():
    middleware = tool_response_overflow_middleware()
    ai_message = AIMessage(content="ok")
    handler = AsyncMock(side_effect=[ContextOverflowError("too long"), ai_message])

    result = await middleware.awrap_model_call(_request(_messages_with_tool_results("y" * 4000)), handler)

    assert isinstance(result.model_response, ModelResponse)
    assert result.model_response.result == [ai_message]


@pytest.mark.asyncio
async def test_tool_message_from_earlier_turn_is_truncated():
    middleware = tool_response_overflow_middleware()
    messages = [
        HumanMessage(content="list all pods", id="h-1"),
        _ai_tool_call("call-1", "listPods"),
        ToolMessage(content="y" * 4000, name="listPods", tool_call_id="call-1", id="t-1"),
        AIMessage(content="here are the pods", id="a-1"),
        HumanMessage(content="thanks, and the clusters?", id="h-2"),
    ]
    handler = AsyncMock(side_effect=[ContextOverflowError("too long"), AIMessage(content="ok")])

    result = await middleware.awrap_model_call(_request(messages), handler)

    persisted = _persisted_messages(result)
    assert persisted[0].id == "t-1"
    assert persisted[0].content.startswith("y" * 2000 + TRUNCATION_NOTE_PREFIX)


@pytest.mark.asyncio
async def test_overflow_after_max_retries_removes_whole_tool_response():
    middleware = tool_response_overflow_middleware(max_retries=2)
    handler = AsyncMock(
        side_effect=[ContextOverflowError("0"), ContextOverflowError("1"), ContextOverflowError("2"), AIMessage(content="ok")]
    )

    result = await middleware.awrap_model_call(_request(_messages_with_tool_results("y" * 40_000)), handler)

    assert handler.await_count == 4
    last_request = handler.await_args_list[3].args[0].messages
    assert last_request[5].content.startswith(TRUNCATION_REMOVED_PREFIX + "40000 characters removed")

    persisted = _persisted_messages(result)
    assert len(persisted) == 1
    assert persisted[0].id == "t-1"
    assert persisted[0].content == last_request[5].content


@pytest.mark.asyncio
async def test_overflow_reraises_when_removed_response_still_overflows():
    middleware = tool_response_overflow_middleware(max_retries=2)
    handler = AsyncMock(side_effect=ContextOverflowError("too long"))

    with pytest.raises(ContextOverflowError):
        await middleware.awrap_model_call(_request(_messages_with_tool_results("y" * 40_000)), handler)

    assert handler.await_count == 4


@pytest.mark.asyncio
async def test_small_tool_message_is_removed_without_halving():
    middleware = tool_response_overflow_middleware()
    handler = AsyncMock(side_effect=[ContextOverflowError("too long"), AIMessage(content="ok")])

    result = await middleware.awrap_model_call(_request(_messages_with_tool_results("tiny")), handler)

    assert handler.await_count == 2
    persisted = _persisted_messages(result)
    assert persisted[0].id == "t-2"
    assert persisted[0].content.startswith(TRUNCATION_REMOVED_PREFIX + "12 characters removed")


@pytest.mark.asyncio
async def test_overflow_without_tool_message_reraises():
    middleware = tool_response_overflow_middleware()
    handler = AsyncMock(side_effect=ContextOverflowError("too long"))

    with pytest.raises(ContextOverflowError):
        await middleware.awrap_model_call(_request([HumanMessage(content="hello")]), handler)

    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_other_errors_are_not_retried():
    middleware = tool_response_overflow_middleware()
    handler = AsyncMock(side_effect=ValueError("boom"))

    with pytest.raises(ValueError):
        await middleware.awrap_model_call(_request(_messages_with_tool_results("y" * 4000)), handler)

    handler.assert_awaited_once()
