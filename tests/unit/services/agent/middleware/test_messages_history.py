"""Unit tests for messages_history middleware (MessagesHistoryMiddleware)."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain.messages import ToolMessage

from app.constants import INTERRUPT_CANCEL_REPLY
from app.services.agent.middleware.messages_history import (
    MessagesHistoryMiddleware,
    _select_history_messages,
    _append_new_to_history,
    merge_usage,
    usage_entry,
    usage_entry_from_metadata,
    usage_entry_from_response,
)


_USAGE = {
    "input_tokens": 100,
    "output_tokens": 20,
    "total_tokens": 120,
    "input_token_details": {"cache_read": 30, "cache_creation": 10},
}


def test_select_history_keeps_human_and_ai_messages():
    """Verify HumanMessages and AIMessages with content are kept."""
    messages = [
        HumanMessage(content="hello"),
        AIMessage(content="world"),
    ]

    result = _select_history_messages(messages)

    assert len(result) == 2


def test_select_history_excludes_summarization_messages():
    """Verify messages with lc_source=summarization are excluded."""
    ai_msg = AIMessage(content="summary", additional_kwargs={"lc_source": "summarization"})
    messages = [HumanMessage(content="hello"), ai_msg]

    result = _select_history_messages(messages)

    assert len(result) == 1
    assert result[0].content == "hello"


def test_select_history_excludes_cancel_reply():
    """Verify AIMessages with INTERRUPT_CANCEL_REPLY content are excluded."""
    messages = [
        HumanMessage(content="hello"),
        AIMessage(content=INTERRUPT_CANCEL_REPLY),
    ]

    result = _select_history_messages(messages)

    assert len(result) == 1
    assert result[0].content == "hello"


def test_select_history_keeps_tool_messages_with_confirmation():
    """Verify ToolMessages with confirmation in additional_kwargs are kept."""
    tool_msg = ToolMessage(
        content="done",
        tool_call_id="tc-1",
        name="createPod",
        additional_kwargs={"confirmation": True},
    )
    messages = [tool_msg]

    result = _select_history_messages(messages)

    assert len(result) == 1


def test_select_history_excludes_plain_tool_messages():
    """Verify plain ToolMessages (no confirmation) are excluded."""
    tool_msg = ToolMessage(content="result", tool_call_id="tc-1", name="listPods")
    messages = [tool_msg]

    result = _select_history_messages(messages)

    assert len(result) == 0


def test_select_history_excludes_ai_messages_without_content():
    """Verify AIMessages with empty content are excluded."""
    messages = [AIMessage(content="")]

    result = _select_history_messages(messages)

    assert len(result) == 0


def test_append_new_to_history_appends_new_messages():
    """Verify new messages are appended to history."""
    msg1 = HumanMessage(content="first", id="m1")
    msg2 = AIMessage(content="second", id="m2")

    result = _append_new_to_history([msg2], [msg1])

    assert result is not None
    assert len(result) == 2
    assert result[0].content == "first"
    assert result[1].content == "second"


def test_append_new_to_history_returns_none_when_no_new_messages():
    """Verify None returned when all messages already exist in history."""
    msg1 = HumanMessage(content="first", id="m1")

    result = _append_new_to_history([msg1], [msg1])

    assert result is None


def test_append_new_to_history_updates_existing_when_flag_set():
    """Verify existing messages are replaced when update_existing=True."""
    original = AIMessage(content="old", id="m1")
    updated = AIMessage(content="new", id="m1")

    result = _append_new_to_history([updated], [original], update_existing=True)

    assert result is not None
    assert len(result) == 1
    assert result[0].content == "new"


def test_messages_history_after_model_appends_to_history():
    """Verify after_model appends qualifying messages to messages_history."""
    middleware = MessagesHistoryMiddleware()
    ai_msg = AIMessage(content="response", id="m1")
    state = {"messages": [ai_msg], "messages_history": []}

    result = middleware.after_model(state, MagicMock())

    assert result is not None
    assert len(result["messages_history"]) == 1


def test_messages_history_after_model_returns_none_when_nothing_new():
    """Verify after_model returns None when history is already up to date."""
    middleware = MessagesHistoryMiddleware()
    ai_msg = AIMessage(content="response", id="m1")
    state = {"messages": [ai_msg], "messages_history": [ai_msg]}

    result = middleware.after_model(state, MagicMock())

    assert result is None


def test_usage_entry_from_metadata_maps_fields():
    """Verify usage_metadata (incl. cache details) maps to a token_usage entry."""
    entry = usage_entry_from_metadata(_USAGE, "agent")

    assert entry == {
        "source": "agent",
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cache_read": 30,
        "cache_creation": 10,
    }


def test_usage_entry_from_metadata_returns_none_when_empty():
    assert usage_entry_from_metadata(None, "agent") is None
    assert usage_entry_from_metadata({}, "ui-tools") is None


def test_usage_entry_from_response_prefers_usage_metadata():
    msg = AIMessage(content="x", id="m1", usage_metadata=_USAGE)
    assert usage_entry_from_response(msg, "summarization")["total_tokens"] == 120


def test_usage_entry_from_response_falls_back_to_openai_style_metadata():
    """Non-streamed calls may only report counts in response_metadata.token_usage."""
    msg = AIMessage(
        content="x", id="m1",
        response_metadata={"token_usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}},
    )
    entry = usage_entry_from_response(msg, "summarization")
    assert entry == {
        "source": "summarization",
        "input_tokens": 30,
        "output_tokens": 5,
        "total_tokens": 35,
        "cache_read": 0,
        "cache_creation": 0,
    }


def test_usage_entry_from_response_falls_back_to_anthropic_style_usage():
    msg = AIMessage(
        content="x", id="m1",
        response_metadata={"usage": {"input_tokens": 12, "output_tokens": 8}},
    )
    entry = usage_entry_from_response(msg, "summarization")
    assert entry["input_tokens"] == 12
    assert entry["output_tokens"] == 8
    assert entry["total_tokens"] == 20  # derived when not provided


def test_usage_entry_from_response_returns_none_when_no_usage_anywhere():
    msg = AIMessage(content="x", id="m1")
    assert usage_entry_from_response(msg, "summarization") is None


def test_merge_usage_merges_by_key():
    left = {"a": {"total_tokens": 1}}
    right = {"b": {"total_tokens": 2}}

    assert merge_usage(left, right) == {"a": {"total_tokens": 1}, "b": {"total_tokens": 2}}
    # Right wins on key collision (idempotent re-recording).
    assert merge_usage({"a": {"total_tokens": 1}}, {"a": {"total_tokens": 9}}) == {"a": {"total_tokens": 9}}
    assert merge_usage(None, None) == {}


def test_after_model_merges_with_existing_token_usage():
    """Verify after_model returns the full merged dict (channel is LastValue)."""
    middleware = MessagesHistoryMiddleware()
    ai_msg = AIMessage(content="response", id="m2", usage_metadata=_USAGE)
    existing = {"m1": {"source": "agent", "total_tokens": 5}}
    state = {"messages": [ai_msg], "messages_history": [], "token_usage": existing}

    result = middleware.after_model(state, MagicMock())

    assert set(result["token_usage"]) == {"m1", "m2"}
    assert result["token_usage"]["m1"]["total_tokens"] == 5
    assert result["token_usage"]["m2"]["total_tokens"] == 120


def test_after_model_captures_agent_token_usage():
    """Verify after_model records the last AIMessage's usage tagged 'agent'."""
    middleware = MessagesHistoryMiddleware()
    ai_msg = AIMessage(content="response", id="m1", usage_metadata=_USAGE)
    state = {"messages": [ai_msg], "messages_history": []}

    result = middleware.after_model(state, MagicMock())

    assert result["token_usage"] == {"m1": usage_entry(ai_msg, "agent")}
    assert result["token_usage"]["m1"]["total_tokens"] == 120
    assert result["token_usage"]["m1"]["cache_read"] == 30


def test_after_model_skips_summarization_message_usage():
    """Verify summarization-injected messages are not counted as agent usage."""
    middleware = MessagesHistoryMiddleware()
    ai_msg = AIMessage(
        content="summary",
        id="m1",
        usage_metadata=_USAGE,
        additional_kwargs={"lc_source": "summarization"},
    )
    state = {"messages": [ai_msg], "messages_history": []}

    result = middleware.after_model(state, MagicMock())

    # No history append (excluded) and no token_usage capture.
    assert result is None


def test_after_model_no_token_usage_without_metadata():
    """Verify AIMessages without usage_metadata produce no token_usage key."""
    middleware = MessagesHistoryMiddleware()
    ai_msg = AIMessage(content="response", id="m1")
    state = {"messages": [ai_msg], "messages_history": []}

    result = middleware.after_model(state, MagicMock())

    assert "token_usage" not in result
    assert len(result["messages_history"]) == 1


def test_messages_history_after_agent_updates_existing():
    """Verify after_agent updates existing messages (e.g. with ui_tools added)."""
    middleware = MessagesHistoryMiddleware()
    original = AIMessage(content="response", id="m1")
    updated = AIMessage(content="response", id="m1", additional_kwargs={"ui_tools": [{"toolName": "show-yaml"}]})
    state = {"messages": [updated], "messages_history": [original]}

    result = middleware.after_agent(state, MagicMock())

    assert result is not None
    assert result["messages_history"][0].additional_kwargs.get("ui_tools") is not None
