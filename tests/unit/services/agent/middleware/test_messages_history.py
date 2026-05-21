"""Unit tests for messages_history middleware (MessagesHistoryMiddleware)."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain.messages import ToolMessage

from app.constants import INTERRUPT_CANCEL_REPLY
from app.services.agent.middleware.messages_history import (
    MessagesHistoryMiddleware,
    _select_history_messages,
    _append_new_to_history,
)


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


def test_messages_history_after_agent_updates_existing():
    """Verify after_agent updates existing messages (e.g. with ui_tools added)."""
    middleware = MessagesHistoryMiddleware()
    original = AIMessage(content="response", id="m1")
    updated = AIMessage(content="response", id="m1", additional_kwargs={"ui_tools": [{"toolName": "show-yaml"}]})
    state = {"messages": [updated], "messages_history": [original]}

    result = middleware.after_agent(state, MagicMock())

    assert result is not None
    assert result["messages_history"][0].additional_kwargs.get("ui_tools") is not None
