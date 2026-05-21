"""Unit tests for cancel_check middleware (cancel_human_validation_middleware)."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain.messages import ToolMessage

from app.services.agent.middleware import INTERRUPT_CANCEL_MESSAGE
from app.services.agent.middleware.cancel_human_validation import cancel_human_validation_middleware


def test_jumps_to_end_on_cancel_message():
    """Verify jump_to end when last tool message is a cancellation."""
    middleware = cancel_human_validation_middleware()
    cancel_msg = ToolMessage(
        content=INTERRUPT_CANCEL_MESSAGE,
        tool_call_id="tc-1",
        name="someTool",
    )
    state = {"messages": [cancel_msg]}

    result = middleware.before_model(state, MagicMock())

    assert result is not None
    assert result["jump_to"] == "end"
    assert isinstance(result["messages"][0], AIMessage)
    assert "canceled" in result["messages"][0].content


def test_cancel_check_returns_none_when_last_message_is_not_cancel():
    """Verify None returned when last message is a normal tool result."""
    middleware = cancel_human_validation_middleware()
    tool_msg = ToolMessage(content="success", tool_call_id="tc-1", name="tool")
    state = {"messages": [tool_msg]}

    result = middleware.before_model(state, MagicMock())

    assert result is None


def test_cancel_check_returns_none_on_empty_messages():
    """Verify None returned for empty messages list."""
    middleware = cancel_human_validation_middleware()
    state = {"messages": []}

    result = middleware.before_model(state, MagicMock())

    assert result is None


def test_cancel_check_returns_none_when_last_message_is_ai():
    """Verify None returned when last message is an AIMessage."""
    middleware = cancel_human_validation_middleware()
    state = {"messages": [AIMessage(content="hello")]}

    result = middleware.before_model(state, MagicMock())

    assert result is None
