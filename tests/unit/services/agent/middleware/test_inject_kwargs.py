"""Unit tests for inject_kwargs middleware (inject_additional_kwargs_middleware)."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain.messages import ToolMessage

from app.services.agent.middleware.inject_kwargs import inject_additional_kwargs_middleware


@patch("app.services.agent.middleware.inject_kwargs.get_config")
def test_injects_request_id_into_ai_message(mock_get_config):
    """Verify request_id and created_at are injected into the last AIMessage."""
    mock_get_config.return_value = {"configurable": {"request_id": "req-42"}}

    middleware = inject_additional_kwargs_middleware()
    ai_msg = AIMessage(content="hello")
    state = {"messages": [ai_msg]}

    result = middleware.after_model(state, MagicMock())

    assert result is not None
    assert result["messages"][0].additional_kwargs["request_id"] == "req-42"
    assert "created_at" in result["messages"][0].additional_kwargs


@patch("app.services.agent.middleware.inject_kwargs.get_config")
def test_inject_kwargs_returns_none_when_no_request_id(mock_get_config):
    """Verify None returned when no request_id in config."""
    mock_get_config.return_value = {"configurable": {}}

    middleware = inject_additional_kwargs_middleware()
    state = {"messages": [AIMessage(content="hello")]}

    result = middleware.after_model(state, MagicMock())

    assert result is None


@patch("app.services.agent.middleware.inject_kwargs.get_config")
def test_inject_kwargs_returns_none_when_empty_messages(mock_get_config):
    """Verify None returned when messages list is empty."""
    mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}

    middleware = inject_additional_kwargs_middleware()
    state = {"messages": []}

    result = middleware.after_model(state, MagicMock())

    assert result is None


@patch("app.services.agent.middleware.inject_kwargs.get_config")
def test_inject_kwargs_returns_none_when_last_message_is_not_ai(mock_get_config):
    """Verify None returned when last message is not an AIMessage."""
    mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}

    middleware = inject_additional_kwargs_middleware()
    state = {"messages": [HumanMessage(content="hi")]}

    result = middleware.after_model(state, MagicMock())

    assert result is None


@patch("app.services.agent.middleware.inject_kwargs.get_config")
def test_injects_interrupt_message_from_preceding_tool_message(mock_get_config):
    """Verify interrupt_message is extracted from the ToolMessage before the AIMessage."""
    mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}

    middleware = inject_additional_kwargs_middleware()
    tool_msg = ToolMessage(
        content="done",
        tool_call_id="tc-1",
        name="createPod",
        additional_kwargs={"interrupt_message": "<confirmation-response>plan</confirmation-response>"},
    )
    ai_msg = AIMessage(content="Created the pod.")
    state = {"messages": [HumanMessage(content="create pod"), tool_msg, ai_msg]}

    result = middleware.after_model(state, MagicMock())

    assert result is not None
    assert result["messages"][0].additional_kwargs["interrupt_message"] == "<confirmation-response>plan</confirmation-response>"


@patch("app.services.agent.middleware.inject_kwargs.get_config")
def test_injects_mcp_response_from_tool_additional_kwargs(mock_get_config):
    """Verify mcp_response is extracted from ToolMessage additional_kwargs."""
    mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}

    middleware = inject_additional_kwargs_middleware()
    tool_msg = ToolMessage(
        content="result",
        tool_call_id="tc-1",
        name="listPods",
        additional_kwargs={"mcp_response": "<mcp-response>{}</mcp-response>"},
    )
    ai_msg = AIMessage(content="Here are the pods.")
    state = {"messages": [HumanMessage(content="list pods"), tool_msg, ai_msg]}

    result = middleware.after_model(state, MagicMock())

    assert result is not None
    assert result["messages"][0].additional_kwargs["mcp_response"] == "<mcp-response>{}</mcp-response>"
