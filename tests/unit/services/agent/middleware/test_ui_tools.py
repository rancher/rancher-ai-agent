"""Unit tests for ui_tools middleware (ui_tools_middleware, _collect_context_until_human, _extract_tool_text)."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain.messages import ToolMessage

from app.services.agent.middleware.ui_tools import (
    ui_tools_middleware,
    _collect_context_until_human,
    _extract_tool_text,
)


@patch("app.services.agent.middleware.ui_tools._dispatch_ui_tools_event")
@patch("app.services.agent.middleware.ui_tools.get_config")
def test_ui_tools_skips_when_last_message_not_ai(mock_get_config, mock_dispatch_event):
    """Verify None returned when last message is not AIMessage."""
    mock_llm = MagicMock()
    middleware = ui_tools_middleware(mock_llm)
    state = {"messages": [HumanMessage(content="hi")]}

    result = middleware.after_agent(state, MagicMock())

    assert result is None
    mock_dispatch_event.assert_not_called()


@patch("app.services.agent.middleware.ui_tools._dispatch_ui_tools_event")
@patch("app.services.agent.middleware.ui_tools.get_config")
def test_ui_tools_skips_when_only_when_direct_and_no_agent_in_config(mock_get_config, mock_dispatch_event):
    """Verify middleware skips when only_when_direct=True and no agent in config."""
    mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}
    mock_llm = MagicMock()
    middleware = ui_tools_middleware(mock_llm, only_when_direct=True)
    state = {"messages": [AIMessage(content="answer")]}

    result = middleware.after_agent(state, MagicMock())

    assert result is None
    mock_dispatch_event.assert_not_called()


@patch("app.services.agent.middleware.ui_tools._dispatch_ui_tools_event")
@patch("app.services.agent.middleware.ui_tools.get_config")
def test_ui_tools_executes_when_only_when_direct_and_agent_set(mock_get_config, mock_dispatch_event):
    """Verify middleware executes when only_when_direct=True and agent is set."""
    mock_get_config.return_value = {
        "configurable": {"request_id": "req-1", "agent": "rancher"}
    }
    mock_dispatch_event.return_value = [{"toolName": "show-yaml", "input": {}}]
    mock_llm = MagicMock()
    middleware = ui_tools_middleware(mock_llm, only_when_direct=True)
    ai_msg = AIMessage(content="answer")
    state = {"messages": [ai_msg]}

    result = middleware.after_agent(state, MagicMock())

    assert result is not None
    assert result["messages"][0].additional_kwargs["ui_tools"] == [
        {"toolName": "show-yaml", "input": {}}
    ]


@patch("app.services.agent.middleware.ui_tools._dispatch_ui_tools_event")
@patch("app.services.agent.middleware.ui_tools.get_config")
def test_ui_tools_returns_none_when_no_ui_tools_selected(mock_get_config, mock_dispatch_event):
    """Verify None returned when no UI tools are selected."""
    mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}
    mock_dispatch_event.return_value = []
    mock_llm = MagicMock()
    middleware = ui_tools_middleware(mock_llm, only_when_direct=False)
    state = {"messages": [AIMessage(content="answer")]}

    result = middleware.after_agent(state, MagicMock())

    assert result is None


@patch("app.services.agent.middleware.ui_tools._dispatch_ui_tools_event")
@patch("app.services.agent.middleware.ui_tools.get_config")
def test_ui_tools_skips_when_no_request_id(mock_get_config, mock_dispatch_event):
    """Verify None returned when request_id is missing from config."""
    mock_get_config.return_value = {"configurable": {}}
    mock_dispatch_event.return_value = [{"toolName": "show-yaml", "input": {}}]
    mock_llm = MagicMock()
    middleware = ui_tools_middleware(mock_llm)
    state = {"messages": [AIMessage(content="answer")]}

    result = middleware.after_agent(state, MagicMock())

    assert result is None


def test_collect_context_collects_messages_back_to_human():
    """Verify messages are collected from end back to last HumanMessage."""
    state = {
        "messages": [
            HumanMessage(content="What pods are running?"),
            AIMessage(content="Let me check."),
            ToolMessage(content="pod-1, pod-2", tool_call_id="tc-1", name="listPods"),
            AIMessage(content="You have 2 pods."),
        ]
    }

    result = _collect_context_until_human(state)

    assert "What pods are running?" in result
    assert "Let me check." in result
    assert "pod-1, pod-2" in result
    assert "You have 2 pods." in result


def test_collect_context_returns_empty_for_empty_messages():
    """Verify empty string for empty messages."""
    result = _collect_context_until_human({"messages": []})

    assert result == ""


def test_collect_context_includes_request_metadata():
    """Verify request_metadata in HumanMessage additional_kwargs is used."""
    human_msg = HumanMessage(
        content="ignored",
        additional_kwargs={
            "request_metadata": {"user_input": "show pods", "context": {"page": "cluster"}}
        },
    )
    state = {"messages": [human_msg, AIMessage(content="Here are the pods.")]}

    result = _collect_context_until_human(state)

    assert "show pods" in result
    assert "cluster" in result


def test_collect_context_stops_at_last_human_message():
    """Verify collection stops at the last HumanMessage and does not include earlier ones."""
    state = {
        "messages": [
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            HumanMessage(content="new question"),
            AIMessage(content="new answer"),
        ]
    }

    result = _collect_context_until_human(state)

    assert "new question" in result
    assert "new answer" in result
    assert "old question" not in result


def test_extract_tool_text_from_list_format():
    """Verify text extraction from list of text items."""
    content = json.dumps([
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ])

    result = _extract_tool_text(content)

    assert "first" in result
    assert "second" in result


def test_extract_tool_text_from_dict_with_text_key():
    """Verify text extraction from dict with 'text' key."""
    content = json.dumps({"text": "hello world"})

    result = _extract_tool_text(content)

    assert result == "hello world"


def test_extract_tool_text_returns_json_string_for_dict_without_text():
    """Verify JSON dump for dict without 'text' key."""
    content = json.dumps({"key": "value"})

    result = _extract_tool_text(content)

    assert '"key"' in result
    assert '"value"' in result


def test_extract_tool_text_returns_plain_string_for_non_json():
    """Verify plain string returned for non-JSON content."""
    result = _extract_tool_text("plain text result")

    assert result == "plain text result"


def test_extract_tool_text_handles_native_list_input():
    """Verify extraction when content is a native list (not JSON string)."""
    content = [{"type": "text", "text": "native item"}]

    result = _extract_tool_text(content)

    assert "native item" in result
