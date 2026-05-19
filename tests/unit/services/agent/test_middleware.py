"""
Unit tests for the middleware module (app.services.agent.middleware).

Tests: create_inject_request_id_middleware, create_cancel_check_middleware,
create_ui_tools_middleware, _collect_context_until_human, _extract_tool_text.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.services.agent.middleware import (
    INTERRUPT_CANCEL_MESSAGE,
    _collect_context_until_human,
    _extract_tool_text,
    cancel_check_middleware,
    inject_additional_kwargs_middleware,
    create_ui_tools_middleware,
)


# ============================================================================
# create_inject_request_id_middleware Tests
# ============================================================================


class TestInjectRequestIdMiddleware:
    """Test inject_request_id after-model middleware."""

    @patch("app.services.agent.middleware.inject_kwargs.get_config")
    def test_injects_request_id_into_ai_message(self, mock_get_config):
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
    def test_returns_none_when_no_request_id(self, mock_get_config):
        """Verify None returned when no request_id in config."""
        mock_get_config.return_value = {"configurable": {}}

        middleware = inject_additional_kwargs_middleware()
        state = {"messages": [AIMessage(content="hello")]}

        result = middleware.after_model(state, MagicMock())

        assert result is None

    @patch("app.services.agent.middleware.inject_kwargs.get_config")
    def test_returns_none_when_empty_messages(self, mock_get_config):
        """Verify None returned when messages list is empty."""
        mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}

        middleware = inject_additional_kwargs_middleware()
        state = {"messages": []}

        result = middleware.after_model(state, MagicMock())

        assert result is None

    @patch("app.services.agent.middleware.inject_kwargs.get_config")
    def test_returns_none_when_last_message_is_not_ai(self, mock_get_config):
        """Verify None returned when last message is not an AIMessage."""
        mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}

        middleware = inject_additional_kwargs_middleware()
        state = {"messages": [HumanMessage(content="hi")]}

        result = middleware.after_model(state, MagicMock())

        assert result is None


# ============================================================================
# create_cancel_check_middleware Tests
# ============================================================================


class TestCancelCheckMiddleware:
    """Test cancel_check before-model middleware."""

    def test_jumps_to_end_on_cancel_message(self):
        """Verify jump_to end when last tool message is a cancellation."""
        middleware = cancel_check_middleware()
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

    def test_returns_none_when_no_cancel(self):
        """Verify None returned when last message is not a cancellation."""
        middleware = cancel_check_middleware()
        tool_msg = ToolMessage(content="success", tool_call_id="tc-1", name="tool")
        state = {"messages": [tool_msg]}

        result = middleware.before_model(state, MagicMock())

        assert result is None

    def test_returns_none_on_empty_messages(self):
        """Verify None returned for empty messages."""
        middleware = cancel_check_middleware()
        state = {"messages": []}

        result = middleware.before_model(state, MagicMock())

        assert result is None


# ============================================================================
# create_ui_tools_middleware Tests
# ============================================================================


class TestUIToolsMiddleware:
    """Test ui_tools_dispatch after-agent middleware."""

    @patch("app.services.agent.middleware.ui_tools._dispatch_ui_tools_event")
    @patch("app.services.agent.middleware.ui_tools.get_config")
    def test_skips_when_last_message_not_ai(self, mock_get_config, mock_dispatch_event):
        """Verify None returned when last message is not AIMessage."""
        mock_llm = MagicMock()
        middleware = create_ui_tools_middleware(mock_llm)
        state = {"messages": [HumanMessage(content="hi")]}

        result = middleware.after_agent(state, MagicMock())

        assert result is None
        mock_dispatch_event.assert_not_called()

    @patch("app.services.agent.middleware.ui_tools._dispatch_ui_tools_event")
    @patch("app.services.agent.middleware.ui_tools.get_config")
    def test_skips_when_only_when_direct_and_no_agent_in_config(
        self, mock_get_config, mock_dispatch_event
    ):
        """Verify middleware skips when only_when_direct=True and no agent in config."""
        mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}
        mock_llm = MagicMock()
        middleware = create_ui_tools_middleware(mock_llm, only_when_direct=True)
        state = {"messages": [AIMessage(content="answer")]}

        result = middleware.after_agent(state, MagicMock())

        assert result is None
        mock_dispatch_event.assert_not_called()

    @patch("app.services.agent.middleware.ui_tools._dispatch_ui_tools_event")
    @patch("app.services.agent.middleware.ui_tools.get_config")
    def test_executes_when_only_when_direct_and_agent_set(
        self, mock_get_config, mock_dispatch_event
    ):
        """Verify middleware executes when only_when_direct=True and agent is set."""
        mock_get_config.return_value = {
            "configurable": {"request_id": "req-1", "agent": "rancher"}
        }
        mock_dispatch_event.return_value = [{"toolName": "show-yaml", "input": {}}]
        mock_llm = MagicMock()
        middleware = create_ui_tools_middleware(mock_llm, only_when_direct=True)
        ai_msg = AIMessage(content="answer")
        state = {"messages": [ai_msg]}

        result = middleware.after_agent(state, MagicMock())

        assert result is not None
        assert result["messages"][0].additional_kwargs["ui_tools"] == [
            {"toolName": "show-yaml", "input": {}}
        ]

    @patch("app.services.agent.middleware.ui_tools._dispatch_ui_tools_event")
    @patch("app.services.agent.middleware.ui_tools.get_config")
    def test_returns_none_when_no_ui_tools_selected(
        self, mock_get_config, mock_dispatch_event
    ):
        """Verify None returned when no UI tools are selected."""
        mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}
        mock_dispatch_event.return_value = []
        mock_llm = MagicMock()
        middleware = create_ui_tools_middleware(mock_llm, only_when_direct=False)
        state = {"messages": [AIMessage(content="answer")]}

        result = middleware.after_agent(state, MagicMock())

        assert result is None


# ============================================================================
# _collect_context_until_human Tests
# ============================================================================


class TestCollectContextUntilHuman:
    """Test _collect_context_until_human helper."""

    def test_collects_messages_back_to_human(self):
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

    def test_returns_empty_for_empty_messages(self):
        """Verify empty string for empty messages."""
        result = _collect_context_until_human({"messages": []})

        assert result == ""

    def test_includes_request_metadata_context(self):
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


# ============================================================================
# _extract_tool_text Tests
# ============================================================================


class TestExtractToolText:
    """Test _extract_tool_text helper."""

    def test_extracts_text_from_list_format(self):
        """Verify text extraction from list of items."""
        import json

        content = json.dumps([
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ])

        result = _extract_tool_text(content)

        assert "first" in result
        assert "second" in result

    def test_extracts_text_from_dict_with_text_key(self):
        """Verify text extraction from dict with 'text' key."""
        import json

        content = json.dumps({"text": "hello world"})

        result = _extract_tool_text(content)

        assert result == "hello world"

    def test_returns_json_string_for_dict_without_text(self):
        """Verify JSON dump for dict without 'text' key."""
        import json

        content = json.dumps({"key": "value"})

        result = _extract_tool_text(content)

        assert '"key"' in result
        assert '"value"' in result

    def test_returns_plain_string_for_non_json(self):
        """Verify plain string returned for non-JSON content."""
        result = _extract_tool_text("plain text result")

        assert result == "plain text result"
