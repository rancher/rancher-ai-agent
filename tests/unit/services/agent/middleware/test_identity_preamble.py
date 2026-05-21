"""Unit tests for identity_preamble middleware (identity_preamble_middleware)."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import SystemMessage

from app.services.agent.middleware.identity_preamble import identity_preamble_middleware
from app.services.agent.system_prompts import IDENTITY_PREAMBLE


@patch("app.services.agent.middleware.identity_preamble.get_config")
def test_injects_preamble_when_agent_is_set(mock_get_config):
    """Verify IDENTITY_PREAMBLE is injected when agent key is in config."""
    mock_get_config.return_value = {"configurable": {"agent": "rancher"}}

    middleware = identity_preamble_middleware()
    state = {"messages": []}

    result = middleware.before_model(state, MagicMock())

    assert result is not None
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], SystemMessage)
    assert result["messages"][0].content == IDENTITY_PREAMBLE


@patch("app.services.agent.middleware.identity_preamble.get_config")
def test_identity_preamble_returns_none_when_no_agent_in_config(mock_get_config):
    """Verify None returned when agent is not set (invoked by supervisor)."""
    mock_get_config.return_value = {"configurable": {"request_id": "req-1"}}

    middleware = identity_preamble_middleware()
    state = {"messages": []}

    result = middleware.before_model(state, MagicMock())

    assert result is None


@patch("app.services.agent.middleware.identity_preamble.get_config")
def test_identity_preamble_returns_none_when_configurable_is_empty(mock_get_config):
    """Verify None returned when configurable dict is empty."""
    mock_get_config.return_value = {"configurable": {}}

    middleware = identity_preamble_middleware()
    state = {"messages": []}

    result = middleware.before_model(state, MagicMock())

    assert result is None
