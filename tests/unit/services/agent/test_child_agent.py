"""
Unit tests for the child agent module (app.services.agent.child).

Tests: create_child_agent factory, middleware registration, tool filtering.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.tools import tool as langchain_tool

from app.services.agent.child import create_child_agent


def _make_langchain_tool(name: str):
    """Create a real langchain tool for tests that require it."""
    @langchain_tool(name)
    def _tool(x: str = "") -> str:
        """A test tool."""
        return "result"
    return _tool


@pytest.fixture
def mock_llm():
    """Mock LLM with bound tools."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for state persistence."""
    return False


@pytest.fixture
def agent_config():
    """Minimal AgentConfig mock."""
    config = MagicMock()
    config.human_validation_tools = []
    config.llm_model_enabled = False
    config.llm_model = None
    return config


# ============================================================================
# create_child_agent Tests
# ============================================================================


@patch("app.services.agent.child.create_agent")
def test_create_child_agent_excludes_plan_tools_from_execution(mock_create_agent, mock_llm, mock_checkpointer, agent_config):
    """Verify Plan tools are excluded from the execution tool list."""
    mock_create_agent.return_value = MagicMock()

    tools = [
        _make_langchain_tool("createPod"),
        _make_langchain_tool("createPodPlan"),
        _make_langchain_tool("listPods"),
    ]

    create_child_agent(
        llm=mock_llm,
        tools=tools,
        checkpointer=mock_checkpointer,
        agent_config=agent_config,
    )

    call_kwargs = mock_create_agent.call_args[1]
    execution_tools = call_kwargs["tools"]
    tool_names = [t.name for t in execution_tools]

    assert "createPod" in tool_names
    assert "listPods" in tool_names
    assert "createPodPlan" not in tool_names


@patch("app.services.agent.child.create_agent")
def test_create_child_agent_registers_expected_middleware(mock_create_agent, mock_llm, mock_checkpointer, agent_config):
    """Verify the child agent registers the expected middleware stack."""
    from app.services.agent.middleware import MessagesHistoryMiddleware

    mock_create_agent.return_value = MagicMock()

    tools = [_make_langchain_tool("testTool")]

    create_child_agent(
        llm=mock_llm,
        tools=tools,
        checkpointer=mock_checkpointer,
        agent_config=agent_config,
    )

    call_kwargs = mock_create_agent.call_args[1]
    middleware = call_kwargs["middleware"]

    middleware_types = [type(m).__name__ for m in middleware]
    assert "MessagesHistoryMiddleware" in middleware_types
    assert "UsageTrackingSummarizationMiddleware" in middleware_types

    # 7 middleware total: MessagesHistory, human_validation, identity_preamble,
    # cancel_check, inject_kwargs, ui_tools, Summarization
    assert len(middleware) == 7


@patch("app.services.agent.child.create_agent")
def test_create_child_agent_appends_tool_use_instructions_to_prompt(mock_create_agent, mock_llm, mock_checkpointer, agent_config):
    """Verify CHILD_TOOL_USE_INSTRUCTIONS is appended to the system prompt."""
    from app.services.agent.child import CHILD_TOOL_USE_INSTRUCTIONS

    mock_create_agent.return_value = MagicMock()

    tools = [_make_langchain_tool("testTool")]

    agent_config.system_prompt = "Base prompt."

    create_child_agent(
        llm=mock_llm,
        tools=tools,
        checkpointer=mock_checkpointer,
        agent_config=agent_config,
    )

    call_kwargs = mock_create_agent.call_args[1]
    system_prompt = call_kwargs["system_prompt"]

    assert system_prompt.startswith("Base prompt.")
    assert CHILD_TOOL_USE_INSTRUCTIONS in system_prompt


@patch("app.services.agent.child.get_llm_with_model")
@patch("app.services.agent.child.create_agent")
def test_create_child_agent_uses_custom_llm_when_enabled(mock_create_agent, mock_get_llm_with_model, mock_llm, mock_checkpointer, agent_config):
    """Verify create_child_agent uses per-agent LLM when llm_model_enabled is True."""
    mock_create_agent.return_value = MagicMock()
    custom_llm = MagicMock()
    mock_get_llm_with_model.return_value = custom_llm

    agent_config.llm_model_enabled = True
    agent_config.llm_model = "gemini-2.0-flash"

    tools = [_make_langchain_tool("testTool")]

    create_child_agent(
        llm=mock_llm,
        tools=tools,
        checkpointer=mock_checkpointer,
        agent_config=agent_config,
    )

    mock_get_llm_with_model.assert_called_once_with("gemini-2.0-flash")
    call_args = mock_create_agent.call_args
    assert call_args[0][0] == custom_llm


@patch("app.services.agent.child.get_llm_with_model")
@patch("app.services.agent.child.create_agent")
def test_create_child_agent_uses_default_llm_when_override_disabled(mock_create_agent, mock_get_llm_with_model, mock_llm, mock_checkpointer, agent_config):
    """Verify create_child_agent uses the default LLM when llm_model_enabled is False."""
    mock_create_agent.return_value = MagicMock()

    agent_config.llm_model_enabled = False
    agent_config.llm_model = "gemini-2.0-flash"

    tools = [_make_langchain_tool("testTool")]

    create_child_agent(
        llm=mock_llm,
        tools=tools,
        checkpointer=mock_checkpointer,
        agent_config=agent_config,
    )

    mock_get_llm_with_model.assert_not_called()
    call_args = mock_create_agent.call_args
    assert call_args[0][0] == mock_llm


@patch("app.services.agent.child.get_llm_with_model")
@patch("app.services.agent.child.create_agent")
def test_create_child_agent_uses_default_llm_when_model_not_set(mock_create_agent, mock_get_llm_with_model, mock_llm, mock_checkpointer, agent_config):
    """Verify create_child_agent uses the default LLM when llm_model is None."""
    mock_create_agent.return_value = MagicMock()

    agent_config.llm_model_enabled = True
    agent_config.llm_model = None

    tools = [_make_langchain_tool("testTool")]

    create_child_agent(
        llm=mock_llm,
        tools=tools,
        checkpointer=mock_checkpointer,
        agent_config=agent_config,
    )

    mock_get_llm_with_model.assert_not_called()
    call_args = mock_create_agent.call_args
    assert call_args[0][0] == mock_llm