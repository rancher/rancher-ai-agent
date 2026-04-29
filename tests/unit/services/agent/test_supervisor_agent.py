"""
Unit tests for the supervisor agent tool wrapping and interrupt/resume flow.

Tests that _create_agent_tool properly handles:
- Normal invocation (no interrupts) with a child-specific thread_id
- Interrupt detection and resume flow for human-in-the-loop
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

from app.services.agent.parent import _create_agent_tool, ChildAgent
from app.services.agent.loader import AgentConfig, AuthenticationType


@pytest.fixture
def agent_config():
    """Minimal AgentConfig for testing."""
    return AgentConfig(
        name="test-agent",
        displayName="Test Agent",
        description="A test agent for unit tests",
        system_prompt="You are a test agent.",
        mcp_url="http://localhost:9999/mcp",
        authentication=AuthenticationType.NONE,
    )


@pytest.fixture
def mock_compiled_graph():
    """Mock compiled graph with ainvoke and aget_state."""
    graph = AsyncMock()
    graph.ainvoke = AsyncMock()
    graph.aget_state = AsyncMock()
    return graph


@pytest.fixture
def child_agent(agent_config, mock_compiled_graph):
    """ChildAgent wrapping the mock compiled graph."""
    return ChildAgent(config=agent_config, agent=mock_compiled_graph)


# ============================================================================
# Normal invocation (no pending interrupts)
# ============================================================================

@pytest.mark.asyncio
async def test_invoke_normal_sends_messages_to_child(child_agent, mock_compiled_graph):
    """When no interrupt is pending, _invoke sends a fresh user message to the child."""
    # aget_state returns no interrupts
    mock_state = MagicMock()
    mock_state.interrupts = ()
    mock_compiled_graph.aget_state.return_value = mock_state

    # Child responds with an AI message
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="Hello from child")]
    }

    tool = _create_agent_tool(child_agent)

    # Patch ensure_config to provide a parent thread_id
    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "parent-thread-123"}
    }):
        result = await tool.ainvoke({"query": "test query"})

    assert result == "Hello from child"

    # Verify ainvoke was called with messages (not Command)
    call_args = mock_compiled_graph.ainvoke.call_args
    input_data = call_args[0][0]
    assert "messages" in input_data
    assert input_data["messages"][0]["content"] == "test query"

    # Verify child gets a derived thread_id
    config_arg = call_args[1].get("config") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["config"]
    assert "parent-thread-123::test-agent" in config_arg["configurable"]["thread_id"]


@pytest.mark.asyncio
async def test_invoke_normal_uses_derived_thread_id(child_agent, mock_compiled_graph):
    """The child thread_id is derived from parent thread_id + agent name."""
    mock_state = MagicMock()
    mock_state.interrupts = ()
    mock_compiled_graph.aget_state.return_value = mock_state
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="ok")]
    }

    tool = _create_agent_tool(child_agent)

    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "session-abc"}
    }):
        await tool.ainvoke({"query": "hello"})

    # Both aget_state and ainvoke should use the same derived thread_id
    state_config = mock_compiled_graph.aget_state.call_args[1]["config"]
    invoke_config = mock_compiled_graph.ainvoke.call_args[1]["config"]

    expected_thread_id = "session-abc::test-agent"
    assert state_config["configurable"]["thread_id"] == expected_thread_id
    assert invoke_config["configurable"]["thread_id"] == expected_thread_id


@pytest.mark.asyncio
async def test_invoke_normal_suppresses_callbacks(child_agent, mock_compiled_graph):
    """The child invocation uses empty callbacks to prevent event leakage."""
    mock_state = MagicMock()
    mock_state.interrupts = ()
    mock_compiled_graph.aget_state.return_value = mock_state
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="ok")]
    }

    tool = _create_agent_tool(child_agent)

    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "t1"}
    }):
        await tool.ainvoke({"query": "hello"})

    invoke_config = mock_compiled_graph.ainvoke.call_args[1]["config"]
    assert invoke_config["callbacks"] == []


@pytest.mark.asyncio
async def test_invoke_returns_last_ai_content(child_agent, mock_compiled_graph):
    """_invoke walks backwards to find the last AI message with content."""
    mock_state = MagicMock()
    mock_state.interrupts = ()
    mock_compiled_graph.aget_state.return_value = mock_state
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [
            AIMessage(content="first"),
            AIMessage(content=""),     # empty content — skip
            AIMessage(content="last"), # this should be returned
        ]
    }

    tool = _create_agent_tool(child_agent)

    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "t1"}
    }):
        result = await tool.ainvoke({"query": "hello"})

    assert result == "last"


@pytest.mark.asyncio
async def test_invoke_returns_fallback_when_no_content(child_agent, mock_compiled_graph):
    """_invoke returns a fallback when there are no messages with content."""
    mock_state = MagicMock()
    mock_state.interrupts = ()
    mock_compiled_graph.aget_state.return_value = mock_state
    mock_compiled_graph.ainvoke.return_value = {"messages": []}

    tool = _create_agent_tool(child_agent)

    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "t1"}
    }):
        result = await tool.ainvoke({"query": "hello"})

    assert result == "No response from agent."


# ============================================================================
# Interrupt/resume flow
# ============================================================================

@pytest.mark.asyncio
async def test_invoke_resume_detects_pending_interrupt(child_agent, mock_compiled_graph):
    """When a child has a pending interrupt, _invoke calls interrupt() and resumes the child."""
    # Set up aget_state to report a pending interrupt
    mock_interrupt = MagicMock()
    mock_interrupt.value = "<confirmation-response>approve creation</confirmation-response>"
    mock_state = MagicMock()
    mock_state.interrupts = (mock_interrupt,)
    mock_compiled_graph.aget_state.return_value = mock_state

    # After resume, child returns a result
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="Resource created successfully")]
    }

    tool = _create_agent_tool(child_agent)

    # Patch interrupt() to simulate returning the user's resume value
    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "parent-thread-456"}
    }), patch("app.services.agent.parent.langgraph.types.interrupt", return_value="yes") as mock_interrupt_fn:
        result = await tool.ainvoke({"query": "create resource"})

    assert result == "Resource created successfully"

    # Verify interrupt() was called with the child's interrupt value
    mock_interrupt_fn.assert_called_once_with(
        "<confirmation-response>approve creation</confirmation-response>"
    )

    # Verify ainvoke was called with Command(resume=...) not messages
    from langgraph.types import Command
    call_args = mock_compiled_graph.ainvoke.call_args
    input_data = call_args[0][0]
    assert isinstance(input_data, Command)


@pytest.mark.asyncio
async def test_invoke_resume_passes_user_response_to_child(child_agent, mock_compiled_graph):
    """The user's resume value is forwarded to the child graph via Command(resume=...)."""
    mock_interrupt = MagicMock()
    mock_interrupt.value = "<confirmation-response>some plan</confirmation-response>"
    mock_state = MagicMock()
    mock_state.interrupts = (mock_interrupt,)
    mock_compiled_graph.aget_state.return_value = mock_state

    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="done")]
    }

    tool = _create_agent_tool(child_agent)

    # User responds with "yes" to approve
    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "t1"}
    }), patch("app.services.agent.parent.langgraph.types.interrupt", return_value="yes"):
        await tool.ainvoke({"query": "do something"})

    # The resume value "yes" should be passed to the child
    from langgraph.types import Command
    call_args = mock_compiled_graph.ainvoke.call_args
    input_data = call_args[0][0]
    assert isinstance(input_data, Command)
    assert input_data.resume == "yes"


@pytest.mark.asyncio
async def test_invoke_resume_uses_same_thread_id(child_agent, mock_compiled_graph):
    """On resume the child uses the same derived thread_id as the initial call."""
    mock_interrupt = MagicMock()
    mock_interrupt.value = "confirm?"
    mock_state = MagicMock()
    mock_state.interrupts = (mock_interrupt,)
    mock_compiled_graph.aget_state.return_value = mock_state

    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="ok")]
    }

    tool = _create_agent_tool(child_agent)

    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "session-xyz"}
    }), patch("app.services.agent.parent.langgraph.types.interrupt", return_value="yes"):
        await tool.ainvoke({"query": "test"})

    expected_thread_id = "session-xyz::test-agent"

    # aget_state and ainvoke should share the same child thread_id
    state_config = mock_compiled_graph.aget_state.call_args[1]["config"]
    invoke_config = mock_compiled_graph.ainvoke.call_args[1]["config"]
    assert state_config["configurable"]["thread_id"] == expected_thread_id
    assert invoke_config["configurable"]["thread_id"] == expected_thread_id


# ============================================================================
# Edge cases
# ============================================================================

@pytest.mark.asyncio
async def test_invoke_fallback_thread_id_when_parent_missing(child_agent, mock_compiled_graph):
    """If parent config has no thread_id, the agent name is used as fallback."""
    mock_state = MagicMock()
    mock_state.interrupts = ()
    mock_compiled_graph.aget_state.return_value = mock_state
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="ok")]
    }

    tool = _create_agent_tool(child_agent)

    # ensure_config returns empty configurable
    with patch("app.services.agent.parent.ensure_config", return_value={}):
        await tool.ainvoke({"query": "hello"})

    state_config = mock_compiled_graph.aget_state.call_args[1]["config"]
    assert state_config["configurable"]["thread_id"] == "test-agent"


@pytest.mark.asyncio
async def test_invoke_no_interrupts_when_state_is_none(child_agent, mock_compiled_graph):
    """If aget_state returns None, treat as no pending interrupt."""
    mock_compiled_graph.aget_state.return_value = None
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="ok")]
    }

    tool = _create_agent_tool(child_agent)

    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "t1"}
    }):
        result = await tool.ainvoke({"query": "hello"})

    assert result == "ok"
    # Should have called ainvoke with messages, not Command
    input_data = mock_compiled_graph.ainvoke.call_args[0][0]
    assert "messages" in input_data
