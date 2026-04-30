"""
Unit tests for the supervisor agent.

Tests:
- create_supervisor_agent factory function
- Helper functions (_build_child_config, _extract_last_message)
- _create_agent_tool wrapping and interrupt/resume flow for human-in-the-loop
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

from app.services.agent.parent import (
    ChildAgent,
    create_supervisor_agent,
    _build_child_config,
    _extract_last_message,
    _create_agent_tool,
)
from app.services.agent.loader import AgentConfig, AuthenticationType


@pytest.fixture
def mock_llm():
    """Mock LLM for routing decisions."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for state persistence."""
    return InMemorySaver()


@pytest.fixture
def mock_child_agents():
    """Mock ChildAgent list for testing."""
    agents = []
    for name, desc in [
        ("Rancher", "Expert in Rancher and Kubernetes management"),
        ("Fleet", "Expert in Fleet GitOps continuous delivery"),
        ("Harvester", "Expert in Harvester HCI and VM management"),
    ]:
        config = MagicMock()
        config.name = name
        config.description = desc
        agent = MagicMock()
        agents.append(ChildAgent(config=config, agent=agent))
    return agents


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
# Factory Function Tests
# ============================================================================

@patch("app.services.agent.parent.create_agent")
def test_create_supervisor_agent_creates_graph(mock_create_agent, mock_llm, mock_child_agents, mock_checkpointer):
    """Verify that create_supervisor_agent delegates to langchain create_agent."""
    mock_graph = MagicMock()
    mock_create_agent.return_value = mock_graph

    result = create_supervisor_agent(mock_llm, mock_child_agents, mock_checkpointer)

    assert result == mock_graph
    mock_create_agent.assert_called_once()


@patch("app.services.agent.parent.create_agent")
def test_create_supervisor_agent_creates_tools_for_all_children(mock_create_agent, mock_llm, mock_child_agents, mock_checkpointer):
    """Verify that each child agent becomes a tool."""
    mock_create_agent.return_value = MagicMock()

    create_supervisor_agent(mock_llm, mock_child_agents, mock_checkpointer)

    call_kwargs = mock_create_agent.call_args[1]
    tools = call_kwargs["tools"]
    assert len(tools) == 3
    tool_names = [t.name for t in tools]
    assert "Rancher" in tool_names
    assert "Fleet" in tool_names
    assert "Harvester" in tool_names


@patch("app.services.agent.parent.create_agent")
def test_create_supervisor_agent_with_many_agents(mock_create_agent, mock_llm, mock_checkpointer):
    """Verify that create_supervisor_agent works with many child agents."""
    mock_create_agent.return_value = MagicMock()

    child_agents = []
    for i in range(5):
        config = MagicMock()
        config.name = f"Agent{i}"
        config.description = f"Agent {i} description"
        child_agents.append(ChildAgent(config=config, agent=MagicMock()))

    create_supervisor_agent(mock_llm, child_agents, mock_checkpointer)

    call_kwargs = mock_create_agent.call_args[1]
    assert len(call_kwargs["tools"]) == 5


# ============================================================================
# Helper Function Tests
# ============================================================================

def test_build_child_config_namespaces_thread_id():
    """Verify child config has namespaced thread_id."""
    with patch("app.services.agent.parent.ensure_config") as mock_ensure:
        mock_ensure.return_value = {
            "configurable": {
                "thread_id": "parent-thread-1",
                "request_id": "req-123",
            }
        }
        config = _build_child_config("rancher")
        assert config["configurable"]["thread_id"] == "parent-thread-1::rancher"
        assert config["configurable"]["request_id"] == "req-123"
        assert config["callbacks"] == []


def test_extract_last_message_returns_content():
    """Verify extraction of last non-empty message content."""
    msg = MagicMock()
    msg.content = "Hello from agent"
    result = _extract_last_message({"messages": [msg]})
    assert result == "Hello from agent"


def test_extract_last_message_empty():
    """Verify fallback when no messages."""
    result = _extract_last_message({"messages": []})
    assert result == "No response from agent."


def test_create_agent_tool_returns_structured_tool():
    """Verify _create_agent_tool wraps child agent as a StructuredTool."""
    config = MagicMock()
    config.name = "rancher"
    config.description = "Rancher agent"
    child = ChildAgent(config=config, agent=MagicMock())

    tool = _create_agent_tool(child)

    assert tool.name == "rancher"
    assert "Rancher agent" in tool.description


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
    # Set up aget_state: first call returns pending interrupt, second (after
    # resume ainvoke) returns no interrupts (child completed successfully).
    mock_interrupt = MagicMock()
    mock_interrupt.value = "<confirmation-response>approve creation</confirmation-response>"
    mock_state_with_interrupt = MagicMock()
    mock_state_with_interrupt.interrupts = (mock_interrupt,)
    mock_state_completed = MagicMock()
    mock_state_completed.interrupts = ()
    mock_compiled_graph.aget_state.side_effect = [
        mock_state_with_interrupt,  # Before invocation: has pending interrupt
        mock_state_completed,       # After invocation: child completed
    ]

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
    mock_state_with_interrupt = MagicMock()
    mock_state_with_interrupt.interrupts = (mock_interrupt,)
    mock_state_completed = MagicMock()
    mock_state_completed.interrupts = ()
    mock_compiled_graph.aget_state.side_effect = [
        mock_state_with_interrupt,
        mock_state_completed,
    ]

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
    mock_state_with_interrupt = MagicMock()
    mock_state_with_interrupt.interrupts = (mock_interrupt,)
    mock_state_completed = MagicMock()
    mock_state_completed.interrupts = ()
    mock_compiled_graph.aget_state.side_effect = [
        mock_state_with_interrupt,
        mock_state_completed,
    ]

    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="ok")]
    }

    tool = _create_agent_tool(child_agent)

    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "session-xyz"}
    }), patch("app.services.agent.parent.langgraph.types.interrupt", return_value="yes"):
        await tool.ainvoke({"query": "test"})

    expected_thread_id = "session-xyz::test-agent"

    # Both aget_state calls and ainvoke should share the same child thread_id
    aget_state_calls = mock_compiled_graph.aget_state.call_args_list
    for call in aget_state_calls:
        assert call[1]["config"]["configurable"]["thread_id"] == expected_thread_id
    invoke_config = mock_compiled_graph.ainvoke.call_args[1]["config"]
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


# ============================================================================
# Child interrupted during invocation (the core bug fix)
# ============================================================================

@pytest.mark.asyncio
async def test_invoke_normal_child_interrupts_during_invocation(child_agent, mock_compiled_graph):
    """When a child triggers interrupt() during normal invocation, _invoke
    re-triggers interrupt at supervisor level so the client receives the prompt.

    Because the child graph is called via ainvoke() (not as a proper LangGraph
    subgraph), its GraphInterrupt is suppressed internally (is_nested=False) and
    ainvoke() returns normally.  _invoke must detect the pending interrupt and
    call interrupt() at the supervisor level.
    """
    mock_interrupt = MagicMock()
    mock_interrupt.value = "<confirmation-response>create resource plan</confirmation-response>"

    mock_state_no_interrupt = MagicMock()
    mock_state_no_interrupt.interrupts = ()

    mock_state_with_interrupt = MagicMock()
    mock_state_with_interrupt.interrupts = (mock_interrupt,)

    # First aget_state: no pending interrupts (fresh invocation)
    # After ainvoke: child was interrupted (has pending interrupt)
    mock_compiled_graph.aget_state.side_effect = [
        mock_state_no_interrupt,    # Before invocation
        mock_state_with_interrupt,  # After invocation (child interrupted)
    ]

    # ainvoke returns partial output (interrupt was suppressed by child runtime)
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="partial output")]
    }

    tool = _create_agent_tool(child_agent)

    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "parent-thread-789"}
    }), patch("app.services.agent.parent.langgraph.types.interrupt") as mock_interrupt_fn:
        # interrupt() at the supervisor level raises GraphInterrupt,
        # but our mock just records the call and returns (won't raise).
        # In production this would propagate the interrupt to the client.
        mock_interrupt_fn.return_value = None  # simulate the raise path
        await tool.ainvoke({"query": "create resource"})

    # Verify interrupt() was re-triggered at supervisor level with child's value
    mock_interrupt_fn.assert_called_once_with(
        "<confirmation-response>create resource plan</confirmation-response>"
    )


@pytest.mark.asyncio
async def test_invoke_resume_child_interrupts_again(child_agent, mock_compiled_graph):
    """When a child triggers a second interrupt during resume (e.g. multiple
    tools requiring validation), _invoke re-triggers interrupt at supervisor level."""
    mock_interrupt_first = MagicMock()
    mock_interrupt_first.value = "<confirmation-response>first tool plan</confirmation-response>"
    mock_interrupt_second = MagicMock()
    mock_interrupt_second.value = "<confirmation-response>second tool plan</confirmation-response>"

    mock_state_first_interrupt = MagicMock()
    mock_state_first_interrupt.interrupts = (mock_interrupt_first,)
    mock_state_second_interrupt = MagicMock()
    mock_state_second_interrupt.interrupts = (mock_interrupt_second,)

    # First aget_state: has pending interrupt (resume path)
    # After resume ainvoke: child was interrupted again (second tool)
    mock_compiled_graph.aget_state.side_effect = [
        mock_state_first_interrupt,   # Before invocation (triggers resume)
        mock_state_second_interrupt,  # After invocation (interrupted again)
    ]

    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="partial after first approve")]
    }

    tool = _create_agent_tool(child_agent)

    interrupt_calls = []
    def mock_interrupt_side_effect(value):
        interrupt_calls.append(value)
        if len(interrupt_calls) == 1:
            return "yes"  # First call: user approved first interrupt
        return None  # Second call: re-trigger for the second interrupt

    with patch("app.services.agent.parent.ensure_config", return_value={
        "configurable": {"thread_id": "parent-thread-multi"}
    }), patch("app.services.agent.parent.langgraph.types.interrupt", side_effect=mock_interrupt_side_effect):
        await tool.ainvoke({"query": "multi-tool operation"})

    # interrupt() should have been called twice:
    # 1. To consume resume value for the first interrupt
    # 2. To re-trigger the second interrupt at supervisor level
    assert len(interrupt_calls) == 2
    assert interrupt_calls[0] == "<confirmation-response>first tool plan</confirmation-response>"
    assert interrupt_calls[1] == "<confirmation-response>second tool plan</confirmation-response>"
