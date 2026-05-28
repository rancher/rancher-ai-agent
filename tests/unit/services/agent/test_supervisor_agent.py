"""
Unit tests for the supervisor agent.

"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

from app.services.agent.supervisor import (
    ChildAgent,
    create_supervisor_agent,
    _build_child_config,
    _extract_last_message,
    _create_agent_tool,
    _AgentCallCounter,
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

@patch("app.services.agent.supervisor.create_agent")
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


@patch("app.services.agent.supervisor.create_agent")
def test_create_supervisor_agent_registers_expected_middleware(mock_create_agent, mock_llm, mock_child_agents, mock_checkpointer):
    """Verify the supervisor registers the expected middleware stack."""
    from app.services.agent.middleware import (
        MessagesHistoryMiddleware,
    )
    from langchain.agents.middleware import SummarizationMiddleware

    mock_create_agent.return_value = MagicMock()

    create_supervisor_agent(mock_llm, mock_child_agents, mock_checkpointer)

    call_kwargs = mock_create_agent.call_args[1]
    middleware = call_kwargs["middleware"]

    # Check expected middleware types are present
    middleware_types = [type(m).__name__ for m in middleware]
    assert "MessagesHistoryMiddleware" in middleware_types
    assert "SummarizationMiddleware" in middleware_types

    # Decorator-based middleware are plain functions, check count
    assert len(middleware) == 5


# ============================================================================
# Helper Function Tests
# ============================================================================

def test_build_child_config_namespaces_thread_id():
    """Verify child config has namespaced thread_id."""
    with patch("app.services.agent.supervisor.ensure_config") as mock_ensure:
        mock_ensure.return_value = {
            "configurable": {
                "thread_id": "parent-thread-1",
                "request_id": "req-123",
            }
        }
        config = _build_child_config("rancher")
        assert config.get("configurable", {}).get("thread_id") == "parent-thread-1::child::rancher"
        assert config.get("configurable", {}).get("request_id") == "req-123"
        assert config.get("callbacks") == []


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

    tool = _create_agent_tool(child, _AgentCallCounter())

    assert tool.name == "rancher"
    assert "Rancher agent" in tool.description


# ============================================================================
# Normal invocation (no pending interrupts)
# ============================================================================

@pytest.mark.asyncio
async def test_invoke_normal_sends_messages_to_child(child_agent, mock_compiled_graph):
    """When no interrupt is pending, _invoke sends a fresh user message to the child."""
    mock_state = MagicMock()
    mock_state.interrupts = ()
    mock_compiled_graph.aget_state.return_value = mock_state
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="Hello from child")]
    }

    tool = _create_agent_tool(child_agent, _AgentCallCounter())

    with patch("app.services.agent.supervisor.ensure_config", return_value={
        "configurable": {"thread_id": "parent-thread-123"}
    }):
        result = await tool.ainvoke({"query": "test query"})

    assert result == "Hello from child"

    call_args = mock_compiled_graph.ainvoke.call_args
    input_data = call_args[0][0]
    assert "messages" in input_data
    assert input_data["messages"][0]["content"] == "test query"


@pytest.mark.asyncio
async def test_invoke_normal_uses_derived_thread_id(child_agent, mock_compiled_graph):
    """The child thread_id is derived from parent thread_id + agent name."""
    mock_state = MagicMock()
    mock_state.interrupts = ()
    mock_compiled_graph.aget_state.return_value = mock_state
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="ok")]
    }

    tool = _create_agent_tool(child_agent, _AgentCallCounter())

    with patch("app.services.agent.supervisor.ensure_config", return_value={
        "configurable": {"thread_id": "session-abc"}
    }):
        await tool.ainvoke({"query": "hello"})

    state_config = mock_compiled_graph.aget_state.call_args[1]["config"]
    invoke_config = mock_compiled_graph.ainvoke.call_args[1]["config"]

    expected_thread_id = "session-abc::child::test-agent"
    assert state_config["configurable"]["thread_id"] == expected_thread_id
    assert invoke_config["configurable"]["thread_id"] == expected_thread_id


@pytest.mark.asyncio
async def test_invoke_returns_last_ai_content(child_agent, mock_compiled_graph):
    """_invoke walks backwards to find the last AI message with content."""
    mock_state = MagicMock()
    mock_state.interrupts = ()
    mock_compiled_graph.aget_state.return_value = mock_state
    mock_compiled_graph.ainvoke.return_value = {
        "messages": [
            AIMessage(content="first"),
            AIMessage(content=""),
            AIMessage(content="last"),
        ]
    }

    tool = _create_agent_tool(child_agent, _AgentCallCounter())

    with patch("app.services.agent.supervisor.ensure_config", return_value={
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

    tool = _create_agent_tool(child_agent, _AgentCallCounter())

    with patch("app.services.agent.supervisor.ensure_config", return_value={
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
    mock_interrupt = MagicMock()
    mock_interrupt.value = "<confirmation-response>approve creation</confirmation-response>"
    mock_state_with_interrupt = MagicMock()
    mock_state_with_interrupt.interrupts = (mock_interrupt,)
    mock_state_completed = MagicMock()
    mock_state_completed.interrupts = ()
    mock_compiled_graph.aget_state.side_effect = [
        mock_state_with_interrupt,
        mock_state_completed,
    ]

    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="Resource created successfully")]
    }

    tool = _create_agent_tool(child_agent, _AgentCallCounter())

    with patch("app.services.agent.supervisor.ensure_config", return_value={
        "configurable": {"thread_id": "parent-thread-456"}
    }), patch("app.services.agent.supervisor.langgraph.types.interrupt", return_value="yes") as mock_interrupt_fn:
        result = await tool.ainvoke({"query": "create resource"})

    assert result == "Resource created successfully"
    mock_interrupt_fn.assert_called_once_with(
        "<confirmation-response>approve creation</confirmation-response>"
    )

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

    tool = _create_agent_tool(child_agent, _AgentCallCounter())

    with patch("app.services.agent.supervisor.ensure_config", return_value={
        "configurable": {"thread_id": "t1"}
    }), patch("app.services.agent.supervisor.langgraph.types.interrupt", return_value="yes"):
        await tool.ainvoke({"query": "do something"})

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

    tool = _create_agent_tool(child_agent, _AgentCallCounter())

    with patch("app.services.agent.supervisor.ensure_config", return_value={
        "configurable": {"thread_id": "session-xyz"}
    }), patch("app.services.agent.supervisor.langgraph.types.interrupt", return_value="yes"):
        await tool.ainvoke({"query": "test"})

    expected_thread_id = "session-xyz::child::test-agent"

    aget_state_calls = mock_compiled_graph.aget_state.call_args_list
    for call in aget_state_calls:
        assert call[1]["config"]["configurable"]["thread_id"] == expected_thread_id
    invoke_config = mock_compiled_graph.ainvoke.call_args[1]["config"]
    assert invoke_config["configurable"]["thread_id"] == expected_thread_id


@pytest.mark.asyncio
async def test_invoke_normal_child_interrupts_during_invocation(child_agent, mock_compiled_graph):
    """When a child triggers interrupt() during normal invocation, _invoke
    re-triggers interrupt at supervisor level so the client receives the prompt."""
    mock_interrupt = MagicMock()
    mock_interrupt.value = "<confirmation-response>create resource plan</confirmation-response>"

    mock_state_no_interrupt = MagicMock()
    mock_state_no_interrupt.interrupts = ()

    mock_state_with_interrupt = MagicMock()
    mock_state_with_interrupt.interrupts = (mock_interrupt,)

    mock_compiled_graph.aget_state.side_effect = [
        mock_state_no_interrupt,
        mock_state_with_interrupt,
    ]

    mock_compiled_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="partial output")]
    }

    tool = _create_agent_tool(child_agent, _AgentCallCounter())

    with patch("app.services.agent.supervisor.ensure_config", return_value={
        "configurable": {"thread_id": "parent-thread-789"}
    }), patch("app.services.agent.supervisor.langgraph.types.interrupt") as mock_interrupt_fn:
        mock_interrupt_fn.return_value = None
        await tool.ainvoke({"query": "create resource"})

    mock_interrupt_fn.assert_called_once_with(
        "<confirmation-response>create resource plan</confirmation-response>"
    )
