"""
Unit tests for the ParentAgentBuilder class.

Tests the creation and behavior of parent agents that route requests to specialized child agents.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from app.services.agent.parent import ParentAgentBuilder, ChildAgent, create_parent_agent


@pytest.fixture
def mock_llm():
    """Mock LLM for routing decisions."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(return_value=AIMessage(content="Rancher"))
    return llm


@pytest.fixture
def mock_child_agent_graph():
    """Mock child agent graph."""
    graph = MagicMock()
    graph.invoke = MagicMock()
    graph.stream = MagicMock()
    return graph


@pytest.fixture
def mock_child_agents(mock_child_agent_graph):
    """Mock child agents for testing."""
    return [
        ChildAgent(
            name="Rancher",
            description="Expert in Rancher UI and Kubernetes management through Rancher",
            agent=mock_child_agent_graph
        ),
        ChildAgent(
            name="Fleet",
            description="Expert in Fleet GitOps continuous delivery for Kubernetes",
            agent=mock_child_agent_graph
        ),
        ChildAgent(
            name="Harvester",
            description="Expert in Harvester HCI and virtual machine management",
            agent=mock_child_agent_graph
        )
    ]


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for state persistence."""
    return MagicMock()


@pytest.fixture
def mock_config():
    """Mock runnable configuration."""
    return {
        "configurable": {
            "request_id": "test_id",
            "request_metadata": {"tags": []}
        }
    }


@pytest.fixture
def agent_state():
    """Mock agent state with messages."""
    return {
        "messages": [
            HumanMessage(content="How do I deploy a workload in Rancher?")
        ],
        "summary": ""
    }


# ============================================================================
# Builder Initialization Tests
# ============================================================================

def test_parent_agent_builder_initialization(mock_llm, mock_child_agents, mock_checkpointer):
    """Verify that ParentAgentBuilder correctly initializes with all required parameters."""
    builder = ParentAgentBuilder(
        llm=mock_llm,
        child_agents=mock_child_agents,
        checkpointer=mock_checkpointer
    )

    assert builder is not None
    assert builder.llm == mock_llm
    assert builder.child_agents == mock_child_agents
    assert builder.checkpointer == mock_checkpointer
    assert builder.tools == []
    assert builder.system_prompt == ""
    
    # Verify all child agents are stored correctly
    assert len(builder.child_agents) == 3
    assert builder.child_agents[0].name == "Rancher"
    assert builder.child_agents[1].name == "Fleet"
    assert builder.child_agents[2].name == "Harvester"


# ============================================================================
# Routing Logic Tests
# ============================================================================

@patch('app.services.agent.parent.dispatch_custom_event')
def test_choose_child_agent_uses_llm_for_routing(mock_dispatch, mock_llm, mock_child_agents, mock_checkpointer, agent_state, mock_config):
    """Verify that choose_child_agent uses LLM to select the appropriate child agent."""
    builder = ParentAgentBuilder(
        llm=mock_llm,
        child_agents=mock_child_agents,
        checkpointer=mock_checkpointer
    )

    result = builder.choose_child_agent(agent_state, mock_config)

    # Verify LLM was invoked
    mock_llm.invoke.assert_called_once()
    
    # Verify result is a Command
    assert isinstance(result, Command)
    assert result.goto == "Rancher"


@patch('app.services.agent.parent.dispatch_custom_event')
def test_choose_child_agent_includes_all_agents_in_prompt(mock_dispatch, mock_llm, mock_child_agents, mock_checkpointer, agent_state, mock_config):
    """Verify that routing prompt includes all available child agents."""
    builder = ParentAgentBuilder(
        llm=mock_llm,
        child_agents=mock_child_agents,
        checkpointer=mock_checkpointer
    )

    builder.choose_child_agent(agent_state, mock_config)

    # Check that LLM was invoked with a system message containing all child agents
    call_args = mock_llm.invoke.call_args[0][0]
    system_message = call_args[0]
    
    assert isinstance(system_message, SystemMessage)
    assert "Rancher" in system_message.content
    assert "Fleet" in system_message.content
    assert "Harvester" in system_message.content
    assert "Expert in Rancher UI" in system_message.content
    assert "Expert in Fleet GitOps" in system_message.content
    assert "Expert in Harvester HCI" in system_message.content


def test_choose_child_agent_with_agent_override(mock_llm, mock_child_agents, mock_checkpointer, agent_state):
    """Verify that agent override in config forces selection of specific agent."""
    builder = ParentAgentBuilder(
        llm=mock_llm,
        child_agents=mock_child_agents,
        checkpointer=mock_checkpointer
    )

    config_with_override = {
        "configurable": {
            "agent": "Fleet"
        }
    }

    result = builder.choose_child_agent(agent_state, config_with_override)

    # Verify LLM was NOT called when override is present
    mock_llm.invoke.assert_not_called()
    
    # Verify Fleet was selected
    assert isinstance(result, Command)
    assert result.goto == "Fleet"
    assert builder.agent_selected == "Fleet"


@patch('app.services.agent.parent.dispatch_custom_event')
def test_choose_child_agent_dispatches_event(mock_dispatch, mock_llm, mock_child_agents, mock_checkpointer, agent_state, mock_config):
    """Verify that choose_child_agent dispatches a custom event with agent selection."""
    mock_llm.invoke.return_value = AIMessage(content="Fleet")
    
    builder = ParentAgentBuilder(
        llm=mock_llm,
        child_agents=mock_child_agents,
        checkpointer=mock_checkpointer
    )

    builder.choose_child_agent(agent_state, mock_config)

    # Verify event was dispatched
    mock_dispatch.assert_called_once()
    call_args = mock_dispatch.call_args
    assert call_args[0][0] == "subagent_choice_event"
    assert "Fleet" in call_args[0][1]


@patch('app.services.agent.parent.dispatch_custom_event')
def test_choose_child_agent_with_conversation_context(mock_dispatch, mock_llm, mock_child_agents, mock_checkpointer, mock_config):
    """Verify that choose_child_agent includes conversation context in routing decision."""
    state_with_history = {
        "messages": [
            HumanMessage(content="What is Fleet?"),
            AIMessage(content="Fleet is a GitOps tool..."),
            HumanMessage(content="How do I use it with Rancher?")
        ],
        "summary": ""
    }
    
    builder = ParentAgentBuilder(
        llm=mock_llm,
        child_agents=mock_child_agents,
        checkpointer=mock_checkpointer
    )

    builder.choose_child_agent(state_with_history, mock_config)

    # Verify LLM was called with conversation history
    call_args = mock_llm.invoke.call_args[0][0]
    # Should have system message + conversation messages
    assert len(call_args) == 4  # 1 system + 3 messages


# ============================================================================
# Graph Building Tests
# ============================================================================

def test_parent_agent_graph_has_correct_nodes(mock_llm, mock_child_agents, mock_checkpointer):
    """Verify that the parent agent graph contains the expected nodes."""
    builder = ParentAgentBuilder(
        llm=mock_llm,
        child_agents=mock_child_agents,
        checkpointer=mock_checkpointer
    )

    graph = builder.build()

    nodes = graph.nodes
    # Should have routing node, summarize node, and all child agent nodes
    assert "choose_child_agent" in nodes
    assert "summarize_conversation" in nodes
    assert "Rancher" in nodes
    assert "Fleet" in nodes
    assert "Harvester" in nodes


# ============================================================================
# Factory Function Tests
# ============================================================================

def test_create_parent_agent_factory(mock_llm, mock_child_agents, mock_checkpointer):
    """Verify that create_parent_agent factory function creates a valid agent."""
    graph = create_parent_agent(
        llm=mock_llm,
        child_agents=mock_child_agents,
        checkpointer=mock_checkpointer
    )

    assert graph is not None
    assert hasattr(graph, 'invoke')
    assert hasattr(graph, 'stream')
    assert graph.checkpointer == mock_checkpointer

def test_create_parent_agent_with_many_children(mock_llm, mock_child_agent_graph, mock_checkpointer):
    """Verify that create_parent_agent works with multiple child agents."""
    many_children = [
        ChildAgent(name=f"Agent{i}", description=f"Agent {i} description", agent=mock_child_agent_graph)
        for i in range(5)
    ]
    
    graph = create_parent_agent(
        llm=mock_llm,
        child_agents=many_children,
        checkpointer=mock_checkpointer
    )

    assert graph is not None
    nodes = graph.nodes
    for i in range(5):
        assert f"Agent{i}" in nodes
