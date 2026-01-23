"""
Unit tests for the agent factory module.

Tests agent creation and MCP tool setup.
"""
import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from contextlib import AsyncExitStack

from app.services.agent.factory import (
    create_agent,
    _create_mcp_tools
)
from app.services.agent.loader import AuthenticationType


# ============================================================================
# create_agent Tests
# ============================================================================

@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory._create_mcp_tools')
@patch('app.services.agent.factory.create_root_agent')
async def test_create_agent_single_agent(mock_create_root, mock_create_tools, mock_load_configs):
    """Verify create_agent creates a root agent when one config is available."""
    # Setup mocks
    mock_llm = MagicMock()
    mock_websocket = MagicMock()
    mock_memory_manager = MagicMock()
    mock_checkpointer = MagicMock()
    mock_memory_manager.get_checkpointer.return_value = mock_checkpointer
    mock_websocket.app.memory_manager = mock_memory_manager
    
    mock_agent_config = MagicMock()
    mock_agent_config.name = "RancherAgent"
    mock_agent_config.system_prompt = "Test prompt"
    mock_load_configs.return_value = [mock_agent_config]
    
    mock_tools = [MagicMock()]
    mock_create_tools.return_value = mock_tools
    
    mock_agent = MagicMock()
    mock_create_root.return_value = mock_agent
    
    # Execute
    async with create_agent(mock_llm, mock_websocket) as agent:
        # Verify
        assert agent == mock_agent
        mock_create_root.assert_called_once_with(
            mock_llm, 
            mock_tools, 
            "Test prompt", 
            mock_checkpointer, 
            mock_agent_config
        )


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory._create_mcp_tools')
@patch('app.services.agent.factory.create_child_agent')
@patch('app.services.agent.factory.create_parent_agent')
async def test_create_agent_multiple_agents(mock_create_parent, mock_create_child, mock_create_tools, mock_load_configs):
    """Verify create_agent creates a parent agent when multiple configs are available."""
    # Setup mocks
    mock_llm = MagicMock()
    mock_websocket = MagicMock()
    mock_memory_manager = MagicMock()
    mock_checkpointer = MagicMock()
    mock_memory_manager.get_checkpointer.return_value = mock_checkpointer
    mock_websocket.app.memory_manager = mock_memory_manager
    
    mock_config1 = MagicMock()
    mock_config1.name = "Agent1"
    mock_config1.description = "First agent"
    mock_config1.system_prompt = "Prompt 1"
    
    mock_config2 = MagicMock()
    mock_config2.name = "Agent2"
    mock_config2.description = "Second agent"
    mock_config2.system_prompt = "Prompt 2"
    
    mock_load_configs.return_value = [mock_config1, mock_config2]
    
    mock_tools = [MagicMock()]
    mock_create_tools.return_value = mock_tools
    
    mock_child_agent = MagicMock()
    mock_create_child.return_value = mock_child_agent
    
    mock_parent_agent = MagicMock()
    mock_create_parent.return_value = mock_parent_agent
    
    # Execute
    async with create_agent(mock_llm, mock_websocket) as agent:
        # Verify
        assert agent == mock_parent_agent
        assert mock_create_child.call_count == 2
        mock_create_parent.assert_called_once()
        
        # Verify parent was called with correct child agents
        call_args = mock_create_parent.call_args
        assert call_args[0][0] == mock_llm
        child_agents = call_args[0][1]
        assert len(child_agents) == 2
        assert child_agents[0].name == "Agent1"
        assert child_agents[1].name == "Agent2"


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
async def test_create_agent_no_configs_raises_error(mock_load_configs):
    """Verify create_agent raises RuntimeError when no configs are available."""
    mock_llm = MagicMock()
    mock_websocket = MagicMock()
    mock_memory_manager = MagicMock()
    mock_websocket.app.memory_manager = mock_memory_manager
    
    mock_load_configs.return_value = []
    
    with pytest.raises(RuntimeError) as exc_info:
        async with create_agent(mock_llm, mock_websocket) as agent:
            pass
    
    assert "No agent configurations available" in str(exc_info.value)


# ============================================================================
# _create_mcp_tools Tests
# ============================================================================

@pytest.mark.asyncio
@patch('app.services.agent.factory.streamablehttp_client')
@patch('app.services.agent.factory.load_mcp_tools')
@patch('app.services.agent.factory.ClientSession')
async def test_create_mcp_tools_basic(mock_session_class, mock_load_tools, mock_http_client):
    """Verify _create_mcp_tools creates tools from MCP server."""
    stack = AsyncExitStack()
    mock_websocket = MagicMock()
    
    mock_config = MagicMock()
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "http://test:8080"
    mock_config.name = "TestAgent"
    mock_config.toolset = None
    
    mock_read = MagicMock()
    mock_write = MagicMock()
    mock_http_client.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
    mock_http_client.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()
    mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_tools = [MagicMock(), MagicMock()]
    mock_load_tools.return_value = mock_tools
    
    async with stack:
        tools = await _create_mcp_tools(stack, mock_websocket, mock_config)
    
    assert tools == mock_tools
    mock_load_tools.assert_called_once_with(mock_session)


@pytest.mark.asyncio
@patch('app.services.agent.factory.streamablehttp_client')
@patch('app.services.agent.factory.load_mcp_tools')
@patch('app.services.agent.factory.ClientSession')
@patch.dict(os.environ, {'RANCHER_URL': 'https://rancher.example.com', 'RANCHER_API_TOKEN': 'test-token'})
async def test_create_mcp_tools_with_rancher_auth(mock_session_class, mock_load_tools, mock_http_client):
    """Verify _create_mcp_tools handles Rancher authentication correctly."""
    stack = AsyncExitStack()
    mock_websocket = MagicMock()
    mock_websocket.cookies = {"R_SESS": "cookie-token"}
    mock_websocket.url.hostname = "rancher.local"
    
    mock_config = MagicMock()
    mock_config.authentication = AuthenticationType.RANCHER
    mock_config.mcp_url = "mcp-service:8080"
    mock_config.name = "RancherAgent"
    
    mock_read = MagicMock()
    mock_write = MagicMock()
    mock_http_client.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
    mock_http_client.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()
    mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_tools = [MagicMock()]
    mock_load_tools.return_value = mock_tools
    
    async with stack:
        tools = await _create_mcp_tools(stack, mock_websocket, mock_config)
    
    # Verify headers were set correctly
    call_args = mock_http_client.call_args
    assert call_args[1]['headers']['R_token'] == 'test-token'
    assert call_args[1]['headers']['R_url'] == 'https://rancher.example.com'


@pytest.mark.asyncio
@patch('app.services.agent.factory.streamablehttp_client')
@patch('app.services.agent.factory.load_mcp_tools')
@patch('app.services.agent.factory.ClientSession')
async def test_create_mcp_tools_connection_failure(mock_session_class, mock_load_tools, mock_http_client):
    """Verify _create_mcp_tools handles MCP connection failures gracefully."""
    stack = AsyncExitStack()
    mock_websocket = MagicMock()
    
    mock_config = MagicMock()
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "http://test:8080"
    mock_config.name = "TestAgent"
    
    mock_read = MagicMock()
    mock_write = MagicMock()
    mock_http_client.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
    mock_http_client.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_session = MagicMock()
    mock_session.initialize = AsyncMock(side_effect=Exception("Connection failed"))
    mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
    
    async with stack:
        tools = await _create_mcp_tools(stack, mock_websocket, mock_config)
    
    # Should return empty list on failure
    assert tools == []

@pytest.mark.asyncio
@patch('app.services.agent.factory.streamablehttp_client')
@patch('app.services.agent.factory.load_mcp_tools')
@patch('app.services.agent.factory.ClientSession')
@patch.dict(os.environ, {'ENABLE_RAG': 'true'})
async def test_create_mcp_tools_rag_only_for_rancher_core(mock_session_class, mock_load_tools, mock_http_client):
    """Verify RAG retrievers are only added for Rancher Core Agent, not other agents."""
    stack = AsyncExitStack()
    mock_websocket = MagicMock()
    
    mock_config = MagicMock()
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "http://test:8080"
    mock_config.name = "Fleet Agent"  # Not Rancher Core Agent
    mock_config.toolset = None
    
    mock_read = MagicMock()
    mock_write = MagicMock()
    mock_http_client.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
    mock_http_client.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()
    mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_mcp_tools = [MagicMock()]
    mock_load_tools.return_value = mock_mcp_tools
    
    async with stack:
        tools = await _create_mcp_tools(stack, mock_websocket, mock_config)
    
    # Should only have MCP tools since agent is not Rancher Core Agent
    assert len(tools) == 1
    assert tools == mock_mcp_tools
