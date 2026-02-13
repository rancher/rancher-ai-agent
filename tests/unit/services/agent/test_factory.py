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
    NoAgentAvailableError,
    create_mcp_client,
    _create_single_agent
)
from app.services.agent.loader import AuthenticationType


# ============================================================================
# create_agent Tests
# ============================================================================

@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory._create_single_agent')
async def test_create_agent_single_agent(mock_create_single, mock_load_configs):
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
    
    mock_agent = MagicMock()
    mock_metadata = [{"name": "RancherAgent", "status": "active"}]
    mock_create_single.return_value = mock_agent
    
    # Execute
    result = await create_agent(mock_llm, mock_websocket)
    
    # Verify
    assert result[0] == mock_agent
    assert result[1] == mock_metadata
    mock_create_single.assert_called_once_with(
        mock_llm, 
        mock_agent_config,
        mock_checkpointer, 
        mock_websocket
    )


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory.create_child_agent')
@patch('app.services.agent.factory.create_parent_agent')
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_create_agent_three_agents(mock_update_status, mock_create_client, mock_create_parent, mock_create_child, mock_load_configs):
    """Verify create_agent creates a parent agent when three configs are available."""
    # Setup mocks
    mock_llm = MagicMock()
    mock_websocket = MagicMock()
    mock_memory_manager = MagicMock()
    mock_checkpointer = MagicMock()
    mock_memory_manager.get_checkpointer.return_value = mock_checkpointer
    mock_websocket.app.memory_manager = mock_memory_manager
    
    mock_config1 = MagicMock()
    mock_config1.name = "RancherAgent"
    mock_config1.description = "Rancher core agent"
    mock_config1.system_prompt = "Prompt 1"
    
    mock_config2 = MagicMock()
    mock_config2.name = "FleetAgent"
    mock_config2.description = "Fleet agent"
    mock_config2.system_prompt = "Prompt 2"
    
    mock_config3 = MagicMock()
    mock_config3.name = "HarvesterAgent"
    mock_config3.description = "Harvester agent"
    mock_config3.system_prompt = "Prompt 3"
    
    mock_load_configs.return_value = [mock_config1, mock_config2, mock_config3]
    
    # Mock MCP client
    mock_client_instance = MagicMock()
    mock_tools = [MagicMock()]
    mock_client_instance.get_tools = AsyncMock(return_value=mock_tools)
    mock_create_client.return_value = mock_client_instance
    
    mock_child_agent = MagicMock()
    mock_create_child.return_value = mock_child_agent
    
    mock_parent_agent = MagicMock()
    mock_create_parent.return_value = mock_parent_agent
    
    # Execute
    result = await create_agent(mock_llm, mock_websocket)
    
    # Verify
    assert result[0] == mock_parent_agent
    assert mock_create_child.call_count == 3
    mock_create_parent.assert_called_once()
    
    # Verify parent was called with correct child agents
    call_args = mock_create_parent.call_args
    assert call_args[0][0] == mock_llm
    child_agents = call_args[0][1]
    assert len(child_agents) == 3
    assert child_agents[0].config.name == "RancherAgent"
    assert child_agents[1].config.name == "FleetAgent"
    assert child_agents[2].config.name == "HarvesterAgent"
    
    # Verify metadata includes all agents
    metadata = result[1]
    assert len(metadata) == 3
    assert all(agent["status"] == "active" for agent in metadata)


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory.create_child_agent')
@patch('app.services.agent.factory.create_parent_agent')
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_create_agent_filters_tools_by_toolset(mock_update_status, mock_create_client, mock_create_parent, mock_create_child, mock_load_configs):
    """Verify create_agent filters tools based on toolset configuration."""
    # Setup mocks
    mock_llm = MagicMock()
    mock_websocket = MagicMock()
    mock_memory_manager = MagicMock()
    mock_checkpointer = MagicMock()
    mock_memory_manager.get_checkpointer.return_value = mock_checkpointer
    mock_websocket.app.memory_manager = mock_memory_manager
    
    mock_config1 = MagicMock()
    mock_config1.name = "RancherAgent"
    mock_config1.description = "Rancher agent with specific toolset"
    mock_config1.system_prompt = "Prompt 1"
    mock_config1.toolset = "rancher-core"  # Specify toolset filter
    
    mock_config2 = MagicMock()
    mock_config2.name = "FleetAgent"
    mock_config2.description = "Fleet agent without toolset filter"
    mock_config2.system_prompt = "Prompt 2"
    mock_config2.toolset = None  # No toolset filter
    
    mock_load_configs.return_value = [mock_config1, mock_config2]
    
    # Mock MCP client with tools that have different toolsets
    # Create mock tools with metadata
    tool_rancher_core = MagicMock()
    tool_rancher_core.name = "rancher_tool"
    tool_rancher_core.metadata = {"_meta": {"toolset": "rancher-core"}}
    
    tool_rancher_extensions = MagicMock()
    tool_rancher_extensions.name = "extensions_tool"
    tool_rancher_extensions.metadata = {"_meta": {"toolset": "rancher-extensions"}}
    
    tool_fleet = MagicMock()
    tool_fleet.name = "fleet_tool"
    tool_fleet.metadata = {"_meta": {"toolset": "fleet"}}
    
    tool_no_toolset = MagicMock()
    tool_no_toolset.name = "generic_tool"
    tool_no_toolset.metadata = {}
    
    all_tools = [tool_rancher_core, tool_rancher_extensions, tool_fleet, tool_no_toolset]
    
    mock_client_instance = MagicMock()
    mock_client_instance.get_tools = AsyncMock(return_value=all_tools)
    mock_create_client.return_value = mock_client_instance
    
    mock_child_agent = MagicMock()
    mock_create_child.return_value = mock_child_agent
    
    mock_parent_agent = MagicMock()
    mock_create_parent.return_value = mock_parent_agent
    
    # Execute
    result = await create_agent(mock_llm, mock_websocket)
    
    # Verify
    assert result[0] == mock_parent_agent
    assert mock_create_child.call_count == 2
    
    # Verify first agent (RancherAgent) received only tools matching "rancher-core" toolset
    first_call_args = mock_create_child.call_args_list[0]
    first_call_tools = first_call_args[0][1]  # Second positional argument is tools
    assert len(first_call_tools) == 1
    assert first_call_tools[0].name == "rancher_tool"
    
    # Verify second agent (FleetAgent) received all tools (no toolset filter)
    second_call_args = mock_create_child.call_args_list[1]
    second_call_tools = second_call_args[0][1]
    assert len(second_call_tools) == 4  # All tools


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory.create_child_agent')
@patch('app.services.agent.factory.create_parent_agent')
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_create_agent_one_fails_mcp_connection(mock_update_status, mock_create_client, mock_create_parent, mock_create_child, mock_load_configs):
    """Verify create_agent handles MCP connection failure for one agent and continues with others."""
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
    mock_config1.ready = False
    
    mock_config2 = MagicMock()
    mock_config2.name = "Agent2"
    mock_config2.description = "Second agent"
    mock_config2.system_prompt = "Prompt 2"
    mock_config2.ready = False
    
    mock_config3 = MagicMock()
    mock_config3.name = "Agent3"
    mock_config3.description = "Third agent"
    mock_config3.system_prompt = "Prompt 3"
    mock_config3.ready = False
    
    mock_load_configs.return_value = [mock_config1, mock_config2, mock_config3]
    
    # Mock MCP client - first one fails, others succeed
    # Create three different client instances
    mock_client_fail = MagicMock()
    mock_client_fail.get_tools = AsyncMock(side_effect=Exception("Connection refused: invalid MCP URL"))
    
    mock_client_success1 = MagicMock()
    mock_tools = [MagicMock()]
    mock_client_success1.get_tools = AsyncMock(return_value=mock_tools)
    
    mock_client_success2 = MagicMock()
    mock_client_success2.get_tools = AsyncMock(return_value=mock_tools)
    
    # Return different clients on each call
    mock_create_client.side_effect = [mock_client_fail, mock_client_success1, mock_client_success2]
    
    mock_child_agent = MagicMock()
    mock_create_child.return_value = mock_child_agent
    
    mock_parent_agent = MagicMock()
    mock_create_parent.return_value = mock_parent_agent
    
    # Execute
    result = await create_agent(mock_llm, mock_websocket)
    
    # Verify two child agents were created (the successful ones)
    assert mock_create_child.call_count == 2
    
    # Should return parent agent since 2 agents succeeded
    assert result[0] == mock_parent_agent
    mock_create_parent.assert_called_once()

    # Verify parent was called with correct child agents
    call_args = mock_create_parent.call_args
    assert call_args[0][0] == mock_llm
    child_agents = call_args[0][1]
    assert len(child_agents) == 2
    assert child_agents[0].config.name == "Agent2"
    assert child_agents[1].config.name == "Agent3"
    
    # Verify metadata includes all three agents with correct status
    metadata = result[1]
    assert len(metadata) == 3
    assert metadata[0]["status"] == "error"
    assert metadata[0]["name"] == "Agent1"
    assert "Connection refused" in metadata[0]["description"]
    assert metadata[1]["status"] == "active"
    assert metadata[1]["name"] == "Agent2"
    assert metadata[2]["status"] == "active"
    assert metadata[2]["name"] == "Agent3"
    
    
    # Verify status update was called for the failed agent
    update_calls = [call for call in mock_update_status.call_args_list if call[0][1] == False]
    assert len(update_calls) > 0


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_create_agent_all_fail_mcp_connection(mock_update_status, mock_create_client, mock_load_configs):
    """Verify create_agent raises NoAgentAvailableError when all agents fail MCP connection."""
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
    mock_config1.ready = False
    
    mock_config2 = MagicMock()
    mock_config2.name = "Agent2"
    mock_config2.description = "Second agent"
    mock_config2.ready = False
    
    mock_load_configs.return_value = [mock_config1, mock_config2]
    
    # Mock MCP client - all fail
    mock_client_fail = MagicMock()
    mock_client_fail.get_tools = AsyncMock(side_effect=Exception("Connection refused: invalid MCP URL"))
    
    mock_create_client.return_value = mock_client_fail
    
    # Execute and verify exception
    with pytest.raises(NoAgentAvailableError) as exc_info:
        await create_agent(mock_llm, mock_websocket)
    
    assert "No agents could be created" in str(exc_info.value)
    
    # Verify status update was called for both failed agents
    assert mock_update_status.call_count >= 2


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
async def test_create_agent_no_configs_raises_error(mock_load_configs):
    """Verify create_agent raises NoAgentAvailableError when no configs are available."""
    mock_llm = MagicMock()
    mock_websocket = MagicMock()
    mock_memory_manager = MagicMock()
    mock_websocket.app.memory_manager = mock_memory_manager
    
    mock_load_configs.return_value = []
    
    with pytest.raises(NoAgentAvailableError) as exc_info:
        await create_agent(mock_llm, mock_websocket)
    
    assert "No agent configurations available" in str(exc_info.value)


# ============================================================================
# create_mcp_client Tests
# ============================================================================

@patch('app.services.agent.factory.MultiServerMCPClient')
def test_create_mcp_client_none_auth(mock_mcp_client):
    """Verify create_mcp_client with no authentication."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "http://test:8080"
    
    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance
    
    result = create_mcp_client(mock_config)
    
    assert result == mock_client_instance
    mock_mcp_client.assert_called_once()
    call_args = mock_mcp_client.call_args[0][0]
    assert call_args["TestAgent"]["url"] == "http://test:8080"
    assert call_args["TestAgent"]["headers"] == {}


@patch('app.services.agent.factory.MultiServerMCPClient')
@patch.dict(os.environ, {'RANCHER_URL': 'https://rancher.example.com', 'RANCHER_API_TOKEN': 'test-token'})
def test_create_mcp_client_rancher_auth_with_websocket(mock_mcp_client):
    """Verify create_mcp_client handles Rancher authentication correctly."""
    mock_websocket = MagicMock()
    mock_websocket.cookies = {"R_SESS": "cookie-token"}
    mock_websocket.url.hostname = "rancher.local"
    
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.RANCHER
    mock_config.mcp_url = "mcp-service:8080"
    
    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance
    
    result = create_mcp_client(mock_config, mock_websocket)
    
    assert result == mock_client_instance
    call_args = mock_mcp_client.call_args[0][0]
    assert call_args["TestAgent"]["url"] == "https://mcp-service:8080"
    assert call_args["TestAgent"]["headers"]['R_token'] == 'test-token'
    assert call_args["TestAgent"]["headers"]['R_url'] == 'https://rancher.example.com'


@patch('app.services.agent.factory.MultiServerMCPClient')
@patch.dict(os.environ, {'INSECURE_SKIP_TLS': 'true', 'MCP_URL': 'mcp:8080'})
def test_create_mcp_client_insecure(mock_mcp_client):
    """Verify create_mcp_client respects INSECURE_SKIP_TLS."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.RANCHER
    mock_config.mcp_url = "mcp-service:8080"
    
    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance
    
    result = create_mcp_client(mock_config)
    
    call_args = mock_mcp_client.call_args[0][0]
    assert call_args["TestAgent"]["url"].startswith("http://")


@patch('app.services.agent.factory.MultiServerMCPClient')
@patch('app.services.agent.factory.get_basic_auth_credentials')
def test_create_mcp_client_basic_auth(mock_get_creds, mock_mcp_client):
    """Verify create_mcp_client handles basic authentication."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.BASIC
    mock_config.mcp_url = "http://test:8080"
    mock_config.authentication_secret = "my-secret"
    
    mock_get_creds.return_value = "dXNlcjpwYXNz"  # base64 encoded
    
    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance
    
    result = create_mcp_client(mock_config)
    
    assert result == mock_client_instance
    call_args = mock_mcp_client.call_args[0][0]
    assert call_args["TestAgent"]["url"] == "http://test:8080"
    assert call_args["TestAgent"]["headers"]['Authorization'] == "Basic dXNlcjpwYXNz"
    mock_get_creds.assert_called_once_with("my-secret")


# ============================================================================
# _create_single_agent Tests  
# ============================================================================

@pytest.mark.asyncio
@patch('app.services.agent.factory.create_root_agent')
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_create_single_agent_success(mock_update_status, mock_create_client, mock_create_root):
    """Verify _create_single_agent creates agent successfully."""
    mock_llm = MagicMock()
    mock_websocket = MagicMock()
    mock_checkpointer = MagicMock()
    
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.system_prompt = "Test prompt"
    
    # Mock MCP client
    mock_client_instance = MagicMock()
    mock_tools = [MagicMock()]
    mock_client_instance.get_tools = AsyncMock(return_value=mock_tools)
    mock_create_client.return_value = mock_client_instance
    
    mock_agent = MagicMock()
    mock_create_root.return_value = mock_agent
    
    # Execute
    result = await _create_single_agent(mock_llm, mock_config, mock_checkpointer, mock_websocket)
    
    # Verify
    assert result == mock_agent
    mock_create_root.assert_called_once_with(mock_llm, mock_tools, "Test prompt", mock_checkpointer, mock_config)
    mock_update_status.assert_called_once()


@pytest.mark.asyncio
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_create_single_agent_mcp_failure(mock_update_status, mock_create_client):
    """Verify _create_single_agent raises error when MCP connection fails."""
    mock_llm = MagicMock()
    mock_websocket = MagicMock()
    mock_checkpointer = MagicMock()
    
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    
    # Mock MCP client to fail
    mock_client_instance = MagicMock()
    mock_client_instance.get_tools = AsyncMock(side_effect=Exception("Connection failed"))
    mock_create_client.return_value = mock_client_instance
    
    # Execute and verify exception
    with pytest.raises(NoAgentAvailableError) as exc_info:
        await _create_single_agent(mock_llm, mock_config, mock_checkpointer, mock_websocket)
    
    assert "Failed to load MCP tools" in str(exc_info.value)
    mock_update_status.assert_called()
