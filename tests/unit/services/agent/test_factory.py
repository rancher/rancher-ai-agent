"""
Unit tests for the agent factory module.

Tests agent creation and MCP tool setup.
"""
import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from contextlib import AsyncExitStack

from app.services.agent.factory import (
    build_agent,
    NoAgentAvailableError,
    create_mcp_client,
    _load_mcp_tools,
    _make_ca_httpx_factory,
)
from app.services.agent.loader import AuthenticationType


# ============================================================================
# build_agent Tests
# ============================================================================

@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory._load_mcp_tools')
@patch('app.services.agent.factory.create_child_agent')
async def test_create_agent_single_agent(mock_create_child, mock_load_tools, mock_load_configs):
    """Verify build_agent creates a child agent when one config is available."""
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
    mock_load_tools.return_value = mock_tools
    mock_agent = MagicMock()
    mock_create_child.return_value = mock_agent
    
    # Execute
    result = await build_agent(mock_llm, mock_websocket)
    
    # Verify
    assert result[0] == mock_agent
    assert result[1] == [{"name": "RancherAgent", "status": "active"}]
    mock_load_tools.assert_called_once_with(mock_agent_config, mock_websocket)
    mock_create_child.assert_called_once_with(
        mock_llm, mock_tools, "Test prompt", mock_checkpointer, mock_agent_config
    )


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory.create_supervisor_agent')
@patch('app.services.agent.factory.create_child_agent')
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_build_agent_three_agents(mock_update_status, mock_create_client, mock_create_child, mock_create_parent, mock_load_configs):
    """Verify build_agent creates a supervisor agent when three configs are available."""
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
    
    mock_parent_agent = MagicMock()
    mock_create_parent.return_value = mock_parent_agent
    mock_create_child.return_value = MagicMock()
    
    # Execute
    result = await build_agent(mock_llm, mock_websocket)
    
    # Verify - build_agent wraps multi-agent result in SupervisorGraph
    from app.services.agent.supervisor import SupervisorGraph
    assert isinstance(result[0], SupervisorGraph)
    assert result[0]._graph == mock_parent_agent
    mock_create_parent.assert_called_once()
    
    # Verify parent was called with correct subagents
    call_args = mock_create_parent.call_args
    assert call_args[0][0] == mock_llm
    subagents = call_args[0][1]
    assert len(subagents) == 3
    assert subagents[0].config.name == "RancherAgent"
    assert subagents[1].config.name == "FleetAgent"
    assert subagents[2].config.name == "HarvesterAgent"
    
    # Verify metadata includes all agents
    metadata = result[1]
    assert len(metadata) == 3
    assert all(agent["status"] == "active" for agent in metadata)


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory.create_supervisor_agent')
@patch('app.services.agent.factory.create_child_agent')
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_build_agent_filters_tools_by_toolset(mock_update_status, mock_create_client, mock_create_child, mock_create_parent, mock_load_configs):
    """Verify build_agent filters tools based on toolset configuration."""
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
    
    mock_parent_agent = MagicMock()
    mock_create_parent.return_value = mock_parent_agent
    mock_create_child.return_value = MagicMock()
    
    # Execute
    result = await build_agent(mock_llm, mock_websocket)
    
    # Verify - build_agent wraps multi-agent result in SupervisorGraph
    from app.services.agent.supervisor import SupervisorGraph
    assert isinstance(result[0], SupervisorGraph)
    
    # Verify subagents passed to create_supervisor_agent have correct tools
    call_args = mock_create_parent.call_args
    subagents = call_args[0][1]
    assert len(subagents) == 2
    
    # First subagent (RancherAgent) should have only rancher-core tools
    assert subagents[0].config.name == "RancherAgent"
    
    # Second subagent (FleetAgent) should have all tools (no toolset filter)
    assert subagents[1].config.name == "FleetAgent"

    # Verify create_child_agent was called with filtered tools
    child_calls = mock_create_child.call_args_list
    assert len(child_calls) == 2
    # RancherAgent: only rancher-core tool
    rancher_tools = child_calls[0][0][1]
    assert len(rancher_tools) == 1
    assert rancher_tools[0].name == "rancher_tool"
    # FleetAgent: all tools (no toolset filter)
    fleet_tools = child_calls[1][0][1]
    assert len(fleet_tools) == 4


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
@patch('app.services.agent.factory.create_supervisor_agent')
@patch('app.services.agent.factory.create_child_agent')
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_build_agent_one_fails_mcp_connection(mock_update_status, mock_create_client, mock_create_child, mock_create_parent, mock_load_configs):
    """Verify build_agent handles MCP connection failure for one agent and continues with others."""
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
    
    mock_parent_agent = MagicMock()
    mock_create_parent.return_value = mock_parent_agent
    mock_create_child.return_value = MagicMock()
    
    # Execute
    result = await build_agent(mock_llm, mock_websocket)
    
    # Should return supervisor since 2 agents succeeded
    from app.services.agent.supervisor import SupervisorGraph
    assert isinstance(result[0], SupervisorGraph)
    mock_create_parent.assert_called_once()

    # Verify parent was called with correct subagents
    call_args = mock_create_parent.call_args
    assert call_args[0][0] == mock_llm
    subagents = call_args[0][1]
    assert len(subagents) == 2
    assert subagents[0].config.name == "Agent2"
    assert subagents[1].config.name == "Agent3"
    
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
async def test_build_agent_all_fail_mcp_connection(mock_update_status, mock_create_client, mock_load_configs):
    """Verify build_agent raises NoAgentAvailableError when all agents fail MCP connection."""
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
        await build_agent(mock_llm, mock_websocket)
    
    assert "No agents could be created" in str(exc_info.value)
    
    # Verify status update was called for both failed agents
    assert mock_update_status.call_count >= 2


@pytest.mark.asyncio
@patch('app.services.agent.factory.load_agent_configs')
async def test_build_agent_no_configs_raises_error(mock_load_configs):
    """Verify build_agent raises NoAgentAvailableError when no configs are available."""
    mock_llm = MagicMock()
    mock_websocket = MagicMock()
    mock_memory_manager = MagicMock()
    mock_websocket.app.memory_manager = mock_memory_manager
    
    mock_load_configs.return_value = []
    
    with pytest.raises(NoAgentAvailableError) as exc_info:
        await build_agent(mock_llm, mock_websocket)
    
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
@patch.dict(os.environ, {'RANCHER_URL': 'https://rancher.example.com', 'RANCHER_API_TOKEN': 'test-token', 'INSECURE_SKIP_TLS': 'false'})
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
    """Verify create_mcp_client respects INSECURE_SKIP_TLS by disabling TLS verification."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.RANCHER
    mock_config.mcp_url = "mcp-service:8080"

    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance

    result = create_mcp_client(mock_config)

    call_args = mock_mcp_client.call_args[0][0]
    assert call_args["TestAgent"]["url"] == "http://mcp:8080"
    assert call_args["TestAgent"]["httpx_client_factory"] is not None


@patch('app.services.agent.factory.MultiServerMCPClient')
@patch.dict(os.environ, {'INSECURE_SKIP_TLS': 'true', 'MCP_URL': 'https://mcp:8080'})
def test_create_mcp_client_insecure_with_existing_scheme(mock_mcp_client):
    """Verify create_mcp_client preserves an existing URL scheme."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.RANCHER
    mock_config.mcp_url = "mcp-service:8080"
    
    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance
    
    result = create_mcp_client(mock_config)
    
    call_args = mock_mcp_client.call_args[0][0]
    assert call_args["TestAgent"]["url"] == "https://mcp:8080"
    assert call_args["TestAgent"]["httpx_client_factory"] is not None


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
# _load_mcp_tools Tests  
# ============================================================================

@pytest.mark.asyncio
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_load_mcp_tools_success(mock_update_status, mock_create_client):
    """Verify _load_mcp_tools loads and returns tools successfully."""
    mock_websocket = MagicMock()
    
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.toolset = None
    
    mock_client_instance = MagicMock()
    mock_tools = [MagicMock()]
    mock_client_instance.get_tools = AsyncMock(return_value=mock_tools)
    mock_create_client.return_value = mock_client_instance
    
    # Execute
    result = await _load_mcp_tools(mock_config, mock_websocket)
    
    # Verify
    assert result == mock_tools
    mock_update_status.assert_called_once_with(mock_config, True, 'MCPConnectionSucceeded', 'MCP tools loaded successfully')


@pytest.mark.asyncio
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_load_mcp_tools_mcp_failure(mock_update_status, mock_create_client):
    """Verify _load_mcp_tools raises NoAgentAvailableError when MCP connection fails."""
    mock_websocket = MagicMock()
    
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    
    mock_client_instance = MagicMock()
    mock_client_instance.get_tools = AsyncMock(side_effect=Exception("Connection failed"))
    mock_create_client.return_value = mock_client_instance
    
    # Execute and verify exception
    with pytest.raises(NoAgentAvailableError) as exc_info:
        await _load_mcp_tools(mock_config, mock_websocket)
    
    assert "Failed to load MCP tools" in str(exc_info.value)
    mock_update_status.assert_called()


@pytest.mark.asyncio
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_load_mcp_tools_filters_by_toolset(mock_update_status, mock_create_client):
    """Verify _load_mcp_tools filters tools by toolset when configured."""
    mock_websocket = MagicMock()

    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.toolset = "rancher-core"

    tool_matching = MagicMock()
    tool_matching.name = "matching_tool"
    tool_matching.metadata = {"_meta": {"toolset": "rancher-core"}}

    tool_other = MagicMock()
    tool_other.name = "other_tool"
    tool_other.metadata = {"_meta": {"toolset": "fleet"}}

    tool_no_meta = MagicMock()
    tool_no_meta.name = "generic_tool"
    tool_no_meta.metadata = {}

    mock_client_instance = MagicMock()
    mock_client_instance.get_tools = AsyncMock(return_value=[tool_matching, tool_other, tool_no_meta])
    mock_create_client.return_value = mock_client_instance

    result = await _load_mcp_tools(mock_config, mock_websocket)

    assert len(result) == 1
    assert result[0].name == "matching_tool"


@patch('app.services.agent.factory.MultiServerMCPClient')
@patch('app.services.agent.factory.get_header_auth_headers')
def test_create_mcp_client_header_auth(mock_get_headers, mock_mcp_client):
    """Verify create_mcp_client handles header authentication."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.HEADER
    mock_config.mcp_url = "http://test:8080"
    mock_config.authentication_secret = "my-headers-secret"

    mock_get_headers.return_value = {
        "X-Api-Key": "my-api-key",
        "Authorization": "Bearer tok123",
    }

    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance

    result = create_mcp_client(mock_config)

    assert result == mock_client_instance
    call_args = mock_mcp_client.call_args[0][0]
    assert call_args["TestAgent"]["url"] == "http://test:8080"
    assert call_args["TestAgent"]["headers"]["X-Api-Key"] == "my-api-key"
    assert call_args["TestAgent"]["headers"]["Authorization"] == "Bearer tok123"
    mock_get_headers.assert_called_once_with("my-headers-secret")


@patch('app.services.agent.factory._make_ca_httpx_factory')
@patch('app.services.agent.factory.MultiServerMCPClient')
@patch('app.services.agent.factory.get_ca_cert_from_secret')
def test_create_mcp_client_with_ca_bundle_ref(mock_get_ca, mock_mcp_client, mock_make_factory):
    """Verify create_mcp_client sets httpx_client_factory when caBundleRef is configured."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "https://mcp:8080"
    ca_ref = MagicMock()
    ca_ref.name = "my-ca-secret"
    ca_ref.key = "ca.crt"
    mock_config.ca_bundle_ref = ca_ref

    mock_get_ca.return_value = "-----BEGIN CERTIFICATE-----\nFAKE\n-----END CERTIFICATE-----\n"
    mock_factory = MagicMock()
    mock_make_factory.return_value = mock_factory

    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance

    result = create_mcp_client(mock_config)

    assert result == mock_client_instance
    mock_get_ca.assert_called_once_with("my-ca-secret", "ca.crt")
    call_args = mock_mcp_client.call_args[0][0]
    assert "httpx_client_factory" in call_args["TestAgent"]
    assert call_args["TestAgent"]["httpx_client_factory"] == mock_factory


@patch('app.services.agent.factory._make_ca_httpx_factory')
@patch('app.services.agent.factory.MultiServerMCPClient')
@patch('app.services.agent.factory.get_ca_cert_from_secret')
def test_create_mcp_client_with_ca_bundle_ref_custom_key(mock_get_ca, mock_mcp_client, mock_make_factory):
    """Verify create_mcp_client passes a custom key from caBundleRef."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "https://mcp:8080"
    ca_ref = MagicMock()
    ca_ref.name = "my-ca-secret"
    ca_ref.key = "tls.crt"
    mock_config.ca_bundle_ref = ca_ref

    mock_get_ca.return_value = "-----BEGIN CERTIFICATE-----\nFAKE\n-----END CERTIFICATE-----\n"

    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance

    create_mcp_client(mock_config)

    mock_get_ca.assert_called_once_with("my-ca-secret", "tls.crt")


@patch('app.services.agent.factory.MultiServerMCPClient')
def test_create_mcp_client_without_ca_bundle_ref(mock_mcp_client):
    """Verify create_mcp_client omits httpx_client_factory when no caBundleRef."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "https://mcp:8080"
    mock_config.ca_bundle_ref = None

    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance

    create_mcp_client(mock_config)

    call_args = mock_mcp_client.call_args[0][0]
    assert "httpx_client_factory" not in call_args["TestAgent"]


@patch('app.services.agent.factory.MultiServerMCPClient')
@patch('app.services.agent.factory.get_ca_cert_from_secret')
def test_create_mcp_client_ca_bundle_ref_failure_logs_and_continues(mock_get_ca, mock_mcp_client):
    """Verify create_mcp_client gracefully handles CA secret loading failure."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "https://mcp:8080"
    ca_ref = MagicMock()
    ca_ref.name = "bad-secret"
    ca_ref.key = "ca.crt"
    mock_config.ca_bundle_ref = ca_ref

    mock_get_ca.side_effect = RuntimeError("secret not found")

    mock_client_instance = MagicMock()
    mock_mcp_client.return_value = mock_client_instance

    result = create_mcp_client(mock_config)

    # Should still return a client, just without custom CA
    assert result == mock_client_instance
    call_args = mock_mcp_client.call_args[0][0]
    assert "httpx_client_factory" not in call_args["TestAgent"]


def test_make_ca_httpx_factory_returns_async_client():
    """Verify _make_ca_httpx_factory returns a factory that produces httpx.AsyncClient."""
    import httpx
    with patch('app.services.agent.factory.ssl') as mock_ssl:
        mock_ctx = MagicMock()
        mock_ssl.create_default_context.return_value = mock_ctx

        factory = _make_ca_httpx_factory("FAKE-PEM")

        mock_ssl.create_default_context.assert_called_once()
        mock_ctx.load_verify_locations.assert_called_once_with(cadata="FAKE-PEM")

        client = factory(headers={"X-Test": "val"}, timeout=httpx.Timeout(10))
        assert isinstance(client, httpx.AsyncClient)
        assert client._transport._pool._ssl_context == mock_ctx
