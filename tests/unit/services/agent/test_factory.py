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
    NeedsOauth2,
    create_mcp_client,
    _load_mcp_tools,
    _OAuth2BearerAuth,
    needs_oauth2_login,
)
from app.services.agent.loader import AuthenticationType


@pytest.fixture(autouse=True)
def _disable_rbac():
    """Disable RBAC filtering for core build_agent tests.

    RBAC behavior is covered separately in test_factory_rbac.py; these tests
    focus on agent construction and MCP wiring.
    """
    with patch('app.services.agent.factory.rbac_enabled', return_value=False):
        yield


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
        mock_llm, mock_tools, mock_checkpointer, mock_agent_config
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
    mock_client_instance.list_tools = AsyncMock(return_value=mock_tools)
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
    tool_rancher_core.metadata = {"mcp": {"tool": {"_meta": {"toolset": "rancher-core"}}}}
    
    tool_rancher_extensions = MagicMock()
    tool_rancher_extensions.name = "extensions_tool"
    tool_rancher_extensions.metadata = {"mcp": {"tool": {"_meta": {"toolset": "rancher-extensions"}}}}
    
    tool_fleet = MagicMock()
    tool_fleet.name = "fleet_tool"
    tool_fleet.metadata = {"mcp": {"tool": {"_meta": {"toolset": "fleet"}}}}
    
    tool_no_toolset = MagicMock()
    tool_no_toolset.name = "generic_tool"
    tool_no_toolset.metadata = {}
    
    all_tools = [tool_rancher_core, tool_rancher_extensions, tool_fleet, tool_no_toolset]
    
    mock_client_instance = MagicMock()
    mock_client_instance.list_tools = AsyncMock(return_value=all_tools)
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
    mock_client_fail.list_tools = AsyncMock(side_effect=Exception("Connection refused: invalid MCP URL"))
    
    mock_client_success1 = MagicMock()
    mock_tools = [MagicMock()]
    mock_client_success1.list_tools = AsyncMock(return_value=mock_tools)
    
    mock_client_success2 = MagicMock()
    mock_client_success2.list_tools = AsyncMock(return_value=mock_tools)
    
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
    mock_client_fail.list_tools = AsyncMock(side_effect=Exception("Connection refused: invalid MCP URL"))
    
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
    
    assert "No AI agents are available" in str(exc_info.value)


# ============================================================================
# create_mcp_client Tests
# ============================================================================

@pytest.fixture
def mock_adapter():
    """Stub out the FastMCP client and MCPAdapter wrapping the transport built by create_mcp_client."""
    with patch('app.services.agent.factory.Client') as mock_client_cls, \
         patch('app.services.agent.factory.MCPAdapter') as mock_adapter_cls:
        yield mock_adapter_cls, mock_client_cls


@pytest.mark.asyncio
@patch('app.services.agent.factory.StreamableHttpTransport')
async def test_create_mcp_client_none_auth(mock_transport, mock_adapter):
    """Verify create_mcp_client with no authentication."""
    mock_adapter_cls, mock_client_cls = mock_adapter
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "http://test:8080"
    mock_config.ca_bundle_ref = None

    result = await create_mcp_client(mock_config)

    assert result is mock_adapter_cls.return_value
    mock_client_cls.assert_called_once_with(mock_transport.return_value)
    mock_adapter_cls.assert_called_once_with(mock_client_cls.return_value)
    mock_transport.assert_called_once_with("http://test:8080", headers={}, auth=None, verify=None)


@pytest.mark.asyncio
@patch('app.services.agent.factory.StreamableHttpTransport')
@patch.dict(os.environ, {'RANCHER_URL': 'https://rancher.example.com', 'RANCHER_API_TOKEN': 'test-token', 'INSECURE_SKIP_TLS': 'false'})
async def test_create_mcp_client_rancher_auth_with_websocket(mock_transport, mock_adapter):
    """Verify create_mcp_client handles Rancher authentication correctly."""
    mock_adapter_cls, _ = mock_adapter
    mock_websocket = MagicMock()
    mock_websocket.cookies = {"R_SESS": "cookie-token"}
    mock_websocket.url.hostname = "rancher.local"
    
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.RANCHER
    mock_config.mcp_url = "mcp-service:8080"
    mock_config.ca_bundle_ref = None
    
    result = await create_mcp_client(mock_config, mock_websocket)
    
    assert result is mock_adapter_cls.return_value
    call_args = mock_transport.call_args
    assert call_args.args[0] == "https://mcp-service:8080"
    assert call_args.kwargs["headers"]['R_token'] == 'test-token'


@pytest.mark.asyncio
@patch('app.services.agent.factory.StreamableHttpTransport')
@patch.dict(os.environ, {'INSECURE_SKIP_TLS': 'true', 'MCP_URL': 'mcp:8080'})
async def test_create_mcp_client_insecure(mock_transport, mock_adapter):
    """Verify create_mcp_client respects INSECURE_SKIP_TLS by disabling TLS verification."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.RANCHER
    mock_config.mcp_url = "mcp-service:8080"

    await create_mcp_client(mock_config)

    call_args = mock_transport.call_args
    assert call_args.args[0] == "http://mcp:8080"
    assert call_args.kwargs["verify"] is False


@pytest.mark.asyncio
@patch('app.services.agent.factory.StreamableHttpTransport')
@patch.dict(os.environ, {'INSECURE_SKIP_TLS': 'true', 'MCP_URL': 'https://mcp:8080'})
async def test_create_mcp_client_insecure_with_existing_scheme(mock_transport, mock_adapter):
    """Verify create_mcp_client preserves an existing URL scheme."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.RANCHER
    mock_config.mcp_url = "mcp-service:8080"
    
    await create_mcp_client(mock_config)
    
    call_args = mock_transport.call_args
    assert call_args.args[0] == "https://mcp:8080"
    assert call_args.kwargs["verify"] is False


@pytest.mark.asyncio
@patch('app.services.agent.factory.StreamableHttpTransport')
@patch('app.services.agent.factory.get_basic_auth_credentials')
async def test_create_mcp_client_basic_auth(mock_get_creds, mock_transport, mock_adapter):
    """Verify create_mcp_client handles basic authentication."""
    mock_adapter_cls, _ = mock_adapter
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.BASIC
    mock_config.mcp_url = "http://test:8080"
    mock_config.authentication_secret = "my-secret"
    mock_config.ca_bundle_ref = None
    
    mock_get_creds.return_value = "dXNlcjpwYXNz"  # base64 encoded
    
    result = await create_mcp_client(mock_config)
    
    assert result is mock_adapter_cls.return_value
    call_args = mock_transport.call_args
    assert call_args.args[0] == "http://test:8080"
    assert call_args.kwargs["headers"]['Authorization'] == "Basic dXNlcjpwYXNz"
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
    mock_client_instance.list_tools = AsyncMock(return_value=mock_tools)
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
    mock_client_instance.list_tools = AsyncMock(side_effect=Exception("Connection failed"))
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
    tool_matching.metadata = {"mcp": {"tool": {"_meta": {"toolset": "rancher-core"}}}}

    tool_other = MagicMock()
    tool_other.name = "other_tool"
    tool_other.metadata = {"mcp": {"tool": {"_meta": {"toolset": "fleet"}}}}

    tool_multi_after = MagicMock()
    tool_multi_after.name = "multi_toolset_after"
    tool_multi_after.metadata = {"mcp": {"tool": {"_meta": {"toolset": "fleet, rancher-core"}}}}

    tool_multi_before = MagicMock()
    tool_multi_before.name = "multi_toolset_before"
    tool_multi_before.metadata = {"mcp": {"tool": {"_meta": {"toolset": "rancher-core, harvester, fleet"}}}}

    tool_no_meta = MagicMock()
    tool_no_meta.name = "generic_tool"
    tool_no_meta.metadata = {}

    mock_client_instance = MagicMock()
    mock_client_instance.list_tools = AsyncMock(return_value=[tool_matching, tool_other, tool_multi_after, tool_multi_before, tool_no_meta])
    mock_create_client.return_value = mock_client_instance

    result = await _load_mcp_tools(mock_config, mock_websocket)

    assert len(result) == 3
    assert result[0].name == "matching_tool"
    assert result[1].name == "multi_toolset_after"
    assert result[2].name == "multi_toolset_before"


@pytest.mark.asyncio
@patch('app.services.agent.factory.StreamableHttpTransport')
@patch('app.services.agent.factory.get_header_auth_headers')
async def test_create_mcp_client_header_auth(mock_get_headers, mock_transport, mock_adapter):
    """Verify create_mcp_client handles header authentication."""
    mock_adapter_cls, _ = mock_adapter
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.HEADER
    mock_config.mcp_url = "http://test:8080"
    mock_config.authentication_secret = "my-headers-secret"
    mock_config.ca_bundle_ref = None

    mock_get_headers.return_value = {
        "X-Api-Key": "my-api-key",
        "Authorization": "Bearer tok123",
    }

    result = await create_mcp_client(mock_config)

    assert result is mock_adapter_cls.return_value
    call_args = mock_transport.call_args
    assert call_args.args[0] == "http://test:8080"
    assert call_args.kwargs["headers"]["X-Api-Key"] == "my-api-key"
    assert call_args.kwargs["headers"]["Authorization"] == "Bearer tok123"
    mock_get_headers.assert_called_once_with("my-headers-secret")


@pytest.mark.asyncio
@patch('app.services.agent.factory.build_ca_ssl_context')
@patch('app.services.agent.factory.StreamableHttpTransport')
@patch('app.services.agent.factory.get_ca_cert_from_secret')
async def test_create_mcp_client_with_ca_bundle_ref(mock_get_ca, mock_transport, mock_build_ctx, mock_adapter):
    """Verify create_mcp_client verifies TLS with the caBundleRef CA when configured."""
    mock_adapter_cls, _ = mock_adapter
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "https://mcp:8080"
    ca_ref = MagicMock()
    ca_ref.name = "my-ca-secret"
    ca_ref.key = "ca.crt"
    mock_config.ca_bundle_ref = ca_ref

    ca_pem = "-----BEGIN CERTIFICATE-----\nFAKE\n-----END CERTIFICATE-----\n"
    mock_get_ca.return_value = ca_pem
    mock_ctx = MagicMock()
    mock_build_ctx.return_value = mock_ctx

    result = await create_mcp_client(mock_config)

    assert result is mock_adapter_cls.return_value
    mock_get_ca.assert_called_once_with("my-ca-secret", "ca.crt")
    mock_build_ctx.assert_called_once_with(ca_pem)
    assert mock_transport.call_args.kwargs["verify"] is mock_ctx


@pytest.mark.asyncio
@patch('app.services.agent.factory.build_ca_ssl_context')
@patch('app.services.agent.factory.StreamableHttpTransport')
@patch('app.services.agent.factory.get_ca_cert_from_secret')
async def test_create_mcp_client_with_ca_bundle_ref_custom_key(mock_get_ca, mock_transport, mock_build_ctx, mock_adapter):
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

    await create_mcp_client(mock_config)

    mock_get_ca.assert_called_once_with("my-ca-secret", "tls.crt")


@pytest.mark.asyncio
@patch('app.services.agent.factory.StreamableHttpTransport')
async def test_create_mcp_client_without_ca_bundle_ref(mock_transport, mock_adapter):
    """Verify create_mcp_client uses default TLS verification when no caBundleRef."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "https://mcp:8080"
    mock_config.ca_bundle_ref = None

    await create_mcp_client(mock_config)

    assert mock_transport.call_args.kwargs["verify"] is None


@pytest.mark.asyncio
@patch('app.services.agent.factory.StreamableHttpTransport')
@patch('app.services.agent.factory.get_ca_cert_from_secret')
async def test_create_mcp_client_ca_bundle_ref_failure_logs_and_continues(mock_get_ca, mock_transport, mock_adapter):
    """Verify create_mcp_client gracefully handles CA secret loading failure."""
    mock_adapter_cls, _ = mock_adapter
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.NONE
    mock_config.mcp_url = "https://mcp:8080"
    ca_ref = MagicMock()
    ca_ref.name = "bad-secret"
    ca_ref.key = "ca.crt"
    mock_config.ca_bundle_ref = ca_ref

    mock_get_ca.side_effect = RuntimeError("secret not found")

    result = await create_mcp_client(mock_config)

    # Should still return a client, just without custom CA
    assert result is mock_adapter_cls.return_value
    assert mock_transport.call_args.kwargs["verify"] is None


@pytest.mark.asyncio
@patch('app.services.agent.factory.StreamableHttpTransport')
async def test_create_mcp_client_oauth2_sends_token_via_auth(mock_transport, mock_adapter):
    """Verify create_mcp_client passes the OAuth2 cookie token through a 401-tracking auth."""
    mock_adapter_cls, _ = mock_adapter
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.OAUTH2
    mock_config.mcp_url = "https://mcp:8080"
    mock_config.ca_bundle_ref = None

    mock_websocket = MagicMock()
    mock_websocket.cookies.get.return_value = "access-token"

    result = await create_mcp_client(mock_config, mock_websocket)

    auth = mock_transport.call_args.kwargs["auth"]
    assert isinstance(auth, _OAuth2BearerAuth)
    assert auth.token == "access-token"
    assert mock_transport.call_args.kwargs["headers"] == {}
    assert result.oauth2_auth is auth


def _run_auth_flow(auth: _OAuth2BearerAuth, status_code: int):
    import httpx2
    request = httpx2.Request("POST", "https://mcp:8080/mcp")
    flow = auth.auth_flow(request)
    sent = next(flow)
    with pytest.raises(StopIteration):
        flow.send(httpx2.Response(status_code, request=request))
    return sent


@pytest.mark.parametrize("token,status_code,expected_header,expected_unauthorized", [
    ("tok", 200, "Bearer tok", False),
    ("tok", 401, "Bearer tok", True),
    (None, 401, None, True),
])
def test_oauth2_bearer_auth(token, status_code, expected_header, expected_unauthorized):
    """Verify _OAuth2BearerAuth sets the bearer header when it has a token and records 401 replies."""
    auth = _OAuth2BearerAuth(token)

    sent = _run_auth_flow(auth, status_code)

    assert sent.headers.get("Authorization") == expected_header
    assert auth.unauthorized is expected_unauthorized


@pytest.mark.asyncio
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_load_mcp_tools_oauth2_unauthorized_raises_needs_oauth2(mock_update_status, mock_create_client):
    """Verify a 401 seen by the OAuth2 auth triggers NeedsOauth2 even though the MCP error hides the status code."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.OAUTH2

    auth = _OAuth2BearerAuth(None)
    auth.unauthorized = True
    mock_client_instance = MagicMock()
    mock_client_instance.oauth2_auth = auth
    mock_client_instance.list_tools = AsyncMock(side_effect=Exception("Server returned an error response"))
    mock_create_client.return_value = mock_client_instance

    with pytest.raises(NeedsOauth2):
        await _load_mcp_tools(mock_config, MagicMock())

    mock_update_status.assert_not_called()


@pytest.mark.asyncio
@patch('app.services.agent.factory.create_mcp_client')
@patch('app.services.agent.factory._update_agent_status')
async def test_load_mcp_tools_oauth2_other_error_raises_no_agent(mock_update_status, mock_create_client):
    """Verify non-401 failures for OAuth2 agents still raise NoAgentAvailableError."""
    mock_config = MagicMock()
    mock_config.name = "TestAgent"
    mock_config.authentication = AuthenticationType.OAUTH2

    mock_client_instance = MagicMock()
    mock_client_instance.oauth2_auth = _OAuth2BearerAuth("tok")
    mock_client_instance.list_tools = AsyncMock(side_effect=Exception("Server returned an error response"))
    mock_create_client.return_value = mock_client_instance

    with pytest.raises(NoAgentAvailableError):
        await _load_mcp_tools(mock_config, MagicMock())


def test_needs_oauth2_login_ignores_non_oauth2_agents():
    """Verify a recorded 401 only means 'needs login' for OAuth2 agents."""
    auth = _OAuth2BearerAuth(None)
    auth.unauthorized = True
    mock_client = MagicMock()
    mock_client.oauth2_auth = auth

    oauth2_cfg = MagicMock(authentication=AuthenticationType.OAUTH2)
    none_cfg = MagicMock(authentication=AuthenticationType.NONE)

    assert needs_oauth2_login(oauth2_cfg, mock_client, "Server returned an error response") is True
    assert needs_oauth2_login(none_cfg, mock_client, "Server returned an error response") is False
    assert needs_oauth2_login(none_cfg, mock_client, "401 Unauthorized") is False
