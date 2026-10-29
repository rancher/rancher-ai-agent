import os
import logging
from datetime import datetime, timezone

import httpx2
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from kubernetes import client

from .loader import (
    AuthenticationType,
    load_agent_configs,
    AgentConfig,
    get_basic_auth_credentials,
    get_header_auth_headers,
    get_ca_cert_from_secret,
    build_ca_ssl_context,
    _load_k8s_config,
)
from .planner import create_planner_agent
from ..rbac import accessible_agent_configs, rbac_enabled, RBACError
from ..oauth2 import get_oauth_cookie_names
from .supervisor import create_supervisor_agent, ChildAgent, SupervisorGraph
from .child import create_child_agent
from ._constants import NoAgentAvailableError, NeedsOauth2
from fastapi import  WebSocket
from langchain_core._api import suppress_langchain_beta_warning
from langchain_core.language_models.llms import BaseLanguageModel
with suppress_langchain_beta_warning():
    from langchain.mcp import MCPAdapter
from langgraph.graph.state import Checkpointer, CompiledStateGraph

NAMESPACE = "cattle-ai-agent-system"


class _OAuth2BearerAuth(httpx2.Auth):
    """
    Sends the OAuth2 access token (if any) and records whether the MCP server
    answered 401. MCP SDK 2 turns HTTP errors into a generic JSON-RPC error
    without the status code, so this is how a rejected/missing token is detected.
    """

    def __init__(self, token: str | None):
        self.token = token
        self.unauthorized = False

    def auth_flow(self, request: httpx2.Request):
        if self.token:
            request.headers["Authorization"] = f"Bearer {self.token}"
        response = yield request
        if response.status_code == 401:
            self.unauthorized = True


async def build_agent(llm: BaseLanguageModel, websocket: WebSocket) -> tuple[CompiledStateGraph | SupervisorGraph, list[dict]]:
    """
    Build an agent graph from AIAgentConfig CRDs.

    Loads agent configurations and creates either a supervisor (multi-agent) or a
    single child agent depending on how many configs are available and successfully
    connect to their MCP servers.

    Args:
        llm: The language model to use for agent reasoning.
        websocket: WebSocket connection for authentication context.

    Returns:
        A tuple of (compiled agent graph, agent metadata list).

    Raises:
        NoAgentAvailableError: If no agent configurations exist or all MCP connections fail.
    """
    checkpointer = websocket.app.memory_manager.get_checkpointer()

    # Fetch only the agents the requesting user is allowed to access. On the RBAC
    # path this is a single scoped call to Rancher (no full CRD listing).
    agent_configs = await _load_agents_for_user(websocket)

    if not agent_configs:
        raise NoAgentAvailableError(
            "No AI agents are available for your role. "
            "If you believe this is an error, contact your administrator."
        )

    logging.info(f"Loaded {len(agent_configs)} accessible agent configuration(s)")

    if len(agent_configs) == 1:
        agent_cfg = agent_configs[0]
        tools = await _load_mcp_tools(agent_cfg, websocket)
        graph = create_child_agent(llm, tools, checkpointer, agent_cfg)
        return graph, [{"name": agent_cfg.name, "status": "active"}]

    # Multi-agent setup
    logging.info(f"Multi-agent setup detected, creating supervisor with {len(agent_configs)} agents.")
    child_agents, agents_metadata = await _build_child_agents(llm, agent_configs, checkpointer, websocket)

    if not child_agents:
        logging.error("Failed to create any child agents due to MCP connection issues")
        raise NoAgentAvailableError(
            "No agents could be created. Please check the MCP server connections and configurations for each agent."
        )

    if len(child_agents) == 1:
        logging.warning("Only one child agent connected successfully. Using it directly instead of a supervisor.")
        return child_agents[0].agent, agents_metadata

    graph = (
        create_planner_agent(llm, child_agents, checkpointer)
        if os.environ.get("PLAN_ENABLED", "false").lower() == "true"
        else create_supervisor_agent(llm, child_agents, checkpointer)
    )
    supervisor = SupervisorGraph(
        graph=graph,
        child_agents={ca.config.name: ca for ca in child_agents},
    )
    return supervisor, agents_metadata


async def _build_child_agents(
    llm: BaseLanguageModel,
    agent_configs: list[AgentConfig],
    checkpointer: Checkpointer,
    websocket: WebSocket,
) -> tuple[list[ChildAgent], list[dict]]:
    """
    Attempt to build a child agent for each config, collecting successes and failures.

    Returns:
        A tuple of (successfully created ChildAgents, metadata for all agents).
    """
    child_agents: list[ChildAgent] = []
    agents_metadata: list[dict] = []

    for agent_cfg in agent_configs:
        try:
            tools = await _load_mcp_tools(agent_cfg, websocket)
            child_agents.append(ChildAgent(
                config=agent_cfg,
                agent=create_child_agent(llm, tools, checkpointer, agent_cfg),
            ))
            agents_metadata.append({"name": agent_cfg.name, "status": "active"})
        except NeedsOauth2 as e:
            logging.debug(f"OAuth2 authentication required for agent '{agent_cfg.name}': {e}")
            child_agents.append(ChildAgent(
                config=agent_cfg,
                agent=create_child_agent(llm, [], checkpointer, agent_cfg),
                needs_oauth2=True
            ))
            agents_metadata.append({"name": agent_cfg.name, "status": "active", "description": "needs_oauth2"})
        except (NoAgentAvailableError, Exception) as e:
            logging.error(f"Failed to load MCP tools for agent '{agent_cfg.name}': {e}")
            agents_metadata.append({"name": agent_cfg.name, "status": "error", "description": str(e)})

    return child_agents, agents_metadata


async def reload_agent_tools(
    llm: BaseLanguageModel,
    agent_cfg: AgentConfig,
    websocket: WebSocket,
) -> CompiledStateGraph:
    """
    Reload MCP tools for an agent and rebuild its child agent graph.

    Used after successful OAuth2 authentication to reconnect to the MCP server
    with valid credentials and rebuild the agent with the new tools.

    Args:
        llm: The language model for the agent.
        agent_cfg: The agent configuration.
        websocket: The WebSocket connection (now with OAuth cookies set).

    Returns:
        The compiled child agent graph.

    Raises:
        NoAgentAvailableError: If MCP connection fails.
    """
    checkpointer = websocket.app.memory_manager.get_checkpointer()
    tools = await _load_mcp_tools(agent_cfg, websocket)
    return create_child_agent(llm, tools, checkpointer, agent_cfg)


async def _load_mcp_tools(agent_cfg: AgentConfig, websocket: WebSocket) -> list:
    """
    Connect to the MCP server for an agent config, load tools, filter by toolset,
    and update the CRD status accordingly.

    Raises:
        NoAgentAvailableError: If the MCP connection fails.
    """

    mcp_client = await create_mcp_client(agent_cfg, websocket)
    
    try:
        tools = await mcp_client.list_tools()
    except* Exception as eg:
        error_message = " ".join(str(e) for e in eg.exceptions)

        if needs_oauth2_login(agent_cfg, mcp_client, error_message):
            raise NeedsOauth2(agent_cfg)
        else:
            _update_agent_status(agent_cfg, False, 'MCPConnectionFailed', f"Failed to load MCP tools: {error_message}")
            raise NoAgentAvailableError(
                f"Failed to load MCP tools for agent '{agent_cfg.name}': {error_message}\n"
                f"Please check the AI Agents configuration and ensure the MCP server is accessible."
            ) from eg

    if agent_cfg.toolset:
        tools = [
            t for t in tools
            if agent_cfg.toolset in [
                ts.strip() 
                for ts in (_tool_meta(t).get("toolset") or "").split(",")
            ]
        ]                
        logging.debug(f"Filtered {len(tools)} tools for toolset '{agent_cfg.toolset}'")

    _update_agent_status(agent_cfg, True, 'MCPConnectionSucceeded', 'MCP tools loaded successfully')
    return tools



def needs_oauth2_login(agent_cfg: AgentConfig, mcp_client: MCPAdapter, error_message: str) -> bool:
    """
    Return True if an OAuth2 agent's MCP server rejected the (missing or expired) token.

    The 401 is read from the client's ``_OAuth2BearerAuth`` because MCP SDK 2 errors
    no longer carry the HTTP status; the message checks are kept as a fallback.
    """
    if agent_cfg.authentication != AuthenticationType.OAUTH2:
        return False
    oauth2_auth = getattr(mcp_client, "oauth2_auth", None)
    error_lower = error_message.lower()
    return (
        (isinstance(oauth2_auth, _OAuth2BearerAuth) and oauth2_auth.unauthorized)
        or "401" in error_message
        or "unauthorized" in error_lower
        or "not authorized" in error_lower
        or "access denied" in error_lower
    )


def _tool_meta(tool) -> dict:
    """Return the MCP ``_meta`` of a tool loaded through ``langchain.mcp`` (stored under ``metadata["mcp"]["tool"]``)."""
    return ((((tool.metadata or {}).get("mcp") or {}).get("tool") or {}).get("_meta")) or {}


async def create_mcp_client(agent_config: AgentConfig, websocket: WebSocket | None = None) -> MCPAdapter:
    """
    Create an MCP client for the agent based on the agent configuration.
    
    This function checks the authentication type specified in the agent configuration and
    constructs the appropriate MCP client with the correct URL and headers for connecting 
    to the MCP server.
    
    Args:
        agent_config: The configuration object for the agent, containing authentication details.
        websocket: Optional WebSocket connection used to extract cookies and URL information for Rancher authentication.
                   If not provided, falls back to environment variables only.
    
    Returns:
        MCPAdapter: A configured MCP adapter ready to connect to the server and load its tools.
    
    Note:
        - For Rancher authentication, extracts R_SESS cookie and uses RANCHER_URL
        - Respects INSECURE_SKIP_TLS environment variable to disable TLS certificate verification
        - For BASIC authentication, encodes credentials in the Authorization header
        - For NONE authentication, creates client with no additional headers
    """
    headers = {}
    oauth2_auth = None

    if agent_config.authentication == AuthenticationType.RANCHER:
        if websocket:
            cookies = websocket.cookies
            token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))
        else:
            token = os.environ.get("RANCHER_API_TOKEN", "")
        
        mcp_url = os.environ.get("MCP_URL", agent_config.mcp_url)
        if not mcp_url.startswith(("http://", "https://")):
            if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == 'true':
                mcp_url = "http://" + mcp_url
            else:
                mcp_url = "https://" + mcp_url
        headers = {
            "R_token": token,
        }
    elif agent_config.authentication == AuthenticationType.BASIC:
        mcp_url = agent_config.mcp_url
        try:
            credentials = get_basic_auth_credentials(agent_config.authentication_secret)
            headers = {
                "Authorization": f"Basic {credentials}"
            }
        except Exception as e:
            logging.error(f"Failed to get basic auth credentials: {str(e)}")

    elif agent_config.authentication == AuthenticationType.HEADER:
        mcp_url = agent_config.mcp_url
        try:
            headers = get_header_auth_headers(agent_config.authentication_secret)
        except Exception as e:
            logging.error(f"Failed to get header auth headers: {str(e)}")

    elif agent_config.authentication == AuthenticationType.OAUTH2:
        mcp_url = agent_config.mcp_url

        cookie_name = get_oauth_cookie_names(agent_config.name)["access_token"]
        oauth_token = websocket.cookies.get(cookie_name) if websocket else None

        mcp_url = agent_config.mcp_url
        oauth2_auth = _OAuth2BearerAuth(oauth_token)

    else:
        mcp_url = agent_config.mcp_url

    verify = None
    if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == "true":
        verify = False

    elif agent_config.ca_bundle_ref:
        try:
            ca_pem = get_ca_cert_from_secret(
                agent_config.ca_bundle_ref.name,
                agent_config.ca_bundle_ref.key,
            )
            verify = build_ca_ssl_context(ca_pem)
        except Exception as e:
            logging.error(f"Failed to load CA cert from secret '{agent_config.ca_bundle_ref.name}': {e}")

    transport = StreamableHttpTransport(mcp_url, headers=headers, auth=oauth2_auth, verify=verify)
    adapter = MCPAdapter(Client(transport))
    # Kept on the adapter so _load_mcp_tools can tell a 401 apart from other connection errors.
    adapter.oauth2_auth = oauth2_auth
    return adapter


def _update_agent_status(agent_cfg: AgentConfig, is_ready: bool, reason: str, message: str):
    """
    Update the status of an AIAgentConfig CRD in Kubernetes.
    
    Args:
        agent_cfg: The agent configuration object
        is_ready: Whether the agent is ready
        reason: Short reason for the status
        message: Detailed message about the status
    """
    # Only update if status has changed
    if agent_cfg.ready == is_ready:
        return
    
    try:
        _load_k8s_config()
        api = client.CustomObjectsApi()
        
        status = {
            'conditions': [{
                'type': 'Ready',
                'status': 'True' if is_ready else 'False',
                'reason': reason,
                'message': message,
                'lastTransitionTime': datetime.now(timezone.utc).isoformat()
            }],
            'phase': 'Ready' if is_ready else 'Failed'
        }
        
        api.patch_namespaced_custom_object_status(
            group='ai.cattle.io',
            version='v1alpha1',
            namespace=NAMESPACE,
            plural='aiagentconfigs',
            name=agent_cfg.name,
            body={'status': status}
        )
        logging.info(f"Updated status for AIAgentConfig '{agent_cfg.name}' to {status['phase']}")
    except Exception as e:
        logging.error(f"Failed to update status for AIAgentConfig '{agent_cfg.name}': {str(e)}")


async def _load_agents_for_user(websocket: WebSocket) -> list[AgentConfig]:
    """
    Load the agent configs the requesting user is allowed to access.

    With RBAC enabled, the scoped list is fetched directly from Rancher (which
    resolves the user's roles and group memberships), so cluster admins see all
    agents and others see only what they're granted. With RBAC disabled, the full
    enabled roster is loaded from the CRDs.

    Fails closed: if the user's identity/permissions cannot be resolved, a
    ``NoAgentAvailableError`` is raised so nothing is exposed.
    """
    if not rbac_enabled():
        return load_agent_configs()

    token = os.environ.get("RANCHER_API_TOKEN", websocket.cookies.get("R_SESS", ""))
    if not token:
        raise NoAgentAvailableError(
            "Unable to determine your identity from the Rancher session. "
            "Please sign in to Rancher again and retry."
        )

    try:
        return await accessible_agent_configs(token)
    except RBACError as e:
        logging.error(f"RBAC resolution failed: {e}")
        raise NoAgentAvailableError(
            "Unable to verify your permissions with Rancher right now. "
            "Please try again shortly or contact your administrator."
        ) from e