import os
import ssl
import logging
from datetime import datetime, timezone

import httpx
from kubernetes import client
from .loader import AuthenticationType, load_agent_configs, AgentConfig, get_basic_auth_credentials, get_header_auth_headers, get_ca_cert_from_secret, _load_k8s_config
from .supervisor import create_supervisor_agent, ChildAgent, SupervisorGraph
from .child import create_child_agent
from fastapi import  WebSocket
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import Checkpointer, CompiledStateGraph

NAMESPACE = "cattle-ai-agent-system"

class NoAgentAvailableError(Exception):
    """Exception raised when loading MCP tools fails."""
    pass


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
    agent_configs = load_agent_configs()

    if not agent_configs:
        logging.error("Failed to load any agent configurations from CRDs")
        raise NoAgentAvailableError("No agent configurations available.")

    logging.info(f"Loaded {len(agent_configs)} agent configuration(s)")

    if len(agent_configs) == 1:
        agent_cfg = agent_configs[0]
        tools = await _load_mcp_tools(agent_cfg, websocket)
        graph = create_child_agent(llm, tools, agent_cfg.system_prompt, checkpointer, agent_cfg)
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

    graph = create_supervisor_agent(llm, child_agents, checkpointer)
    supervisor = SupervisorGraph(
        graph=graph,
        child_agents={ca.config.name: ca.agent for ca in child_agents},
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
                agent=create_child_agent(llm, tools, agent_cfg.system_prompt, checkpointer, agent_cfg),
            ))
            agents_metadata.append({"name": agent_cfg.name, "status": "active"})
        except (NoAgentAvailableError, Exception) as e:
            logging.error(f"Failed to load MCP tools for agent '{agent_cfg.name}': {e}")
            agents_metadata.append({"name": agent_cfg.name, "status": "error", "description": str(e)})

    return child_agents, agents_metadata


async def _load_mcp_tools(agent_cfg: AgentConfig, websocket: WebSocket) -> list:
    """
    Connect to the MCP server for an agent config, load tools, filter by toolset,
    and update the CRD status accordingly.

    Raises:
        NoAgentAvailableError: If the MCP connection fails.
    """
    mcp_client = create_mcp_client(agent_cfg, websocket)
    try:
        tools = await mcp_client.get_tools()
    except* Exception as eg:
        error_message = " ".join(str(e) for e in eg.exceptions)
        _update_agent_status(agent_cfg, False, 'MCPConnectionFailed', f"Failed to load MCP tools: {error_message}")
        raise NoAgentAvailableError(
            f"Failed to load MCP tools for agent '{agent_cfg.name}': {error_message}\n"
            f"Please check the AI Agents configuration and ensure the MCP server is accessible."
        ) from eg

    if agent_cfg.toolset:
        tools = [t for t in tools if t.metadata.get("_meta", {}).get("toolset") == agent_cfg.toolset]
        logging.debug(f"Filtered {len(tools)} tools for toolset '{agent_cfg.toolset}'")

    _update_agent_status(agent_cfg, True, 'MCPConnectionSucceeded', 'MCP tools loaded successfully')
    return tools



def create_mcp_client(agent_config: AgentConfig, websocket: WebSocket | None = None) -> MultiServerMCPClient:
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
        MultiServerMCPClient: A configured MCP client ready to connect to the server.
    
    Note:
        - For Rancher authentication, extracts R_SESS cookie and uses RANCHER_URL
        - Respects INSECURE_SKIP_TLS environment variable to disable TLS certificate verification
        - For BASIC authentication, encodes credentials in the Authorization header
        - For NONE authentication, creates client with no additional headers
    """
    headers = {}

    if agent_config.authentication == AuthenticationType.RANCHER:
        if websocket:
            cookies = websocket.cookies
            rancher_url = os.environ.get("RANCHER_URL", "https://" + websocket.url.hostname)
            token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))
        else:
            rancher_url = os.environ.get("RANCHER_URL", "")
            token = os.environ.get("RANCHER_API_TOKEN", "")
        
        mcp_url = os.environ.get("MCP_URL", agent_config.mcp_url)
        if not mcp_url.startswith(("http://", "https://")):
            if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == 'true':
                mcp_url = "http://" + mcp_url
            else:
                mcp_url = "https://" + mcp_url
        headers = {
            "R_token": token,
            "R_url": rancher_url
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

    else:
        mcp_url = agent_config.mcp_url

    client_config: dict = {
        "url": mcp_url,
        "transport": "streamable_http",
        "headers": headers,
    }
    if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == "true":
        client_config["httpx_client_factory"] = _make_insecure_http_client
        
    elif agent_config.ca_bundle_ref:
        try:
            ca_pem = get_ca_cert_from_secret(
                agent_config.ca_bundle_ref.name,
                agent_config.ca_bundle_ref.key,
            )
            client_config["httpx_client_factory"] = _make_ca_httpx_factory(ca_pem)
        except Exception as e:
            logging.error(f"Failed to load CA cert from secret '{agent_config.ca_bundle_ref.name}': {e}")


    return MultiServerMCPClient({
        agent_config.name: client_config,
    })


def _make_ca_httpx_factory(ca_pem: str):
    """
    Return an httpx_client_factory that trusts *both* the system CAs and
    the supplied PEM-encoded CA certificate.

    The factory follows the ``McpHttpClientFactory`` protocol expected by
    ``langchain-mcp-adapters`` / the MCP Python SDK.
    """
    ctx = ssl.create_default_context()  # loads system CAs
    ctx.load_verify_locations(cadata=ca_pem)

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        kwargs: dict = {"follow_redirects": True, "verify": ctx}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if headers is not None:
            kwargs["headers"] = headers
        if auth is not None:
            kwargs["auth"] = auth
        return httpx.AsyncClient(**kwargs)

    return factory


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


def _make_insecure_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """httpx_client_factory that disables TLS certificate verification."""
    return httpx.AsyncClient(headers=headers or {}, timeout=timeout, auth=auth, verify=False)
