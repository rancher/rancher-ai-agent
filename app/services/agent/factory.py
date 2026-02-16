import os
import logging
from datetime import datetime, timezone

from kubernetes import client, config
from .root import create_root_agent
from .loader import AuthenticationType, load_agent_configs, AgentConfig, get_basic_auth_credentials
from .child import create_child_agent
from .parent import create_parent_agent, ChildAgent
from fastapi import  WebSocket
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import Checkpointer

NAMESPACE = "cattle-ai-agent-system"


class NoAgentAvailableError(Exception):
    """Exception raised when loading MCP tools fails."""
    pass


async def create_agent(llm: BaseLanguageModel, websocket: WebSocket):
    """
    Create and configure an agent based on the available builtin agents.
    
    This factory function determines whether to create a parent agent with multiple
    child agents or a single child agent, depending on the agent configurations loaded
    from CRDs or fallback to built-in agents.
    
    Args:
        llm: The language model to use for agent reasoning and responses.
        websocket: WebSocket connection used to extract authentication cookies and URL info.
    
    Returns:
        CompiledStateGraph: Either a parent agent managing multiple child agents,
            or a single child agent for the Rancher Core Agent.
    
    Note:
        This is an async context manager that properly manages the lifecycle of
        MCP (Model Context Protocol) connections and tools.
    """
    checkpointer = websocket.app.memory_manager.get_checkpointer()
    
    # Load agent configs from CRDs (or create defaults if none exist)
    agents = load_agent_configs()
    
    if len(agents) == 0:
        logging.error("Failed to load any agent configurations from CRDs")
        raise NoAgentAvailableError("No agent configurations available. ")

    logging.info(f"Loaded {len(agents)} agent configuration(s)")
    
    if len(agents) > 1:
        logging.info(f"Multi-agent setup detected, creating parent agent with {len(agents)} agents.")  
        child_agents = []
        agents_metadata = []

        for agent_cfg in agents:
            client = create_mcp_client(agent_cfg, websocket)
            try:
                tools = await client.get_tools()
                # Filter tools by toolset if specified in agent config
                if agent_cfg.toolset:
                    tools = [
                        tool for tool in tools 
                        if tool.metadata.get("_meta", {}).get("toolset") == agent_cfg.toolset
                    ]
                    logging.debug(f"Filtered {len(tools)} tools for toolset '{agent_cfg.toolset}'")

                child_agents.append(ChildAgent(
                    config=agent_cfg,
                    agent=create_child_agent(llm, tools, agent_cfg.system_prompt, checkpointer, agent_cfg, all_children_agents=agents)
                ))
                
                _update_agent_status(agent_cfg, True, 'MCPConnectionSucceeded', 'MCP tools loaded successfully')

                agents_metadata.append({
                    "name": agent_cfg.name,
                    "status": "active",
                })
            except* Exception as eg:
                error_message = ""
                for e in eg.exceptions:
                    error_message += f"{str(e)} "
                logging.error(f"Failed to load MCP tools for agent '{agent_cfg.name}': {error_message}")
                
                _update_agent_status(agent_cfg, False, 'MCPConnectionFailed', f"Failed to load MCP tools: {error_message}")

                agents_metadata.append({
                    "name": agent_cfg.name,
                    "status": "error",
                    "description": f"{error_message}"
                })

        if len(child_agents) == 0:
            logging.error("Failed to create any child agents due to MCP connection issues")
            raise NoAgentAvailableError("No agents could be created. Please check the MCP server connections and configurations for each agent.")
        
        if len(child_agents) == 1:
            logging.warning("Only one child agent was successfully created. Returning the child agent directly instead of a parent agent.")
            return await _create_single_agent(llm, child_agents[0].config, checkpointer, websocket), agents_metadata

        parent_agent = create_parent_agent(llm, child_agents, checkpointer)

        return parent_agent, agents_metadata
    else:
        return await _create_single_agent(llm, agents[0], checkpointer, websocket), [{"name": agents[0].name, "status": "active"}]



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
        - Respects INSECURE_SKIP_TLS environment variable for HTTP/HTTPS selection
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
        if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == "true":
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

    else:
        mcp_url = agent_config.mcp_url

    return MultiServerMCPClient({
        agent_config.name: {
            "url": mcp_url,
            "transport": "streamable_http",
            "headers": headers,
        },
    })


async def _create_single_agent(
    llm: BaseLanguageModel,
    agent_cfg: AgentConfig,
    checkpointer: Checkpointer,
    websocket: WebSocket
) -> tuple:
    """
    Create a single child agent based on the provided agent configuration.
    
    This function is used when only one agent configuration is available. It establishes
    the MCP connection, loads the tools, and creates a child agent accordingly.
    
    Args:
        llm: The language model to use for the agent.
        agent_cfg: The configuration object for the agent, containing MCP connection details and system prompt.
        checkpointer: Checkpointer for persisting agent state.
        websocket: WebSocket connection used to extract cookies and URL information for Rancher authentication.
    """

    client = create_mcp_client(agent_cfg, websocket)
    try:
        tools = await client.get_tools()
        _update_agent_status(agent_cfg, True, 'MCPConnectionSucceeded', 'MCP tools loaded successfully')
    except* Exception as eg:
        error_message = ""
        for e in eg.exceptions:
            error_message += f"{str(e)} "
        logging.error(f"Failed to load MCP tools for agent '{agent_cfg.name}': {error_message}")
        
        _update_agent_status(agent_cfg, False, 'MCPConnectionFailed', f"Failed to load MCP tools: {error_message}")
        
        raise NoAgentAvailableError(
            f"Failed to load MCP tools for all enabled agents.\\n"
            f"Please check the AI Agents configuration and ensure the MCP server is accessible with the provided connection details."
        )

    return create_root_agent(llm, tools, agent_cfg.system_prompt, checkpointer, agent_cfg)

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
        # Load in-cluster config (works when running in a pod)
        try:
            config.load_incluster_config()
        except config.ConfigException:
            # Fall back to kubeconfig (for local development)
            config.load_kube_config()
        
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
