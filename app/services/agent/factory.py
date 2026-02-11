import os
import logging


from .root import create_root_agent
from .loader import AuthenticationType, load_agent_configs, AgentConfig, get_basic_auth_credentials
from .child import create_child_agent
from .parent import create_parent_agent, ChildAgent
from fastapi import  WebSocket
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import Checkpointer

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
            mcp_url, header = get_mcp_url_and_headers(agent_cfg, websocket)
            client = MultiServerMCPClient({
                agent_cfg.name: {
                    "url": mcp_url,
                    "transport": "streamable_http",
                    "headers": header,
                },
            })      
            try:
                tools = await client.get_tools()
        
                child_agents.append(ChildAgent(
                    name=agent_cfg.name,
                    description=agent_cfg.description,
                    agent=create_child_agent(llm, tools, agent_cfg.system_prompt, checkpointer, agent_cfg, all_children_agents=agents)
                ))

                agents_metadata.append({
                    "name": agent_cfg.name,
                    "status": "active",
                })
            except* Exception as eg:
                error_message = ""
                for e in eg.exceptions:
                    error_message += f"{str(e)} "
                logging.error(f"Failed to load MCP tools for agent '{agent_cfg.name}': {error_message}")

                agents_metadata.append({
                    "name": agent_cfg.name,
                    "status": "error",
                    "description": f"{error_message}"
                })

        if len(child_agents) == 0:
            logging.error("Failed to create any child agents due to MCP connection issues")
            raise NoAgentAvailableError(
                "No agents could be created. Please check the MCP server connections and configurations for each agent."
            )
        if len(child_agents) == 1:
            logging.warning("Only one child agent was successfully created. Returning the child agent directly instead of a parent agent.")
            return await _create_single_agent(llm, agents[0], checkpointer, websocket)

        parent_agent = create_parent_agent(llm, child_agents, checkpointer)

        return parent_agent, agents_metadata
    else:
        return await _create_single_agent(llm, agents[0], checkpointer, websocket)

def get_mcp_url_and_headers(agent_config: AgentConfig, websocket: WebSocket) -> tuple[str, dict]:
    """
    Determine the MCP URL and headers for authentication based on the agent configuration.
    
    This function checks the authentication type specified in the agent configuration and
    constructs the appropriate MCP URL and headers for connecting to the MCP server.
    
    Args:
        agent_config: The configuration object for the agent, containing authentication details.
        websocket: WebSocket connection used to extract cookies and URL information for Rancher authentication.
    
    Returns:
        tuple: A tuple containing the MCP URL (str) and a dictionary of headers for authentication.
    
    Note:
        - For Rancher authentication, extracts R_SESS cookie and uses RANCHER_URL
        - Respects INSECURE_SKIP_TLS environment variable for HTTP/HTTPS selection
        - For BASIC authentication, encodes credentials in the Authorization header
        - For NONE authentication, returns the MCP URL with no additional headers
    """
    if agent_config.authentication == AuthenticationType.RANCHER:
        cookies = websocket.cookies
        rancher_url = os.environ.get("RANCHER_URL","https://"+websocket.url.hostname)
        token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))
        mcp_url = os.environ.get("MCP_URL", agent_config.mcp_url)
        if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == "true":
            mcp_url = "http://" + mcp_url
        else:
            mcp_url = "https://" + mcp_url
        headers={
                "R_token": token,
                "R_url": rancher_url
            }
    elif agent_config.authentication == AuthenticationType.BASIC:
        mcp_url = agent_config.mcp_url
        credentials = get_basic_auth_credentials(agent_config.authentication_secret)
        headers = {
            "Authorization": f"Basic {credentials}"
        }

    else:
        mcp_url = agent_config.mcp_url
        headers = {}

    return mcp_url, headers

async def _create_single_agent(llm: BaseLanguageModel, agent_cfg: AgentConfig, checkpointer: Checkpointer, websocket: WebSocket) -> ChildAgent:
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

    mcp_url, header = get_mcp_url_and_headers(agent_cfg, websocket)
    client = MultiServerMCPClient({
        agent_cfg.name: {
            "url": mcp_url,
            "transport": "streamable_http",
            "headers": header,
        },
    })      
    try:
        tools = await client.get_tools()
    
    except* Exception as eg:
        error_message = ""
        for e in eg.exceptions:
            error_message += f"{str(e)} "
        logging.error(f"Failed to load MCP tools for agent '{agent_cfg.name}': {error_message}")
        raise NoAgentAvailableError(
            f"Failed to load MCP tools for agent '{agent_cfg.name}'. Please check the MCP server connection and configuration. Error details: {error_message}"
        )

    return create_root_agent(llm, tools, agent_cfg.system_prompt, checkpointer, agent_cfg), [{
            "name": agent_cfg.name,
            "status": "active",
        }]