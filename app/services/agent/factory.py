import os
import logging
import json

from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass

from .loader import AuthenticationType, load_agent_configs
from .child import create_child_agent
from .parent import create_parent_agent, ChildAgent
from ..rag import fleet_documentation_retriever, rancher_documentation_retriever
from fastapi import  Request, WebSocket
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.language_models.llms import BaseLanguageModel

@dataclass
class AgentConfig:
    """Configuration for a child agent."""
    name: str
    description: str
    system_prompt: str
    mcp_url: str
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentConfig':
        """Create an AgentConfig from a dictionary."""
        return cls(
            name=data.get("agent", ""),
            description=data.get("description", ""),
            system_prompt=data.get("systemPrompt", ""),
            mcp_url=data.get("mcpURL", "")
        )

def parse_agent_configs(json_str: str) -> list[AgentConfig]:
    """
    Parse JSON string into a list of AgentConfig objects.
    
    Args:
        json_str: JSON string containing agent configurations
    
    Returns:
        List of AgentConfig objects
    """
    try:
        data = json.loads(json_str)
        return [AgentConfig.from_dict(agent_dict) for agent_dict in data]
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse agent configs: {e}")
        return []
    except Exception as e:
        logging.error(f"Error creating agent configs: {e}")
        return []

@asynccontextmanager
async def create_agent(llm: BaseLanguageModel, websocket: WebSocket):
    """
    Create and configure an agent based on the available builtin agents.
    
    This factory function determines whether to create a parent agent with multiple
    child agents or a single child agent, depending on the agent configurations loaded
    from CRDs or fallback to built-in agents.
    
    Args:
        llm: The language model to use for agent reasoning and responses.
        websocket: WebSocket connection used to extract authentication cookies and URL info.
    
    Yields:
        CompiledStateGraph: Either a parent agent managing multiple child agents,
            or a single child agent for the Rancher Core Agent.
    
    Note:
        This is an async context manager that properly manages the lifecycle of
        MCP (Model Context Protocol) connections and tools.
    """
    checkpointer = websocket.app.memory_manager.get_checkpointer()
    
    # Load agent configs from CRDs (or create defaults if none exist)
    builtin_agents = load_agent_configs()
    
    if len(builtin_agents) == 0:
        logging.error("Failed to load any agent configurations from CRDs")
        raise RuntimeError(
            "No agent configurations available. "
            "Please ensure AIAgentConfig CRDs are properly installed in the cattle-ai-agent-system namespace "
            "or check the Kubernetes API server connection."
        )

    logging.info(f"Loaded {len(builtin_agents)} agent configuration(s)")
    
    if len(builtin_agents) > 1:
        logging.info(f"Multi-agent setup detected, creating parent agent with {len(builtin_agents)} agents.")        
        async with AsyncExitStack() as stack:
            child_agents = []
            for agent_cfg in builtin_agents:
                tools = await _create_mcp_tools(stack, websocket, agent_cfg)
                child_agents.append(ChildAgent(
                    name=agent_cfg.name,
                    description=agent_cfg.description,
                    agent=create_child_agent(llm, tools, agent_cfg.system_prompt, checkpointer, agent_cfg)
                ))
            parent_agent = create_parent_agent(llm, child_agents, checkpointer)

            yield parent_agent
    else:
        logging.info("Single agent configuration detected")
        agent_cfg = builtin_agents[0]
        
        async with AsyncExitStack() as stack:
            tools = await _create_mcp_tools(stack, websocket, agent_cfg)
            agent = create_child_agent(llm, tools, agent_cfg.system_prompt, checkpointer, agent_cfg)
            
            yield agent


async def _create_mcp_tools(stack: AsyncExitStack, websocket: WebSocket, agent_config: AgentConfig) -> list:    
    """
    Create and configure MCP (Model Context Protocol) tools for an agent.
    
    This function establishes a connection to an MCP server and loads the available
    tools. It handles authentication based on the agent configuration, supporting
    both Rancher-specific authentication and generic configurations.
    
    Args:
        stack: AsyncExitStack for managing async context resources and cleanup.
        websocket: WebSocket connection used to extract cookies and URL information
            for Rancher authentication.
        agent_config: Configuration object containing agent details including
            authentication type, MCP URL, and agent name.
    
    Returns:
        list: A list of tools loaded from the MCP server. If RAG is enabled for
            the Rancher Core Agent, documentation retriever tools are prepended.
            Returns an empty list if connection to MCP server fails.
    
    Note:
        - For Rancher authentication, extracts R_SESS cookie and uses RANCHER_URL
        - Respects INSECURE_SKIP_TLS environment variable for HTTP/HTTPS selection
        - Adds RAG documentation retrievers if ENABLE_RAG is true for Rancher Core Agent
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
    else:
        mcp_url = agent_config.mcp_url
        headers = {}

    read, write, _ = await stack.enter_async_context(
        streamablehttp_client(
            url=mcp_url,
            headers=headers
        )
    )
    
    try:
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = await load_mcp_tools(session)
    except Exception as e: #TODO not catching exception here!!!
        logging.error(f"Failed to connect to MCP server at {mcp_url}: {e}")
        logging.warning(f"Agent '{agent_config.name}' will run without MCP tools")
        tools = []

    if agent_config.name == "Rancher Core Agent" and os.environ.get("ENABLE_RAG", "false").lower() == "true":
        tools = [fleet_documentation_retriever, rancher_documentation_retriever] + tools

    return tools

