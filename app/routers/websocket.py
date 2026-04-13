import os
import uuid
import logging
import json

from ..dependencies import get_llm
from ..services.agent.factory import NoAgentAvailableError, create_agent
from ..services.agent.loader import load_agent_configs, AuthenticationType
from ..services.oauth2 import (
    OAuthClient,
    OAuthDiscoveryError,
    OAuthDiscoveryResult,
    discover_oauth_metadata,
    get_redirect_uri,
    get_oauth_client_credentials,
    generate_oauth_cookie_key,
    get_oauth_cookie_names,
)
from .oauth2 import oauth_state_store
from ..services.auth import get_user_id
from dataclasses import dataclass
from fastapi import APIRouter
from fastapi import  WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState
from langgraph.graph.state import CompiledStateGraph
from langfuse.langchain import CallbackHandler
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command

router = APIRouter()

async def get_user_id_from_websocket(websocket: WebSocket) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token from the WebSocket cookies.
    """
    cookies = websocket.cookies
    rancher_url = os.environ.get("RANCHER_URL","https://"+websocket.url.hostname)
    token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))

    return await get_user_id(rancher_url, token)

def build_chat_metadata(thread_id: str, agents_metadata: list[dict], websocket: WebSocket) -> str:
    """
    Builds the chat metadata to be sent to the client upon WebSocket connection.
    This can include information about available agents, tools, storage type or any other relevant data
    that the client might need to know before starting the conversation.

    Returns:
        A dictionary containing the chat metadata.
    """
    storage_type = websocket.app.memory_manager.storage_type.value
    agents = json.dumps(agents_metadata)

    return f'<chat-metadata>{{"chatId": "{thread_id}", "agents": {agents}, "storageType": "{storage_type}"}}</chat-metadata>'

@dataclass
class WebSocketRequest:
    """Represents a parsed WebSocket request from the client."""
    prompt: str
    user_input: str
    context: dict
    tags: list[str] = None
    labels: dict = None
    agent: str = ""

async def _discover_and_authenticate_oauth(websocket: WebSocket) -> dict[str, str]:
    """
    Discover OAuth endpoints for OAUTH2 agents and perform authentication.

    Follows the MCP specification for OAuth discovery:
    1. Load agent configs to identify agents requiring OAUTH2
    2. For each OAUTH2 agent, discover OAuth metadata from its MCP server URL
    3. Check httponly cookies for cached tokens before triggering interactive authentication
    4. If tokens are expired, attempt a silent refresh using the refresh token cookie
    5. Fall back to the full interactive OAuth flow only when no valid tokens are available

    Args:
        websocket: The WebSocket connection for communication.

    Returns:
        A dictionary mapping agent names to their OAuth access tokens.
    """
    agents = load_agent_configs()
    oauth_agents = [a for a in agents if a.authentication == AuthenticationType.OAUTH2]

    if not oauth_agents:
        return {}

    tokens = {}
    redirect_uri = get_redirect_uri(websocket)
    cookies = websocket.cookies

    # Discover OAuth metadata for each unique MCP server URL
    discovered: dict[str, OAuthDiscoveryResult] = {}
    for agent_cfg in oauth_agents:
        if agent_cfg.mcp_url not in discovered:
            logging.info(f"Discovering OAuth metadata for agent '{agent_cfg.name}' at {agent_cfg.mcp_url}")
            discovered[agent_cfg.mcp_url] = await discover_oauth_metadata(agent_cfg.mcp_url)

    # Authenticate per unique authorization server to avoid duplicate flows
    authenticated_servers: dict[str, str] = {}
    for agent_cfg in oauth_agents:
        discovery = discovered[agent_cfg.mcp_url]
        auth_server_key = discovery.authorization_endpoint

        if auth_server_key not in authenticated_servers:
            # 1. Check cookies for a valid (non-expired) access token
            cookie_key = generate_oauth_cookie_key(auth_server_key)
            cookie_names = get_oauth_cookie_names(cookie_key)

            cached_token = _get_valid_token_from_cookies(cookies, cookie_names)
            if cached_token:
                logging.info(f"Using cached OAuth token from cookies for {auth_server_key}")
                authenticated_servers[auth_server_key] = cached_token
            else:
                # 2. Try silent refresh if a refresh token cookie exists
                refreshed_token = await _try_refresh_token(
                    cookies, cookie_names, discovery.token_endpoint, agent_cfg.oauth_secret
                )
                if refreshed_token:
                    logging.info(f"Refreshed OAuth token from cookies for {auth_server_key}")
                    authenticated_servers[auth_server_key] = refreshed_token
                else:
                    # 3. Fall back to interactive OAuth flow
                    access_token, _ = await _perform_oauth_authentication(
                        websocket, discovery, redirect_uri, agent_cfg.oauth_secret
                    )
                    authenticated_servers[auth_server_key] = access_token

        tokens[agent_cfg.name] = authenticated_servers[auth_server_key]

    return tokens


def _get_valid_token_from_cookies(
    cookies: dict[str, str], cookie_names: dict[str, str]
) -> str | None:
    """
    Check if cookies contain an OAuth access token.

    Returns the access token string if present, None otherwise.
    """
    return cookies.get(cookie_names["access_token"]) or None


async def _try_refresh_token(
    cookies: dict[str, str],
    cookie_names: dict[str, str],
    token_endpoint: str,
    oauth_secret: str | None,
) -> str | None:
    """
    Attempt to refresh an expired access token using the refresh token from cookies.

    Requires a refresh token cookie, a token endpoint, and client credentials
    from a Kubernetes secret. Returns the new access token or None if refresh fails.
    """
    refresh_tok = cookies.get(cookie_names["refresh_token"])

    if not refresh_tok:
        return None

    if not oauth_secret:
        logging.debug("Cannot refresh token: no oauth_secret configured for agent")
        return None

    try:
        credentials = get_oauth_client_credentials(oauth_secret)
        oauth_client = OAuthClient(
            client_id=credentials.client_id,
            client_secret=credentials.client_secret,
        )
        new_token = await oauth_client.refresh_token(token_endpoint, refresh_tok)
        return new_token.get("access_token")
    except Exception as e:
        logging.warning(f"Failed to refresh OAuth token: {e}")
        return None


async def _perform_oauth_authentication(
    websocket: WebSocket,
    discovery: OAuthDiscoveryResult,
    redirect_uri: str,
    authentication_secret: str | None = None,
) -> tuple[str, str]:
    """
    Performs OAuth authentication flow using discovered MCP OAuth endpoints.

    Creates an OAuth client using one of these strategies (in order):
    1. Pre-configured credentials from a Kubernetes secret (if authentication_secret is set)
    2. Dynamic Client Registration (RFC 7591) if the auth server supports it

    Then generates an authorization URL, sends it to the client via WebSocket,
    and waits for the access and refresh tokens.

    Args:
        websocket: The WebSocket connection for communication.
        discovery: The discovered OAuth metadata from the MCP server.
        redirect_uri: The OAuth callback redirect URI.
        authentication_secret: Optional Kubernetes secret name containing client credentials.

    Returns:
        A tuple of (access_token, refresh_token) received from the client.
    """
    scope = " "

    # Strategy 1: Use pre-configured client credentials from K8s secret
    if authentication_secret:
        try:
            credentials = get_oauth_client_credentials(authentication_secret)
            # Scopes from the secret take precedence over discovered scopes
            if credentials.scopes:
                scope = credentials.scopes
            oauth_client = OAuthClient(client_id=credentials.client_id, client_secret=credentials.client_secret, scope=scope)
            logging.info(f"Using pre-configured OAuth credentials from secret '{authentication_secret}'")
        except Exception as e:
            logging.warning(f"Failed to load OAuth credentials from secret '{authentication_secret}': {e}")
            raise OAuthDiscoveryError(
                f"Failed to load OAuth client credentials from secret '{authentication_secret}': {e}"
            )
    # Strategy 2: Dynamic Client Registration
    elif discovery.registration_endpoint:
        logging.info(f"Performing dynamic client registration at {discovery.registration_endpoint}")
        oauth_client = await OAuthClient.from_dynamic_registration(
            registration_endpoint=discovery.registration_endpoint,
            redirect_uri=redirect_uri,
            scope=scope,
        )
    else:
        raise OAuthDiscoveryError(
            "Authorization server does not support dynamic client registration and "
            "no OAuth client credentials are configured. "
            "Please create a Kubernetes secret with 'clientId' and 'clientSecret' keys "
            "and reference it via the 'oauthSecret' field in the AIAgentConfig."
        )

    url, verifier, state = await oauth_client.get_auth_url(
        discovery.authorization_endpoint, redirect_uri
    )

    # Store state for the callback handler
    # TODO - This is a temporary in-memory store. This needs to be improved!
    cookie_key = generate_oauth_cookie_key(discovery.authorization_endpoint)
    oauth_state_store[state] = {
        "verifier": verifier,
        "oauth_client": oauth_client,
        "token_endpoint": discovery.token_endpoint,
        "cookie_key": cookie_key,
    }

    await websocket.send_text(f'<authentication>{{"type": "oauth2", "url": "{str(url)}"}}</authentication>')
    token_response = await websocket.receive_text()
    token_data = json.loads(token_response)

    return token_data["access_token"], token_data.get("refresh_token", "")

@router.websocket("/v1/ws/messages")
@router.websocket("/v1/ws/messages/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str = None, llm: BaseLanguageModel = Depends(get_llm)):
    """
    WebSocket endpoint for the agent.
    
    Accepts a WebSocket connection, sets up the agent and
    handles the back-and-forth communication with the client.
    """
    
    user_id = await get_user_id_from_websocket(websocket)
    
    if not thread_id:
        thread_id = str(uuid.uuid4())
    logging.debug(f"Starting websocket session with thread_id: {thread_id}, user_id: {user_id}")
    
    await websocket.accept()
    logging.debug("ws/messages connection opened")

    # Discover and authenticate OAuth for any OAUTH2 agents
    try:
        tokens = await _discover_and_authenticate_oauth(websocket)
    except (OAuthDiscoveryError, Exception) as e:
        logging.error(f"OAuth discovery/authentication failed: {e}")
        await websocket.send_text(f'<chat-error>{json.dumps({"message": f"OAuth authentication failed: {str(e)}"})}</chat-error>')
        await websocket.close()
        return

    try:
        agent, agents_metadata =  await create_agent(llm=llm, websocket=websocket, tokens=tokens) 
    except NoAgentAvailableError as e:
        logging.error(f"Error creating agent: {e}")
        await websocket.send_text(f'<chat-error>{json.dumps({"message": str(e)})}</chat-error>')
        await websocket.close()
        return

    await websocket.send_text(build_chat_metadata(thread_id, agents_metadata, websocket))

    base_config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
    }

    if os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_HOST"):
        langfuse_handler = CallbackHandler()
        base_config["callbacks"] = [langfuse_handler]

    while True:
        try:
            request = await websocket.receive_text()
            request_id = str(uuid.uuid4())

            ws_request = _parse_websocket_request(request)
            config = _build_config(base_config, request_id, ws_request)
            input_data = await _build_input_data(agent, config, ws_request)

            await _call_agent(
                agent=agent,
                input_data=input_data,
                config=config,
                websocket=websocket)
            
        except WebSocketDisconnect:
            logging.info(f"Client {websocket.client.host} disconnected.")

            break
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(f'<error>{json.dumps({"message": str(e)})}</error>')
            else:
                break
        finally:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("</message>")

async def _call_agent(
    agent: CompiledStateGraph,
    input_data: any, 
    config: dict,
    websocket: WebSocket,
) -> None:
    """
    Streams the agent's response to a WebSocket connection, handling interruptions.
    
    Args:
        agent: The compiled LangGraph agent.
        input_data: The input data for the agent's run.
        config: The run configuration.
        websocket: The WebSocket connection.
        stream_mode: The types of events to stream from the agent.
    """

    await websocket.send_text("<message>")
    
    async for stream in agent.astream_events(
        input_data,
        config=config,
        stream_mode=["updates", "messages", "custom", "events"],
    ):
        if stream["event"] == "on_chat_model_stream":
            if text := _extract_streaming_text(stream):
                await websocket.send_text(text)
        
        if stream["event"] == "on_custom_event":
            await websocket.send_text(stream["data"])
    
        if stream["event"] == "on_chain_stream":
            if interrupt_value := _extract_interrupt_value(stream):
                await websocket.send_text(interrupt_value)

def _extract_streaming_text(stream: dict) -> str | None:
    """
    Extracts text content from a chat model stream event.
    
    Only extracts text from 'agent' or 'model' nodes to avoid streaming
    intermediate processing steps.
    
    Args:
        stream: The stream event dictionary from astream_events.
        
    Returns:
        The extracted text content, or None if not applicable.
    """
    STREAMABLE_NODES = ("agent", "model")
    
    node = stream.get("metadata", {}).get("langgraph_node")
    if node not in STREAMABLE_NODES:
        return None
    
    chunk = stream.get("data", {}).get("chunk")
    if not chunk or not chunk.content:
        return None
    
    return _extract_text_from_chunk_content(chunk.content)

def _extract_interrupt_value(stream: dict) -> str | None:
    """
    Extracts the interrupt value from a chain stream event.
    
    LangGraph sends interrupt signals through on_chain_stream events with a specific
    structure: data.chunk is a tuple like ("updates", {"__interrupt__": [Interrupt(...)]})
    
    Args:
        stream: The stream event dictionary from astream_events.
        
    Returns:
        The interrupt value string if present, None otherwise.
    """
    data = stream.get("data")
    if not isinstance(data, dict):
        return None
    
    chunk = data.get("chunk")
    if not isinstance(chunk, (list, tuple)) or len(chunk) < 2:
        return None
    
    if chunk[0] != "updates":
        return None
    
    updates = chunk[1]
    if not isinstance(updates, dict):
        return None
    
    interrupts = updates.get("__interrupt__", [])
    if not interrupts:
        return None
    
    return interrupts[0].value or None
    
def _extract_text_from_chunk_content(chunk_content: any) -> str:
    """
    Extracts the text content from a chunk received from the LLM.

    This function handles different formats that LLMs might return:
    1. A list of dictionaries, where each dictionary contains a 'text' key.
       This is common for models like Gemini that might structure their output.
    2. A single dictionary with a 'text' key.
    3. A simple string or other direct content.

    Args:
        chunk_content: The content field from an LLM chunk.

    Returns:
        str: The extracted text content, or an empty string if no text is found.
    """
    if isinstance(chunk_content, list):
        return "".join([item.get("text", "") for item in chunk_content if isinstance(item, dict)])
    elif isinstance(chunk_content, dict) and "text" in chunk_content:
        return chunk_content["text"]
    
    return str(chunk_content) if chunk_content is not None else ""

def _parse_websocket_request(request: str) -> WebSocketRequest:
    """
    Parses the incoming websocket request and enriches the prompt with context.

    The request can be a JSON string with 'prompt', 'context', and 'agent' keys,
    or a plain text string. If context is provided, it will be appended to the
    prompt to guide tool call parameter population.

    Args:
        request: The raw request string from the websocket.

    Returns:
        A WebSocketRequest object containing the parsed data with enriched prompt.
    """
    try:
        json_request = json.loads(request)
        user_input = json_request.get("prompt", "")
        context = json_request.get("context", {})
        
        # Enrich prompt with context if present
        prompt = user_input
        if context:
            context_parts = [f"{key}:{value}" for key, value in context.items()]
            context_suffix = (
                ". Use the following parameters to populate tool calls when appropriate. \n"
                "Only include parameters relevant to the user's request "
                "(e.g., omit namespace for cluster-wide operations). \n"
                f"Parameters (separated by ;): \n {';'.join(context_parts)};"
            )
            prompt += context_suffix
        
        return WebSocketRequest(
            prompt=prompt,
            user_input=user_input,
            context=context,
            tags=json_request.get("tags", []),
            labels=json_request.get("labels", {}),
            agent=json_request.get("agent", "")
        )
    except json.JSONDecodeError:
        return WebSocketRequest(prompt=request, user_input="", context={}, tags=[], labels={}, agent="")

def _build_config(base_config: dict, request_id: str, ws_request: WebSocketRequest) -> dict:
    """
    Builds the configuration dictionary for an agent run.
    
    Merges base configuration with request-specific settings including:
    - request_id for tracking individual requests
    - agent selection (if specified)
    - ephemeral flag handling (prevents memory storage)
    
    Args:
        base_config: The base configuration with thread_id and user_id.
        request_id: Unique identifier for this request.
        ws_request: The parsed WebSocket request.
        
    Returns:
        A configuration dictionary ready for agent.astream_events.
    """
    config = {
        **base_config,
        "configurable": {**base_config["configurable"]},
    }

    config["configurable"]["request_id"] = request_id
    config["configurable"]["request_metadata"] = {
        "agent": ws_request.agent,
        "user_input": ws_request.user_input,
        "context": ws_request.context,
        "labels": ws_request.labels,
        "tags": ws_request.tags
    }

    if ws_request.agent:
        config["configurable"]["agent"] = ws_request.agent
    else:
        config["configurable"]["agent"] = ""

    # Exclude "ephemeral" messages from being stored in memory
    tags = ws_request.tags or []
    if "ephemeral" in tags:
        config["configurable"]["thread_id"] = ""
    
    return config

async def _build_input_data(agent: CompiledStateGraph, config: dict, ws_request: WebSocketRequest) -> dict | Command:
    """
    Builds the input data for the agent, handling interrupt resumption.
    
    If the agent is waiting on an interrupt, resumes with the user's response.
    Otherwise, creates a new user message.
    
    Args:
        agent: The compiled LangGraph agent.
        config: The configuration dictionary for the agent run.
        ws_request: The parsed WebSocket request.
        
    Returns:
        Either a Command to resume an interrupt, or a messages dict for a new turn.
    """
    state = await agent.aget_state(config=config)
    
    if state.interrupts:
        return Command(resume=ws_request.prompt)

    input_messages = [
        HumanMessage(
            content=ws_request.prompt,
            additional_kwargs={
                "request_id": config["configurable"]["request_id"],
                "request_metadata": config["configurable"]["request_metadata"]
            }
        )
    ]

    return {
        "messages": input_messages,
    }
