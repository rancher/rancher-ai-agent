import os
import uuid
import logging
import json

from httpx import HTTPStatusError

from ..dependencies import get_llm
from ..services.agent.factory import create_agent
from dataclasses import dataclass
from fastapi import APIRouter
from fastapi import  WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState
from langgraph.graph.state import CompiledStateGraph
from langfuse.langchain import CallbackHandler
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ..services.auth import get_user_id
from ..services.oauth2 import OAuthClient
from ..dependencies import get_llm
from .oauth2 import oauth_state_store

router = APIRouter()

#TODO move this
oauth_client = OAuthClient(client_id="client-fp76fxsc8q", client_secret="secret-rj2xsf9t65d4bdfzl7krdbpjlvtl78c4g2kgc4lr9gp245g4pblm8d84", metadata_url="https://raul-cabello.ngrok.app/oidc/.well-known/openid-configuration")
auth_endpoint = "https://raul-cabello.ngrok.app/oidc/authorize"
token_endpoint = "https://raul-cabello.ngrok.app/oidc/token"
redirect_uri = "http://localhost:8000/oauth/callback"

async def get_user_id_from_websocket(websocket: WebSocket) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token from the WebSocket cookies.
    """
    cookies = websocket.cookies
    rancher_url = os.environ.get("RANCHER_URL","https://"+websocket.url.hostname)
    token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))

    return await get_user_id(rancher_url, token)

async def _perform_oauth_authentication(websocket: WebSocket) -> tuple[str, str]:
    """
    Performs OAuth authentication flow with the client.
    
    Generates an OAuth URL, sends it to the client, and waits for the access and refresh tokens.
    
    Args:
        websocket: The WebSocket connection for communication.
        
    Returns:
        A tuple of (access_token, refresh_token) received from the client.
    """
    # Generate URL and the secret verifier
    url, verifier, state = await oauth_client.get_auth_url(auth_endpoint, redirect_uri)
    
    # Store the verifier and oauth_client for later use in the callback
    # TODO - This is a temporary in-memory store. This needs to be improved!
    oauth_state_store[state] = {
        "verifier": verifier,
        "oauth_client": oauth_client
    }
    await websocket.send_text(f'<authentication>{{"type": "oauth2", "url": "{str(url)}"}}</authentication>')
    token_response = await websocket.receive_text()
    token_data = json.loads(token_response)
    
    return token_data["access_token"], token_data["refresh_token"]

@dataclass
class WebSocketRequest:
    """Represents a parsed WebSocket request from the client."""
    prompt: str
    user_input: str
    context: dict
    tags: list[str] = None
    agent: str = ""


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
    
    await websocket.send_text(f'<chat-metadata>{{"chatId": "{thread_id}"}}</chat-metadata>')

    access_token, refresh_token = await _perform_oauth_authentication(websocket)
    await _handle_agent_session(llm, websocket, thread_id, user_id, access_token=access_token, refresh_token=refresh_token)

async def _handle_agent_session(
    llm: BaseLanguageModel,
    websocket: WebSocket,
    thread_id: str,
    user_id: str,
    access_token: str,
    refresh_token: str = None,
    request_id: str = None,
    initial_request: str = None
) -> None:
    """
    Handles the agent session lifecycle and message processing loop.
    
    Creates an agent, configures it with the given parameters, and processes
    incoming WebSocket messages in a loop until disconnection or error.
    
    Args:
        llm: The language model to use for the agent.
        websocket: The WebSocket connection for communication.
        thread_id: Unique identifier for the conversation thread.
        user_id: The authenticated user's ID.
        access_token: The access token for authentication.
        refresh_token: The refresh token for authentication.
    """
    try:
        async with create_agent(llm=llm, websocket=websocket, access_token=access_token) as agent:
            base_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "access_token": access_token,
                },
            }

            if os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_HOST"):
                langfuse_handler = CallbackHandler()
                base_config["callbacks"] = [langfuse_handler]

            needs_reauth = False

            while True:
                try:
                    if not initial_request:
                        request = await websocket.receive_text()
                        request_id = str(uuid.uuid4())
                    else:
                        request = initial_request
                        initial_request = None

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

                except HTTPStatusError as e:
                    logging.error(f"HTTP auth error: {e}")
                    if e.response.status_code == 401:
                        needs_reauth = True
                        break
                
                except Exception as e:
                    logging.error(f"An error occurred: {e}", exc_info=True)
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(f'<error>{{"message": "{str(e)}"}}</error>')
                    else:
                        break
                finally:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text("</message>")
    except* HTTPStatusError as eg:
        for e in eg.exceptions:
            logging.error(f"MCP auth failed: {e}")
            if hasattr(e, 'response') and e.response.status_code == 401:
                logging.info("ERROR! Received 401, retrying agent session...")
                needs_reauth = True
                
    finally:
        if needs_reauth:
            try:
                token = await oauth_client.refresh_token(token_endpoint=token_endpoint, refresh_token=refresh_token)
                logging.info("Successfully refreshed access token, resuming agent session...")
                access_token = token["access_token"]
                refresh_token = token.get("refresh_token", refresh_token)
            except Exception as e: #TODO
                logging.error(f"Failed to refresh token: {e}")
                access_token, refresh_token = await _perform_oauth_authentication(websocket)

            await _handle_agent_session(
                llm=llm,
                websocket=websocket,
                thread_id=thread_id,
                user_id=user_id,
                access_token=access_token,
                refresh_token=refresh_token,
                request_id=request_id,
                initial_request=request) 


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
            agent=json_request.get("agent", "")
        )
    except json.JSONDecodeError:
        return WebSocketRequest(prompt=request, user_input="", context={}, tags=[], agent="")

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
