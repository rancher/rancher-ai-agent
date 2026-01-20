import os
import uuid
import logging
import json

from ..dependencies import get_llm
from ..services.agent.factory import create_agent
from dataclasses import dataclass
from fastapi import APIRouter
from fastapi import  WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState
from langgraph.graph.state import CompiledStateGraph
from langfuse.langchain import CallbackHandler
from langchain_core.language_models.llms import BaseLanguageModel
from langgraph.types import Command

from ..services.auth import get_user_id
from ..dependencies import get_llm

router = APIRouter()

async def get_user_id_from_websocket(websocket: WebSocket) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token from the WebSocket cookies.
    """
    cookies = websocket.cookies
    rancher_url = os.environ.get("RANCHER_URL","https://"+websocket.url.hostname)
    token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))

    return await get_user_id(rancher_url, token)

@dataclass
class WebSocketRequest:
    """Represents a parsed WebSocket request from the client."""
    prompt: str
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

    async with create_agent(llm=llm, websocket=websocket) as agent:
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
                input_data = _build_input_data(agent, config, ws_request)

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
                    await websocket.send_text(f'<error>{{"message": "{str(e)}"}}</error>')
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
            await _store_mcp_response(agent, config, stream["data"])
            await websocket.send_text(stream["data"])
    
        if stream["event"] == "on_chain_stream":
            if interrupt_value := _extract_interrupt_value(stream):
                #await _store_interrupt(agent, config, interrupt_value)
                await websocket.send_text(interrupt_value)

async def _store_mcp_response(agent: CompiledStateGraph, config: dict, mcp_data: any) -> None:
    """
    Stores MCP response data in the agent's metadata.
    
    Retrieves the current agent state, appends the MCP response to the
    mcp_responses list, and updates the state.
    
    Args:
        agent: The compiled LangGraph agent.
        config: The run configuration.
        mcp_data: The MCP response data to store.
    """
    current_state = await agent.aget_state(config)
    metadata = current_state.values.get("agent_metadata", {})
    mcp_responses = metadata.get("mcp_responses", [])
    if mcp_responses is None:
        mcp_responses = []
    mcp_responses.append(mcp_data)
    metadata["mcp_responses"] = mcp_responses
    
    await agent.aupdate_state(
        config,
        {"agent_metadata": metadata}
    )

async def _store_interrupt(agent: CompiledStateGraph, config: dict, interrupt_value: str) -> None:
    """
    Stores an interrupt value in the agent's metadata.
    
    Retrieves the current agent state, sets the last_interrupt field,
    and updates the state.
    
    Args:
        agent: The compiled LangGraph agent.
        config: The run configuration.
        interrupt_value: The interrupt value to store.
    """
    current_state = await agent.aget_state(config)
    metadata = current_state.values.get("agent_metadata", {})
    metadata["last_interrupt"] = interrupt_value
    
    await agent.aupdate_state(
        config,
        {"agent_metadata": metadata},
        as_node="tools"
    )

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
        prompt = json_request.get("prompt", "")
        context = json_request.get("context", {})
        
        # Enrich prompt with context if present
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
            context=context,
            tags=json_request.get("tags", []),
            agent=json_request.get("agent", "")
        )
    except json.JSONDecodeError:
        return WebSocketRequest(prompt=request, context={}, tags=[], agent="")

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

    if ws_request.agent:
        config["configurable"]["agent"] = ws_request.agent
    else:
        config["configurable"]["agent"] = ""

    # Exclude "ephemeral" messages from being stored in memory
    tags = ws_request.tags or []
    if "ephemeral" in tags:
        config["configurable"]["thread_id"] = ""
    
    return config

def _build_input_data(agent: CompiledStateGraph, config: dict, ws_request: WebSocketRequest) -> dict | Command:
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
    state = agent.get_state(config=config)
    
    if state.interrupts:
        return Command(resume=ws_request.prompt)
    
    input_messages = [{"role": "user", "content": ws_request.prompt}]

    return {
        "messages": input_messages,
        "agent_metadata": {
            "prompt": ws_request.prompt,
            "context": ws_request.context,
            "tags": ws_request.tags,
            "mcp_responses": [],
        },
    }
