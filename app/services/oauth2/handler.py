import json
import logging
import os

from fastapi import WebSocket

from ..agent._constants import NoAgentAvailableError
from ..agent.loader import AgentConfig
from .client import OAuthClientManager
from .cookies import get_oauth_cookie_names
from .credentials import AGENT_NAMESPACE
from .store import oauth_store

async def handle_oauth_authentication(agent_name: str, websocket: WebSocket, add_message_tag: bool) -> None:
    """
    Handle OAuth2 authentication for an agent that requires it.

    Attempts silent token refresh first; if that fails, initiates the full
    OAuth flow and waits for the client to complete the callback.

    Re-raises OAuthSecretError so the caller can handle the error response.

    Args:
        agent_name: The name of the agent requiring OAuth2 authentication.
        websocket: The WebSocket connection.
        add_message_tag: Whether to add a message tag.
    """
    token_refreshed = await _initiate_oauth_flow(agent_name, websocket, add_message_tag)
    if not token_refreshed:
        response = await websocket.receive_text()
        if response != "authentication_confirmed":
            raise Exception(f"OAuth2 authentication failed")
        
        await _inject_oauth_cookie(agent_name, websocket)


async def _try_refresh_oauth_token(agent_name: str, websocket: WebSocket) -> bool:
    """
    Attempt to refresh the OAuth2 access token using a stored refresh token.

    Checks websocket cookies for an existing refresh token and, if found,
    uses it to obtain a new access token without requiring user interaction.

    Args:
        agent_cfg: The agent configuration requiring OAuth2 authentication.
        websocket: The WebSocket connection whose cookies may contain a refresh token.

    Returns:
        True if the token was successfully refreshed and injected, False otherwise.
    """
    try:
        cookie_names = get_oauth_cookie_names(agent_name)

        refresh_token = websocket.cookies.get(cookie_names["refresh_token"])
        if not refresh_token:
            return False

        # Send a custom message to the client so it can call the HTTP refresh
        # endpoint to persist the new tokens as browser cookies.
        refresh_data = json.dumps({"agent": agent_name})
        await websocket.send_text(f'<token-refresh>{refresh_data}</token-refresh>')
        # Wait for the refresh token response
        response = await websocket.receive_text()

        if response == "token_refresh_confirmed":
            # The client has refreshed the token and set it as a cookie, so we can now inject it into the WebSocket's cookie dict for the agent to use.
            await _inject_oauth_cookie(agent_name, websocket)
        else:
            return False

        logging.debug(f"Successfully refreshed OAuth token for agent '{agent_name}'")
        return True

    except Exception as e:
        logging.debug(f"Token refresh failed for agent '{agent_name}': {e}")
        return False


async def _initiate_oauth_flow(agent_name: str, websocket: WebSocket, add_message_tag: bool) -> bool:
    """
    Initiate the OAuth2 authentication flow for an agent that requires it.

    First attempts to refresh the access token using a stored refresh token.
    If no refresh token is available or the refresh fails, reads pre-provisioned
    credentials and metadata from the agent's authentication secret, generates
    the authorization URL, stores the state for the callback, and sends the
    auth URL to the client.

    Args:
        agent_name: The name of the agent requiring OAuth2 authentication.
        websocket: The WebSocket connection to send the auth URL to.
        add_message_tag: Whether to add a message tag.

    Returns:
        True if the token was silently refreshed (no user interaction needed),
        False if the full OAuth flow was initiated and user action is required.
    """
    if await _try_refresh_oauth_token(agent_name, websocket):
        return True

    manager = OAuthClientManager.get_instance()
    client = manager.get_client(agent_name)

    if not client:
        raise NoAgentAvailableError(
            f"Agent '{agent_name}' requires OAuth2 but no OAuth client is registered. "
            "Ensure the AIAgentConfig has a valid authenticationSecret with clientID, "
            "clientSecret, and metadata_endpoint."
        )

    redirect_uri = get_redirect_uri(websocket.url.hostname)
    rv = await client.create_authorization_url(redirect_uri)

    state = rv["state"]
    session_token = os.environ.get("RANCHER_API_TOKEN", websocket.cookies.get("R_SESS", ""))
    oauth_store.set_state(state, {
        "code_verifier": rv.get("code_verifier"),
        "agent_name": agent_name,
        "redirect_uri": redirect_uri,
    }, session_token)

    authentication_message = f'<authentication>{json.dumps({"type": "oauth2", "url": rv["url"], "agent": agent_name})}</authentication>'

    if add_message_tag:
        authentication_message = f'<message>{authentication_message}</message>'

    await websocket.send_text(authentication_message)

    return False


async def _inject_oauth_cookie(agent_name: str, websocket: WebSocket) -> None:
    """
    Inject the OAuth access token into the WebSocket's cookies dict.

    WebSocket cookies are frozen at handshake time so they never reflect cookies
    set later by the HTTP OAuth callback. This reads the token from the shared
    oauth_store (populated by the callback) and injects it so that
    create_mcp_client can find it via websocket.cookies.get(...).
    """
    cookie_name = get_oauth_cookie_names(agent_name)["access_token"]
    session_token = os.environ.get("RANCHER_API_TOKEN", websocket.cookies.get("R_SESS", ""))
    token = oauth_store.pop_token(cookie_name, session_token)
    if token:
        websocket.cookies[cookie_name] = token
        logging.debug(f"Injected OAuth token into websocket cookies for agent '{agent_name}'")
    else:
        logging.warning(f"No OAuth token found in store for agent '{agent_name}'")


def get_redirect_uri(url: str | None = None) -> str:
    """Determine the OAuth redirect URI."""
    configured = os.environ.get("OAUTH_REDIRECT_URI")
    if configured:
        return configured

    return f"https://{url}/api/v1/namespaces/{AGENT_NAMESPACE}/services/http:rancher-ai-agent:80/proxy/oauth/callback"
