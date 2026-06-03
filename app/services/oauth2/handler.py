import json
import logging
import os

from fastapi import WebSocket

from ..agent._constants import NoAgentAvailableError
from ..agent.loader import AgentConfig
from .client import OAuthClient
from .cookies import get_oauth_cookie_names
from .credentials import AGENT_NAMESPACE, get_oauth_client_credentials
from .discovery import discover_oauth_metadata
from .store import oauth_store

async def handle_oauth_authentication(agent_cfg: AgentConfig, websocket: WebSocket) -> None:
    """
    Handle OAuth2 authentication for an agent that requires it.

    Attempts silent token refresh first; if that fails, initiates the full
    OAuth flow and waits for the client to complete the callback.

    Re-raises OAuthSecretError so the caller can handle the error response.

    Args:
        agent_cfg: The agent configuration requiring OAuth2 authentication.
        websocket: The WebSocket connection.
    """
    token_refreshed = await _initiate_oauth_flow(agent_cfg, websocket)
    if not token_refreshed:
        await websocket.receive_text()
        await _inject_oauth_cookie(agent_cfg, websocket)


async def _try_refresh_oauth_token(agent_cfg: AgentConfig, websocket: WebSocket) -> bool:
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
        cookie_names = get_oauth_cookie_names(agent_cfg.name)

        refresh_token = websocket.cookies.get(cookie_names["refresh_token"])
        if not refresh_token:
            return False

        # Send a custom message to the client so it can call the HTTP refresh
        # endpoint to persist the new tokens as browser cookies.
        refresh_data = json.dumps({"agent": agent_cfg.name})
        await websocket.send_text(f'<token-refreshed>{refresh_data}</token-refreshed>')
        # Wait for the refresh token response
        #TODO check response!
        response = await websocket.receive_text()

        if response == "ok":
            # The client has refreshed the token and set it as a cookie, so we can now inject it into the WebSocket's cookie dict for the agent to use.
            await _inject_oauth_cookie(agent_cfg, websocket)


        logging.debug(f"Successfully refreshed OAuth token for agent '{agent_cfg.name}'")
        return True

    except Exception as e:
        logging.debug(f"Token refresh failed for agent '{agent_cfg.name}': {e}")
        return False


async def _initiate_oauth_flow(agent_cfg: AgentConfig, websocket: WebSocket) -> bool:
    """
    Initiate the OAuth2 authentication flow for an agent that requires it.

    First attempts to refresh the access token using a stored refresh token.
    If no refresh token is available or the refresh fails, performs the full
    OAuth discovery on the agent's MCP URL, generates the authorization URL,
    stores the state for the callback, and sends the auth URL to the client.

    Args:
        agent_cfg: The agent configuration requiring OAuth2 authentication.
        websocket: The WebSocket connection to send the auth URL to.

    Returns:
        True if the token was silently refreshed (no user interaction needed),
        False if the full OAuth flow was initiated and user action is required.
    """
    # Try to refresh the token before initiating the full OAuth flow
    if await _try_refresh_oauth_token(agent_cfg, websocket):
        return True

    metadata = await discover_oauth_metadata(agent_cfg.mcp_url)

    oauth_client = None
    credentials = None

    # Fetch credentials if a secret exists
    if agent_cfg.authentication_secret:
        credentials = get_oauth_client_credentials(agent_cfg.authentication_secret)

    #  Try static client credentials first
    if credentials and credentials.client_id and credentials.client_secret:
        oauth_client = OAuthClient(
            client_id=credentials.client_id,
            client_secret=credentials.client_secret,
            scope=credentials.scopes,
        )

    # Fallback to dynamic client registration
    elif metadata.registration_endpoint:
        kwargs = {
            "registration_endpoint": metadata.registration_endpoint,
            "redirect_uri": get_redirect_uri(websocket.url.hostname),
        }

        # Only inject scope if credentials were successfully fetched
        if credentials:
            kwargs["scope"] = credentials.scopes

        oauth_client = await OAuthClient.from_dynamic_registration(**kwargs)

    if not oauth_client:
        raise NoAgentAvailableError(
            f"Agent '{agent_cfg.name}' requires OAuth2 but has no authentication secret "
            "and the server does not support dynamic client registration."
        )

    auth_endpoint = metadata.authorization_endpoint
    redirect_uri = get_redirect_uri(websocket.url.hostname)

    url, verifier, state = await oauth_client.get_auth_url(auth_endpoint, redirect_uri)

    cookie_key = agent_cfg.name
    session_token = websocket.cookies.get("R_SESS", "")
    oauth_store.set_state(state, {
        "verifier": verifier,
        "oauth_client": oauth_client,
        "token_endpoint": metadata.token_endpoint,
        "cookie_key": cookie_key,
    }, session_token)

    await websocket.send_text(
        f'<authentication>{{"type": "oauth2", "url": "{str(url)}", "agent": "{agent_cfg.name}"}}</authentication>'
    )
    return False


async def _inject_oauth_cookie(agent_cfg: AgentConfig, websocket: WebSocket) -> None:
    """
    Inject the OAuth access token into the WebSocket's cookies dict.

    WebSocket cookies are frozen at handshake time so they never reflect cookies
    set later by the HTTP OAuth callback. This reads the token from the shared
    oauth_store (populated by the callback) and injects it so that
    create_mcp_client can find it via websocket.cookies.get(...).
    """
    cookie_name = get_oauth_cookie_names(agent_cfg.name)["access_token"]
    session_token = websocket.cookies.get("R_SESS", "")
    token = oauth_store.pop_token(cookie_name, session_token)
    if token:
        websocket.cookies[cookie_name] = token
        logging.debug(f"Injected OAuth token into websocket cookies for agent '{agent_cfg.name}'")
    else:
        logging.warning(f"No OAuth token found in store for agent '{agent_cfg.name}'")


def get_redirect_uri(url: str | None = None) -> str:
    """Determine the OAuth redirect URI."""
    configured = os.environ.get("OAUTH_REDIRECT_URI")
    if configured:
        return configured

    return f"https://{url}/api/v1/namespaces/{AGENT_NAMESPACE}/services/http:rancher-ai-agent:80/proxy/oauth/callback"
