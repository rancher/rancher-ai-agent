import logging
import httpx
import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import HTTPException, status
from pydantic import BaseModel, HttpUrl
from urllib.parse import urlparse
from ..services.auth import get_user_id_from_request
from ..services.oauth2.client import OAuthClientManager, _get_tls_verify
from ..services.oauth2.discovery import discover_metadata_endpoint
from ..services.agent.loader import AgentConfig, AuthenticationType, load_agent_configs
from ..services.oauth2 import (
    OAuthDiscoveryError,
    get_oauth_cookie_names,
    get_redirect_uri,
    oauth_store,
)

_COOKIE_MAX_AGE = 7 * 24 * 60 * 60  # one week in seconds

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/api/oauth2", tags=["oauth"])

@router.get("/callback")
async def get(request: Request):
    """OAuth callback that returns HTML to communicate token back to parent window and stores tokens as httponly cookies."""
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    # Verify state to prevent CSRF attacks and retrieve stored data
    session_token = os.environ.get("RANCHER_API_TOKEN", request.cookies.get("R_SESS", ""))
    oauth_data = oauth_store.pop_state(state, session_token)
    if not oauth_data:
        return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head><title>Authentication Error</title></head>
            <body>
                <h2>Authentication Error</h2>
                <p>Invalid state or session expired. Please close this window and try again.</p>
            </body>
            </html>
        """, status_code=400)

    code_verifier = oauth_data["code_verifier"]
    agent_name = oauth_data["agent_name"]
    redirect_uri = oauth_data["redirect_uri"]

    # Get the registered OAuth client for this agent
    manager = OAuthClientManager.get_instance()
    client = manager.get_client(agent_name)
    if not client:
        return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head><title>Authentication Error</title></head>
            <body>
                <h2>Authentication Error</h2>
                <p>OAuth client not found for this agent. Please close this window and try again.</p>
            </body>
            </html>
        """, status_code=400)

    try:
        token = await client.fetch_access_token(
            redirect_uri=redirect_uri,
            code=code,
            code_verifier=code_verifier,
        )

        access_token = token.get("access_token", "")
        refresh_token = token.get("refresh_token", "")

        # Return HTML that tells UI the token was received and it can close the popup.
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Authentication Successful</title></head>
        <body>
            <h2>Authentication Successful!</h2>
            <p>Closing window...</p>
            <script>
                // Send success signal to the parent window via BroadcastChannel
                const channel = new BroadcastChannel('oauth_channel');
                channel.postMessage({{ type: 'oauth_success' }});
                channel.close();
                window.close();
            </script>
        </body>
        </html>
        """

        response = HTMLResponse(content=html_content)

        # Store tokens as httponly cookies for subsequent connections
        cookie_names = get_oauth_cookie_names(agent_name)
        is_secure = request.url.scheme == "https"

        if access_token:
            response.set_cookie(
                cookie_names["access_token"], access_token,
                httponly=True, secure=is_secure, samesite="strict", path="/",
                max_age=_COOKIE_MAX_AGE,
            )
            # Store the access token so the WebSocket handler can inject it
            # into the connection's cookies (WebSocket cookies are frozen at
            # handshake time and won't reflect new HTTP cookies).
            oauth_store.set_token(cookie_names["access_token"], access_token, session_token)
        else:
            logger.warning(f"No access token received")

        if refresh_token:
            response.set_cookie(
                cookie_names["refresh_token"], refresh_token,
                httponly=True, secure=is_secure, samesite="strict", path="/",
                max_age=_COOKIE_MAX_AGE,
            )
            oauth_store.set_token(cookie_names["refresh_token"], refresh_token, session_token)

        return response

    except Exception as e:
        logger.error(f"Token exchange failed for agent '{agent_name}': {e}")
        return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head><title>Authentication Error</title></head>
            <body>
                <h2>Authentication Error</h2>
                <p>Authentication failed. Please close this window and try again.</p>
            </body>
            </html>
        """, status_code=500)


@router.post("/refresh")
async def refresh_token_endpoint(request: Request):
    """
    Refresh the OAuth2 access token using the refresh token stored in cookies.

    Expects a JSON body with 'agent' (the agent config name). The endpoint
    uses the singleton OAuth client (registered at AIAgentConfig create/update)
    to perform the refresh.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid request body"}, status_code=400)

    agent_name = body.get("agent", "")
    if not agent_name:
        return JSONResponse({"error": "Missing agent name"}, status_code=400)

    # Find the agent config
    agent_cfg = _find_oauth_agent_config(agent_name)
    if not agent_cfg:
        return JSONResponse({"error": f"Agent '{agent_name}' not found or not OAuth2"}, status_code=404)

    manager = OAuthClientManager.get_instance()
    client = manager.get_client(agent_name)
    if not client:
        return JSONResponse({"error": f"No OAuth client registered for agent '{agent_name}'"}, status_code=400)

    cookie_names = get_oauth_cookie_names(agent_cfg.name)

    refresh_token_value = request.cookies.get(cookie_names["refresh_token"])
    if not refresh_token_value:
        return JSONResponse({"error": "No refresh token available"}, status_code=401)

    try:
        token = await client.fetch_access_token(
            grant_type="refresh_token",
            refresh_token=refresh_token_value,
        )
    except Exception as e:
        logger.debug(f"Token refresh failed for agent '{agent_name}': {e}")
        return JSONResponse({"error": f"Token refresh failed: {e}"}, status_code=401)

    access_token = token.get("access_token", "")
    new_refresh_token = token.get("refresh_token", "")

    if not access_token:
        return JSONResponse({"error": "No access token in response"}, status_code=502)

    response = JSONResponse({"status": "ok"})
    is_secure = request.url.scheme == "https"

    response.set_cookie(
        cookie_names["access_token"], access_token,
        httponly=True, secure=is_secure, samesite="strict", path="/",
        max_age=_COOKIE_MAX_AGE,
    )
    # Also store in the token store so the WebSocket can pick it up
    session_token = request.cookies.get("R_SESS", "")
    oauth_store.set_token(cookie_names["access_token"], access_token, session_token)

    if new_refresh_token:
        response.set_cookie(
            cookie_names["refresh_token"], new_refresh_token,
            httponly=True, secure=is_secure, samesite="strict", path="/",
            max_age=_COOKIE_MAX_AGE,
        )
        oauth_store.set_token(cookie_names["refresh_token"], new_refresh_token, session_token)


    return response


@router.get("/metadata")
async def get_metadata(mcpUrl: str, request: Request):
    """
    Discover OAuth metadata for an MCP server URL.

    Returns discovered metadata (or `null` if discovery fails).
    """
    user_id = await get_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    if not mcpUrl:
        return JSONResponse({"error": "Missing mcpUrl"}, status_code=400)

    try:
        return await discover_metadata_endpoint(mcpUrl)
    except OAuthDiscoveryError as e:
        logger.info(f"OAuth metadata discovery failed for {mcpUrl}: {e}")
        return JSONResponse(content=None, status_code=200)


class RegistrationPayload(BaseModel):
    metadataEndpoint: HttpUrl
    
@router.post("/dynamic-registration")
async def dynamic_registration(payload: RegistrationPayload, request: Request):
    """
    Perform OAuth2 dynamic client registration (RFC 7591).
    """
    # Without authentication this endpoint would let any network-reachable caller
    # use the agent pod as an SSRF proxy to probe internal services.
    user_id = await get_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    metadata_endpoint = payload.metadataEndpoint
    if not metadata_endpoint:
        return JSONResponse({"error": "Missing metadataEndpoint"}, status_code=400)

    metadata_origin = f"{urlparse(str(metadata_endpoint)).scheme}://{urlparse(str(metadata_endpoint)).netloc}"

    async with httpx.AsyncClient(follow_redirects=True, verify=_get_tls_verify()) as http_client:
        metadata = await http_client.get(str(metadata_endpoint))
        metadata.raise_for_status()
        data = metadata.json()
        registration_endpoint = data.get("registration_endpoint")
        if not registration_endpoint:
            return JSONResponse({"error": "Dynamic Client Registration not supported"}, status_code=400)

        # Enforce same-origin on the registration_endpoint returned by the metadata server.
        # Without this check an attacker-controlled metadata server could redirect the
        # second POST to an arbitrary internal URL (two-hop SSRF).
        reg_origin = f"{urlparse(registration_endpoint).scheme}://{urlparse(registration_endpoint).netloc}"
        if reg_origin != metadata_origin:
            logger.warning(
                f"Dynamic registration endpoint origin does not match metadata endpoint"
            )
            return JSONResponse({"error": "Registration endpoint must share the same origin as the metadata endpoint"}, status_code=400)

        registration_data = {
            "client_name": "Rancher AI Agent",
            "redirect_uris": [get_redirect_uri(request.url.hostname)],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"]
        }

        response = await http_client.post(registration_endpoint, json=registration_data)
        response.raise_for_status()
        reg_data = response.json()
        client_id = reg_data.get("client_id")
        client_secret = reg_data.get("client_secret", "")
        if not client_id:
            return JSONResponse({"error": "Registration response did not include client_id"}, status_code=502)
        if not client_secret:
            return JSONResponse({"error": "Registration response did not include client_secret"}, status_code=502)

        return JSONResponse({
            "clientId": client_id,
            "clientSecret": client_secret,
        })


def _find_oauth_agent_config(agent_name: str) -> AgentConfig | None:
    """Find the AgentConfig for a given agent name if it uses OAuth2."""
    for cfg in load_agent_configs():
        if cfg.name == agent_name and cfg.authentication == AuthenticationType.OAUTH2:
            return cfg
    return None