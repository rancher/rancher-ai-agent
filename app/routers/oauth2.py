import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from ..services.agent.loader import AgentConfig, AuthenticationType, load_agent_configs
from ..services.oauth2 import OAuthClient, discover_oauth_metadata, get_oauth_client_credentials, get_oauth_cookie_names, get_redirect_uri, oauth_store

logger = logging.getLogger(__name__)

router = APIRouter()

_COOKIE_MAX_AGE = 7 * 24 * 60 * 60  # one week in seconds

@router.get("/oauth/callback")
async def get(request: Request):
    """OAuth callback that returns HTML to communicate token back to parent window and stores tokens as httponly cookies."""
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    # Verify state to prevent CSRF attacks and retrieve stored data
    session_token = request.cookies.get("R_SESS", "")
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

    verifier = oauth_data["verifier"]
    oauth_client = oauth_data["oauth_client"]
    token_endpoint = oauth_data["token_endpoint"]
    cookie_key = oauth_data.get("cookie_key", "")

    # Exchange the code for the actual Access Token
    redirect_uri = get_redirect_uri(request.url.hostname)

    try:
        token = await oauth_client.fetch_token(
            token_endpoint=token_endpoint,
            authorization_response=str(request.url),
            redirect_uri=redirect_uri,
            verifier=verifier
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
        if cookie_key:
            cookie_names = get_oauth_cookie_names(cookie_key)
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
        return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head><title>Authentication Error</title></head>
            <body>
                <h2>Authentication Error</h2>
                <p>Failed to exchange token: {str(e)}</p>
                <p>Please close this window and try again.</p>
            </body>
            </html>
        """, status_code=500)


@router.post("/oauth/refresh")
async def refresh_token_endpoint(request: Request):
    """
    Refresh the OAuth2 access token using the refresh token stored in cookies.

    Expects a JSON body with 'agent' (the agent config name). The endpoint
    resolves credentials from the agent's Kubernetes secret and performs
    OAuth discovery to obtain the token endpoint.
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

    if not agent_cfg.authentication_secret:
        return JSONResponse({"error": f"Agent '{agent_name}' has no authentication secret"}, status_code=400)

    # Discover OAuth metadata and resolve the cookie key
    try:
        metadata = await discover_oauth_metadata(agent_cfg.mcp_url)
    except Exception as e:
        return JSONResponse({"error": f"OAuth discovery failed: {e}"}, status_code=502)

    cookie_names = get_oauth_cookie_names(agent_cfg.name)

    refresh_token_value = request.cookies.get(cookie_names["refresh_token"])
    if not refresh_token_value:
        return JSONResponse({"error": "No refresh token available"}, status_code=401)

    # Build the OAuth client from the agent's credentials
    credentials = get_oauth_client_credentials(agent_cfg.authentication_secret)
    oauth_client = OAuthClient(
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        scope=credentials.scopes,
    )

    try:
        token = await oauth_client.refresh_token(metadata.token_endpoint, refresh_token_value)
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


def _find_oauth_agent_config(agent_name: str) -> AgentConfig | None:
    """Find the AgentConfig for a given agent name if it uses OAuth2."""
    for cfg in load_agent_configs():
        if cfg.name == agent_name and cfg.authentication == AuthenticationType.OAUTH2:
            return cfg
    return None