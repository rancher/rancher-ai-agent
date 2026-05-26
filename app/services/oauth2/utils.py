"""OAuth shared utilities."""

import os


AGENT_NAMESPACE = "cattle-ai-agent-system"


def _get_tls_verify() -> bool:
    """Get TLS verification setting from environment."""
    return os.environ.get('INSECURE_SKIP_TLS', 'false').lower() != 'true'


def get_redirect_uri(websocket=None) -> str:
    """
    Determine the OAuth redirect URI.
    Priority:
    1. OAUTH_REDIRECT_URI environment variable
    2. Derived from WebSocket connection URL
    3. Default localhost fallback
    """
    configured = os.environ.get("OAUTH_REDIRECT_URI")
    if configured:
        return configured

    if websocket:
        scheme = "https" if websocket.url.scheme == "wss" else "http"
        host = websocket.url.hostname
        port = websocket.url.port
        port_str = f":{port}" if port and port not in (80, 443) else ""
        return f"{scheme}://{host}{port_str}/oauth/callback"

    # TODO figure out redirect, return error here!
    return "http://localhost:8000/oauth/callback"
