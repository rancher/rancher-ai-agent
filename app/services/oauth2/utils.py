"""OAuth shared utilities."""

import os


AGENT_NAMESPACE = "cattle-ai-agent-system"


def _get_tls_verify() -> bool:
    """Get TLS verification setting from environment."""
    return os.environ.get('INSECURE_SKIP_TLS', 'false').lower() != 'true'


def get_redirect_uri(url: str | None) -> str:
    """
    Determine the OAuth redirect URI.
    """
    configured = os.environ.get("OAUTH_REDIRECT_URI")
    if configured:
        return configured

    return f"https://{url}/api/v1/namespaces/{AGENT_NAMESPACE}/services/http:rancher-ai-agent:80/proxy/oauth/callback"
