"""
MCP OAuth 2.0 Discovery implementation.
Implements the MCP specification for OAuth discovery:
- Resource metadata discovery via WWW-Authenticate headers (RFC 9728 Section 5.1)
- Resource metadata discovery via well-known URIs (RFC 9728)
- Authorization server metadata discovery (RFC 8414)
"""

import re
import logging

import httpx

from urllib.parse import urlparse

from .models import (
    OAuthDiscoveryError,
)
from .client import _get_tls_verify
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredMetadata:
    """Data class to hold discovered OAuth metadata."""
    metadataEndpoint: str
    scopesSupported: list = field(default_factory=list)


async def discover_metadata_endpoint(mcp_url: str) -> DiscoveredMetadata:
    """
    Discover OAuth metadata endpoint from an MCP server
    Args:
        mcp_url: The URL of the MCP server endpoint.
    Returns:
        The discovered OAuth metadata endpoint URL.
    Raises:
        OAuthDiscoveryError: If all discovery strategies fail.
    """
    async with httpx.AsyncClient(follow_redirects=True, verify=_get_tls_verify()) as client:
        resource_metadata_url = await _discover_from_www_authenticate(client, mcp_url)
        if resource_metadata_url:
            response = await client.get(resource_metadata_url)
            response.raise_for_status()
            protected_resource_metadata = response.json()
            authorization_servers = protected_resource_metadata.get("authorization_servers") or []
            issuer_url = authorization_servers[0] if authorization_servers else None
            if not issuer_url:
                raise OAuthDiscoveryError(
                    f"Resource metadata at {resource_metadata_url} did not include any authorization servers."
                )
            auth_server_metadata_endpoint = await _discover_auth_server_metadata_endpoint(client, issuer_url)

            return DiscoveredMetadata(
                metadataEndpoint=auth_server_metadata_endpoint,
                scopesSupported=protected_resource_metadata.get("scopes_supported", []),
            )

        else:
            raise OAuthDiscoveryError(f"Failed to discover OAuth metadata for MCP server at {mcp_url}. ")

  
def _parse_www_authenticate(header: str) -> str | None:
    """
    Parse WWW-Authenticate header for resource_metadata.
    Handles Bearer scheme as per RFC 9728 Section 5.1 and RFC 6750 Section 3.
    Example header:
        Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"
    Returns:
        resource_metadata_url
    """
    resource_metadata_url = None

    rm_match = re.search(r'resource_metadata="([^"]+)"', header)
    if rm_match:
        resource_metadata_url = rm_match.group(1)


    return resource_metadata_url


async def _discover_from_www_authenticate(
    client: httpx.AsyncClient, mcp_url: str
) -> str | None:
    """
    Discover resource metadata URL from a 401 WWW-Authenticate header.
    Makes a request to the MCP server and parses the WWW-Authenticate header
    for the resource_metadata URL and scope parameters.
    Returns:
        Tuple of (resource_metadata_url, required_scopes) or (None, None).
    """
    try:
        response = await client.get(mcp_url)

        if response.status_code != 401:
            logger.debug(f"MCP server at {mcp_url} returned {response.status_code}, expected 401")
            return None

        # Use case-insensitive lookup to handle any header casing (e.g. Www-Authenticate)
        www_auth = next(
            (v for k, v in response.headers.items() if k.lower() == "www-authenticate"),
            "",
        )
        if not www_auth:
            logger.debug(f"No WWW-Authenticate header in 401 response from {mcp_url}")
            return None

        return _parse_www_authenticate(www_auth)
    except httpx.RequestError as e:
        logger.warning(f"Failed to connect to MCP server at {mcp_url} for OAuth discovery: {e}")
        return None


async def _discover_auth_server_metadata_endpoint(
    client: httpx.AsyncClient, protected_resource_endpoint: str
) -> str:
    """
    Discover authorization server metadata following RFC 8414.
    For issuer URLs with path components (e.g., https://auth.example.com/tenant1):
    1. https://auth.example.com/.well-known/oauth-authorization-server/tenant1
    2. https://auth.example.com/.well-known/openid-configuration/tenant1
    3. https://auth.example.com/tenant1/.well-known/openid-configuration
    For issuer URLs without path components (e.g., https://auth.example.com):
    1. https://auth.example.com/.well-known/oauth-authorization-server
    2. https://auth.example.com/.well-known/openid-configuration
    """
    parsed = urlparse(protected_resource_endpoint)
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")

    urls_to_try = []

    if path:
        urls_to_try.append(f"{base}/.well-known/oauth-authorization-server{path}")
        urls_to_try.append(f"{base}/.well-known/openid-configuration{path}")
        urls_to_try.append(f"{base}{path}/.well-known/openid-configuration")
        # Also try root-level as final fallback (some servers, e.g. Atlassian MCP,
        # host auth server metadata at root even when the MCP endpoint has a path)
        urls_to_try.append(f"{base}/.well-known/oauth-authorization-server")
        urls_to_try.append(f"{base}/.well-known/openid-configuration")
    else:
        urls_to_try.append(f"{base}/.well-known/oauth-authorization-server")
        urls_to_try.append(f"{base}/.well-known/openid-configuration")

    for url in urls_to_try:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return url
        except (httpx.RequestError, KeyError, ValueError) as e:
            logger.debug(f"Failed to fetch auth server metadata from {url}: {e}")

    raise OAuthDiscoveryError(
        f"Failed to discover authorization server metadata for issuer {protected_resource_endpoint}. "
        f"Tried: {', '.join(urls_to_try)}"
    )

