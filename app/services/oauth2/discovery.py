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
    AuthorizationServerMetadata,
    OAuthDiscoveryError,
    OAuthDiscoveryResult,
    ResourceMetadata,
)
from .utils import _get_tls_verify

logger = logging.getLogger(__name__)


def _parse_www_authenticate(header: str) -> tuple[str | None, list[str] | None]:
    """
    Parse WWW-Authenticate header for resource_metadata and scope.
    Handles Bearer scheme as per RFC 9728 Section 5.1 and RFC 6750 Section 3.
    Example header:
        Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource",
               scope="files:read"
    Returns:
        Tuple of (resource_metadata_url, required_scopes), with None for missing values.
    """
    resource_metadata_url = None
    scopes = None

    rm_match = re.search(r'resource_metadata="([^"]+)"', header)
    if rm_match:
        resource_metadata_url = rm_match.group(1)

    scope_match = re.search(r'scope="([^"]+)"', header)
    if scope_match:
        scopes = scope_match.group(1).split()

    return resource_metadata_url, scopes


async def _discover_from_www_authenticate(
    client: httpx.AsyncClient, mcp_url: str
) -> tuple[str | None, list[str] | None]:
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
            return None, None

        # Use case-insensitive lookup to handle any header casing (e.g. Www-Authenticate)
        www_auth = next(
            (v for k, v in response.headers.items() if k.lower() == "www-authenticate"),
            "",
        )
        if not www_auth:
            logger.debug(f"No WWW-Authenticate header in 401 response from {mcp_url}")
            return None, None

        return _parse_www_authenticate(www_auth)
    except httpx.RequestError as e:
        logger.warning(f"Failed to connect to MCP server at {mcp_url} for OAuth discovery: {e}")
        return None, None


async def _discover_from_well_known(
    client: httpx.AsyncClient, mcp_url: str
) -> str | None:
    """
    Discover resource metadata from well-known URIs (RFC 9728).
    Tries the following URLs in order:
    1. Path-specific: https://example.com/.well-known/oauth-protected-resource/path/to/mcp
    2. Root: https://example.com/.well-known/oauth-protected-resource
    """
    parsed = urlparse(mcp_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")

    urls_to_try = []
    if path:
        urls_to_try.append(f"{base}/.well-known/oauth-protected-resource{path}")
    urls_to_try.append(f"{base}/.well-known/oauth-protected-resource")

    for url in urls_to_try:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                logger.info(f"Found resource metadata at {url}")
                return url
        except httpx.RequestError as e:
            logger.debug(f"Failed to fetch resource metadata from {url}: {e}")

    return None


async def _fetch_resource_metadata(
    client: httpx.AsyncClient, url: str
) -> ResourceMetadata:
    """Fetch and parse OAuth Protected Resource Metadata (RFC 9728)."""
    try:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

        return ResourceMetadata(
            resource=data.get("resource", ""),
            authorization_servers=data.get("authorization_servers", []),
            scopes_supported=data.get("scopes_supported", []),
            bearer_methods_supported=data.get("bearer_methods_supported", []),
        )
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        raise OAuthDiscoveryError(f"Failed to fetch resource metadata from {url}: {e}")


async def _discover_auth_server_metadata(
    client: httpx.AsyncClient, issuer_url: str
) -> AuthorizationServerMetadata:
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
    parsed = urlparse(issuer_url)
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
                data = response.json()
                logger.info(f"Found authorization server metadata at {url}")
                return AuthorizationServerMetadata(
                    issuer=data.get("issuer", issuer_url),
                    authorization_endpoint=data["authorization_endpoint"],
                    token_endpoint=data["token_endpoint"],
                    registration_endpoint=data.get("registration_endpoint"),
                    scopes_supported=data.get("scopes_supported", []),
                    response_types_supported=data.get("response_types_supported", []),
                    code_challenge_methods_supported=data.get("code_challenge_methods_supported", []),
                )
        except (httpx.RequestError, KeyError, ValueError) as e:
            logger.debug(f"Failed to fetch auth server metadata from {url}: {e}")

    raise OAuthDiscoveryError(
        f"Failed to discover authorization server metadata for issuer {issuer_url}. "
        f"Tried: {', '.join(urls_to_try)}"
    )


async def discover_oauth_metadata(mcp_url: str) -> OAuthDiscoveryResult:
    """
    Discover OAuth metadata from an MCP server following the MCP specification.
    Implements the full discovery chain:
    1. Request MCP server, parse 401 WWW-Authenticate header for resource_metadata URL
    2. Fall back to well-known resource metadata URIs (RFC 9728)
    3. Fetch resource metadata to find authorization server, then discover its metadata (RFC 8414)
    Fallback (for servers that skip RFC 9728 resource metadata):
    4. If resource metadata discovery fails entirely, attempt RFC 8414 authorization server
       metadata discovery directly on the MCP server's base domain.
    Args:
        mcp_url: The URL of the MCP server endpoint.
    Returns:
        OAuthDiscoveryResult with discovered authorization and token endpoints.
    Raises:
        OAuthDiscoveryError: If all discovery strategies fail.
    """
    async with httpx.AsyncClient(follow_redirects=True, verify=_get_tls_verify()) as client:
        # Step 1: Try WWW-Authenticate header discovery
        resource_metadata_url, required_scopes = await _discover_from_www_authenticate(client, mcp_url)

        # Step 2: Fall back to well-known resource metadata URIs (RFC 9728)
        if not resource_metadata_url:
            resource_metadata_url = await _discover_from_well_known(client, mcp_url)

        if resource_metadata_url:
            # Step 3: Fetch resource metadata and discover the authorization server
            resource_metadata = await _fetch_resource_metadata(client, resource_metadata_url)

            # TODO how to show supported scopes? use status?

            if not resource_metadata.authorization_servers:
                raise OAuthDiscoveryError(
                    f"Resource metadata at {resource_metadata_url} did not include any authorization servers."
                )

            issuer_url = resource_metadata.authorization_servers[0]
            auth_server_metadata = await _discover_auth_server_metadata(client, issuer_url)

            return OAuthDiscoveryResult(
                authorization_endpoint=auth_server_metadata.authorization_endpoint,
                token_endpoint=auth_server_metadata.token_endpoint,
                registration_endpoint=auth_server_metadata.registration_endpoint,
                scopes_supported=auth_server_metadata.scopes_supported,
                required_scopes=required_scopes or resource_metadata.scopes_supported,
                resource_metadata=resource_metadata,
                auth_server_metadata=auth_server_metadata,
            )

        # Step 4: Fallback — some MCP servers (e.g. Atlassian) don't implement RFC 9728
        # resource metadata but do serve RFC 8414 auth server metadata directly on the
        # MCP server's base domain.
        logger.info(
            f"No RFC 9728 resource metadata found for {mcp_url}. "
            "Falling back to direct RFC 8414 authorization server metadata discovery."
        )
        try:
            auth_server_metadata = await _discover_auth_server_metadata(client, mcp_url)
        except OAuthDiscoveryError:
            raise OAuthDiscoveryError(
                f"Failed to discover OAuth metadata for MCP server at {mcp_url}. "
                "Tried: RFC 9728 WWW-Authenticate header, RFC 9728 well-known resource metadata URIs, "
                "and RFC 8414 authorization server metadata on the MCP server base domain. "
                "Ensure the server exposes OAuth metadata or configure client credentials manually."
            )

        return OAuthDiscoveryResult(
            authorization_endpoint=auth_server_metadata.authorization_endpoint,
            token_endpoint=auth_server_metadata.token_endpoint,
            registration_endpoint=auth_server_metadata.registration_endpoint,
            scopes_supported=auth_server_metadata.scopes_supported,
            required_scopes=required_scopes or auth_server_metadata.scopes_supported,
            resource_metadata=None,
            auth_server_metadata=auth_server_metadata,
        )
