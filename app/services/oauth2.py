"""
MCP OAuth 2.0 Discovery and Client implementation.

Implements the MCP specification for OAuth discovery:
- Resource metadata discovery via WWW-Authenticate headers (RFC 9728 Section 5.1)
- Resource metadata discovery via well-known URIs (RFC 9728)
- Authorization server metadata discovery (RFC 8414)
- Dynamic client registration (RFC 7591)
- OAuth 2.0 Authorization Code flow with PKCE
"""

import hashlib
import base64
import os
import secrets
import logging
import re
from urllib.parse import urlparse
from dataclasses import dataclass, field

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client

logger = logging.getLogger(__name__)


class OAuthDiscoveryError(Exception):
    """Raised when OAuth discovery fails."""
    pass


@dataclass
class ResourceMetadata:
    """OAuth Protected Resource Metadata (RFC 9728)."""
    resource: str = ""
    authorization_servers: list[str] = field(default_factory=list)
    scopes_supported: list[str] = field(default_factory=list)
    bearer_methods_supported: list[str] = field(default_factory=list)


@dataclass
class AuthorizationServerMetadata:
    """OAuth Authorization Server Metadata (RFC 8414)."""
    issuer: str = ""
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    registration_endpoint: str | None = None
    scopes_supported: list[str] = field(default_factory=list)
    response_types_supported: list[str] = field(default_factory=list)
    code_challenge_methods_supported: list[str] = field(default_factory=list)


@dataclass
class OAuthDiscoveryResult:
    """Complete result of the MCP OAuth discovery process."""
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None = None
    scopes_supported: list[str] = field(default_factory=list)
    required_scopes: list[str] = field(default_factory=list)
    resource_metadata: ResourceMetadata | None = None
    auth_server_metadata: AuthorizationServerMetadata | None = None


@dataclass
class OAuthClientCredentials:
    """OAuth2 client credentials retrieved from a Kubernetes secret."""
    client_id: str
    client_secret: str = ""
    scopes: str = ""


OAUTH_COOKIE_PREFIX = "mcp_oauth"


def generate_oauth_cookie_key(authorization_endpoint: str) -> str:
    """Generate a short key for OAuth cookie names based on the authorization endpoint."""
    return hashlib.sha256(authorization_endpoint.encode()).hexdigest()[:8]


def get_oauth_cookie_names(cookie_key: str) -> dict[str, str]:
    """
    Get the cookie names for a given OAuth cookie key.

    Returns a dict with keys: access_token, refresh_token.
    """
    return {
        "access_token": f"{OAUTH_COOKIE_PREFIX}_at_{cookie_key}",
        "refresh_token": f"{OAUTH_COOKIE_PREFIX}_rt_{cookie_key}",
    }


def _get_tls_verify() -> bool:
    """Get TLS verification setting from environment."""
    return os.environ.get('INSECURE_SKIP_TLS', 'false').lower() != 'true'


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


class OAuthClient:
    """OAuth2 client with PKCE support for MCP server authentication."""

    def __init__(self, client_id: str, client_secret: str = "", scope: str | None = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.client = AsyncOAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
        )

    @classmethod
    async def from_dynamic_registration(
        cls,
        registration_endpoint: str,
        redirect_uri: str,
        client_name: str = "Rancher AI Agent",
        scope: str | None = None,
    ) -> "OAuthClient":
        """
        Create an OAuthClient via Dynamic Client Registration (RFC 7591).

        Args:
            registration_endpoint: The authorization server's registration endpoint.
            redirect_uri: The redirect URI for the client.
            client_name: Human-readable name for the client.
            scope: Space-separated scopes to request.

        Returns:
            A configured OAuthClient with registered credentials.

        Raises:
            OAuthDiscoveryError: If registration fails.
        """
        async with httpx.AsyncClient(follow_redirects=True, verify=_get_tls_verify()) as http_client:
            registration_data = {
                "client_name": client_name,
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "client_secret_basic",
            }
            if scope:
                registration_data["scope"] = "read:jira-user"

            try:
                response = await http_client.post(registration_endpoint, json=registration_data)
                response.raise_for_status()
                data = response.json()

                logger.info(f"Successfully registered OAuth client '{client_name}' at {registration_endpoint}")
                return cls(
                    client_id=data["client_id"],
                    client_secret=data.get("client_secret", ""),
                    scope=scope,
                )
            except (httpx.RequestError, httpx.HTTPStatusError, KeyError) as e:
                raise OAuthDiscoveryError(
                    f"Dynamic client registration failed at {registration_endpoint}: {e}"
                )

    def generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code_verifier and code_challenge pair."""
        verifier = secrets.token_urlsafe(64)
        sha256_hash = hashlib.sha256(verifier.encode('utf-8')).digest()
        challenge = base64.urlsafe_b64encode(sha256_hash).decode('utf-8').replace('=', '')
        return verifier, challenge

    async def get_auth_url(self, auth_endpoint: str, redirect_uri: str) -> tuple[str, str, str]:
        """
        Create the authorization URL with PKCE.

        Args:
            auth_endpoint: The authorization endpoint URL.
            redirect_uri: The redirect URI for the callback.

        Returns:
            Tuple of (authorization_url, code_verifier, state).
        """
        verifier, challenge = self.generate_pkce_pair()
        state = secrets.token_urlsafe(16)

        url, _ = self.client.create_authorization_url(
            auth_endpoint,
            redirect_uri=redirect_uri,
            code_challenge=challenge,
            code_challenge_method='S256',
            state=state,
        )
        return url, verifier, state

    async def fetch_token(self, token_endpoint: str, authorization_response: str, redirect_uri: str, verifier: str) -> dict:
        """Exchange the authorization code for an access token using PKCE."""
        token = await self.client.fetch_token(
            token_endpoint,
            authorization_response=authorization_response,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
        )
        return token

    async def refresh_token(self, token_endpoint: str, refresh_token: str) -> dict:
        """Refresh the access token using a refresh token."""
        token = await self.client.refresh_token(
            token_endpoint,
            refresh_token=refresh_token,
        )
        return token


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

    return "http://localhost:8000/oauth/callback"


def get_oauth_client_credentials(secret_name: str) -> OAuthClientCredentials:
    """
    Retrieve OAuth2 client credentials from a Kubernetes secret.

    The secret must contain a 'clientId' key (or legacy 'client_id') and
    optionally a 'clientSecret' key (or legacy 'client_secret') and
    a 'scopes' key containing a space-separated string of OAuth scopes.

    Key names follow the AIAgentConfig CRD oauthSecret convention:
        clientId, clientSecret, scopes

    Args:
        secret_name: Name of the secret in the agent namespace.

    Returns:
        OAuthClientCredentials with client_id, client_secret, and scopes.
    """
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    v1 = client.CoreV1Api()
    namespace = os.environ.get("AGENT_NAMESPACE", "cattle-ai-agent-system")
    secret = v1.read_namespaced_secret(secret_name, namespace)

    if not secret.data:
        raise RuntimeError(
            f"OAuth secret '{secret_name}' in namespace '{namespace}' is empty."
        )

    # Support both camelCase (CRD convention) and snake_case (legacy)
    raw = secret.data
    id_key = next((k for k in ('clientId', 'client_id') if k in raw), None)
    if not id_key:
        raise RuntimeError(
            f"OAuth secret '{secret_name}' in namespace '{namespace}' "
            "must contain a 'clientId' key."
        )

    client_id = base64.b64decode(raw[id_key]).decode('utf-8')
    client_secret = ""
    secret_key = next((k for k in ('clientSecret', 'client_secret') if k in raw), None)
    if secret_key:
        client_secret = base64.b64decode(raw[secret_key]).decode('utf-8')

    scopes = ""
    if 'scopes' in raw:
        scopes = base64.b64decode(raw['scopes']).decode('utf-8').strip()

    return OAuthClientCredentials(client_id=client_id, client_secret=client_secret, scopes=scopes)