"""OAuth 2.0 data models and exceptions."""

from dataclasses import dataclass, field


class OAuthDiscoveryError(Exception):
    """Raised when OAuth discovery fails."""
    pass


@dataclass
class ResourceMetadata:
    """OAuth Protected Resource Metadata."""
    resource: str = ""
    authorization_servers: list[str] = field(default_factory=list)
    scopes_supported: list[str] = field(default_factory=list)
    bearer_methods_supported: list[str] = field(default_factory=list)


@dataclass
class AuthorizationServerMetadata:
    """OAuth Authorization Server Metadata."""
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
