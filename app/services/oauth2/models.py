"""OAuth 2.0 data models and exceptions."""

from dataclasses import asdict, dataclass, field


class OAuthDiscoveryError(Exception):
    """Raised when OAuth discovery fails."""
    pass


class OAuthSecretError(Exception):
    """Raised when the OAuth credentials secret does not exist in Kubernetes."""
    pass


@dataclass
class ResourceMetadata:
    """OAuth Protected Resource Metadata."""
    resource: str = ""
    authorization_servers: list[str] = field(default_factory=list)
    scopes_supported: list[str] = field(default_factory=list)
    bearer_methods_supported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ResourceMetadata":
        return cls(
            resource=data.get("resource", ""),
            authorization_servers=data.get("authorization_servers", []),
            scopes_supported=data.get("scopes_supported", []),
            bearer_methods_supported=data.get("bearer_methods_supported", []),
        )


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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AuthorizationServerMetadata":
        return cls(
            issuer=data.get("issuer", ""),
            authorization_endpoint=data.get("authorization_endpoint", ""),
            token_endpoint=data.get("token_endpoint", ""),
            registration_endpoint=data.get("registration_endpoint"),
            scopes_supported=data.get("scopes_supported", []),
            response_types_supported=data.get("response_types_supported", []),
            code_challenge_methods_supported=data.get("code_challenge_methods_supported", []),
        )


@dataclass
class OAuthDiscoveryResult:
    """Complete result of the MCP OAuth discovery process."""
    auth_server_metadata: AuthorizationServerMetadata
    required_scopes: list[str] = field(default_factory=list)
    resource_metadata: ResourceMetadata | None = None

    def to_dict(self) -> dict:
        data = {
            "auth_server_metadata": self.auth_server_metadata.to_dict(),
            "required_scopes": self.required_scopes,
        }
        if self.resource_metadata:
            data["resource_metadata"] = self.resource_metadata.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "OAuthDiscoveryResult":
        resource_metadata = None
        if "resource_metadata" in data and data["resource_metadata"]:
            resource_metadata = ResourceMetadata.from_dict(data["resource_metadata"])
        auth_server_metadata = AuthorizationServerMetadata.from_dict(data["auth_server_metadata"])
        return cls(
            auth_server_metadata=auth_server_metadata,
            required_scopes=data.get("required_scopes", []),
            resource_metadata=resource_metadata,
        )


@dataclass
class OAuthClientCredentials:
    """OAuth2 client credentials retrieved from a Kubernetes secret."""
    client_id: str
    client_secret: str = ""
    scopes: str = ""
