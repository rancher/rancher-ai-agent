"""OAuth 2.0 data models and exceptions."""

from dataclasses import dataclass


class OAuth2Canceled(Exception):
    """Raised when the OAuth2 flow is cancelled by the user."""
    pass


class OAuthDiscoveryError(Exception):
    """Raised when OAuth discovery fails."""
    pass


class OAuthSecretError(Exception):
    """Raised when the OAuth credentials secret does not exist in Kubernetes."""
    pass

@dataclass
class OAuthClientCredentials:
    """OAuth2 client credentials retrieved from a Kubernetes secret."""
    client_id: str
    client_secret: str = ""
    scope: str = ""
    metadata_endpoint: str = ""