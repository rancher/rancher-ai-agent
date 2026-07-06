"""OAuth2 client manager using authlib starlette integration."""

import logging
import os

from authlib.integrations.starlette_client import OAuth

logger = logging.getLogger(__name__)


class OAuthClientManager:
    """Singleton OAuth client manager using authlib starlette integration.

    Manages OAuth2 client registrations for AIAgentConfig resources with
    OAUTH2 authentication type. Each agent is registered as a named client
    with its credentials and server metadata URL from the Kubernetes secret.
    """

    _instance: "OAuthClientManager | None" = None

    def __init__(self):
        self._oauth = OAuth()

    @classmethod
    def get_instance(cls) -> "OAuthClientManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_client(
        self,
        name: str,
        client_id: str,
        client_secret: str,
        scope: str,
        server_metadata_url: str,
    ) -> None:
        """Register or update an OAuth client for an agent.

        If a client with the same name already exists, it is removed and
        re-registered with the new configuration.

        Args:
            name: Agent name used as the client key.
            client_id: OAuth2 client ID from the authentication secret.
            client_secret: OAuth2 client secret from the authentication secret.
            scope: Space-separated OAuth2 scopes.
            server_metadata_url: OpenID/OAuth2 server metadata endpoint URL.
        """
        # Remove existing registration so it can be re-registered
        self._oauth._registry.pop(name, None)
        self._oauth._clients.pop(name, None)

        self._oauth.register(
            name=name,
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=server_metadata_url,
            client_kwargs={
                "scope": scope,
                "code_challenge_method": "S256",
            },
        )
        logger.info(f"Registered OAuth client for agent '{name}'")

    def remove_client(self, name: str) -> None:
        """Remove a registered OAuth client."""
        self._oauth._registry.pop(name, None)
        self._oauth._clients.pop(name, None)

    def get_client(self, name: str):
        """Get the registered OAuth client for an agent.

        Returns:
            A StarletteOAuth2App instance, or None if not registered.
        """
        return self._oauth.create_client(name)

    def has_client(self, name: str) -> bool:
        """Check if a client is registered for the given agent name."""
        return name in self._oauth._registry


def get_tls_verify() -> bool:
    """Get TLS verification setting from environment."""
    return os.environ.get('INSECURE_SKIP_TLS', 'false').lower() != 'true'
