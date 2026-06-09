"""OAuth2 client with PKCE support for MCP server authentication."""

import hashlib
import base64
import os
import secrets
import logging

import httpx

from authlib.integrations.httpx_client import AsyncOAuth2Client

from .models import OAuthDiscoveryError

logger = logging.getLogger(__name__)


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
                registration_data["scope"] = scope

            try:
                response = await http_client.post(registration_endpoint, json=registration_data)
                response.raise_for_status()
                data = response.json()

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
            scope=self.scope,
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
            scope=self.scope,
        )
        return token


def _get_tls_verify() -> bool:
    """Get TLS verification setting from environment."""
    return os.environ.get('INSECURE_SKIP_TLS', 'false').lower() != 'true'
