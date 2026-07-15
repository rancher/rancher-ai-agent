"""Tests for app.services.oauth2.discovery"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.oauth2.discovery import (
    DiscoveredMetadata,
    _parse_www_authenticate,
    _discover_from_www_authenticate,
    _discover_auth_server_metadata_endpoint,
    discover_metadata_endpoint,
)
from app.services.oauth2.models import OAuthDiscoveryError


class TestParseWwwAuthenticate:
    def test_extracts_resource_metadata_url(self):
        header = 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
        result = _parse_www_authenticate(header)
        assert result == "https://mcp.example.com/.well-known/oauth-protected-resource"

    def test_returns_none_when_no_resource_metadata(self):
        header = 'Bearer realm="example"'
        result = _parse_www_authenticate(header)
        assert result is None

    def test_returns_none_for_empty_header(self):
        result = _parse_www_authenticate("")
        assert result is None

    def test_handles_extra_parameters(self):
        header = 'Bearer realm="example", resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="openid"'
        result = _parse_www_authenticate(header)
        assert result == "https://mcp.example.com/.well-known/oauth-protected-resource"


class TestDiscoverFromWwwAuthenticate:
    @pytest.mark.asyncio
    async def test_returns_url_from_401_with_www_authenticate(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {
            "www-authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/sse")
        assert result == "https://mcp.example.com/.well-known/oauth-protected-resource"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_401(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/sse")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_www_authenticate_header(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/sse")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_request_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("connection failed", request=MagicMock()))

        result = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/sse")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_case_insensitive_header(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {
            "Www-Authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/sse")
        assert result == "https://mcp.example.com/.well-known/oauth-protected-resource"


class TestDiscoverAuthServerMetadataEndpoint:
    @pytest.mark.asyncio
    async def test_finds_oauth_authorization_server_at_root(self):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _discover_auth_server_metadata_endpoint(
            mock_client, "https://auth.example.com"
        )
        assert result == "https://auth.example.com/.well-known/oauth-authorization-server"

    @pytest.mark.asyncio
    async def test_finds_oauth_authorization_server_with_path(self):
        async def mock_get(url):
            response = MagicMock()
            if url == "https://auth.example.com/.well-known/oauth-authorization-server/tenant1":
                response.status_code = 200
            else:
                response.status_code = 404
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get

        result = await _discover_auth_server_metadata_endpoint(
            mock_client, "https://auth.example.com/tenant1"
        )
        assert result == "https://auth.example.com/.well-known/oauth-authorization-server/tenant1"

    @pytest.mark.asyncio
    async def test_falls_back_to_openid_configuration(self):
        async def mock_get(url):
            response = MagicMock()
            if url == "https://auth.example.com/.well-known/openid-configuration":
                response.status_code = 200
            else:
                response.status_code = 404
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get

        result = await _discover_auth_server_metadata_endpoint(
            mock_client, "https://auth.example.com"
        )
        assert result == "https://auth.example.com/.well-known/openid-configuration"

    @pytest.mark.asyncio
    async def test_falls_back_to_root_with_path(self):
        """When path-based URLs fail, falls back to root-level endpoints."""
        async def mock_get(url):
            response = MagicMock()
            if url == "https://auth.example.com/.well-known/oauth-authorization-server":
                response.status_code = 200
            else:
                response.status_code = 404
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get

        result = await _discover_auth_server_metadata_endpoint(
            mock_client, "https://auth.example.com/tenant1"
        )
        assert result == "https://auth.example.com/.well-known/oauth-authorization-server"

    @pytest.mark.asyncio
    async def test_raises_when_all_attempts_fail(self):
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(OAuthDiscoveryError, match="Failed to discover"):
            await _discover_auth_server_metadata_endpoint(
                mock_client, "https://auth.example.com"
            )

    @pytest.mark.asyncio
    async def test_handles_request_errors_gracefully(self):
        async def mock_get(url):
            if "oauth-authorization-server" in url:
                raise httpx.RequestError("connection failed", request=MagicMock())
            response = MagicMock()
            response.status_code = 200
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get

        result = await _discover_auth_server_metadata_endpoint(
            mock_client, "https://auth.example.com"
        )
        assert result == "https://auth.example.com/.well-known/openid-configuration"


class TestDiscoverMetadataEndpoint:
    @pytest.mark.asyncio
    @patch("app.services.oauth2.discovery.get_tls_verify", return_value=True)
    async def test_discovers_metadata_successfully(self, mock_tls):
        resource_metadata_url = "https://mcp.example.com/.well-known/oauth-protected-resource"
        auth_server_url = "https://auth.example.com/.well-known/oauth-authorization-server"

        with patch("app.services.oauth2.discovery._discover_from_www_authenticate") as mock_www_auth, \
             patch("app.services.oauth2.discovery._discover_auth_server_metadata_endpoint") as mock_auth_server:

            mock_www_auth.return_value = resource_metadata_url

            resource_response = MagicMock()
            resource_response.status_code = 200
            resource_response.json.return_value = {
                "authorization_servers": ["https://auth.example.com"],
                "scopes_supported": ["openid", "profile"],
            }
            resource_response.raise_for_status = MagicMock()

            mock_auth_server.return_value = auth_server_url

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client_instance = AsyncMock()
                mock_client_instance.get = AsyncMock(return_value=resource_response)
                mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client_instance

                result = await discover_metadata_endpoint("https://mcp.example.com/sse")

                assert isinstance(result, DiscoveredMetadata)
                assert result.metadataEndpoint == auth_server_url
                assert result.scopesSupported == ["openid", "profile"]

    @pytest.mark.asyncio
    @patch("app.services.oauth2.discovery.get_tls_verify", return_value=True)
    async def test_raises_when_www_authenticate_fails(self, mock_tls):
        with patch("app.services.oauth2.discovery._discover_from_www_authenticate") as mock_www_auth:
            mock_www_auth.return_value = None

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client_instance = AsyncMock()
                mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client_instance

                with pytest.raises(OAuthDiscoveryError, match="Failed to discover"):
                    await discover_metadata_endpoint("https://mcp.example.com/sse")

    @pytest.mark.asyncio
    @patch("app.services.oauth2.discovery.get_tls_verify", return_value=True)
    async def test_raises_when_no_authorization_servers(self, mock_tls):
        resource_metadata_url = "https://mcp.example.com/.well-known/oauth-protected-resource"

        with patch("app.services.oauth2.discovery._discover_from_www_authenticate") as mock_www_auth:
            mock_www_auth.return_value = resource_metadata_url

            resource_response = MagicMock()
            resource_response.status_code = 200
            resource_response.json.return_value = {
                "authorization_servers": [],
                "scopes_supported": [],
            }
            resource_response.raise_for_status = MagicMock()

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client_instance = AsyncMock()
                mock_client_instance.get = AsyncMock(return_value=resource_response)
                mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client_instance

                with pytest.raises(OAuthDiscoveryError, match="did not include any authorization servers"):
                    await discover_metadata_endpoint("https://mcp.example.com/sse")
