"""Tests for app.services.oauth2.discovery"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from app.services.oauth2.discovery import (
    _parse_www_authenticate,
    _discover_from_www_authenticate,
    _discover_from_well_known,
    _fetch_resource_metadata,
    _discover_auth_server_metadata,
    discover_oauth_metadata,
)
from app.services.oauth2.models import (
    OAuthDiscoveryError,
    ResourceMetadata,
    AuthorizationServerMetadata,
)


class TestParseWwwAuthenticate:
    def test_full_header(self):
        header = 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="files:read files:write"'
        url, scopes = _parse_www_authenticate(header)
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes == ["files:read", "files:write"]

    def test_resource_metadata_only(self):
        header = 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
        url, scopes = _parse_www_authenticate(header)
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes is None

    def test_scope_only(self):
        header = 'Bearer scope="read write"'
        url, scopes = _parse_www_authenticate(header)
        assert url is None
        assert scopes == ["read", "write"]

    def test_empty_header(self):
        url, scopes = _parse_www_authenticate("")
        assert url is None
        assert scopes is None

    def test_no_bearer(self):
        header = 'Basic realm="test"'
        url, scopes = _parse_www_authenticate(header)
        assert url is None
        assert scopes is None

    def test_single_scope(self):
        header = 'Bearer scope="read"'
        url, scopes = _parse_www_authenticate(header)
        assert scopes == ["read"]


class TestDiscoverFromWwwAuthenticate:
    @pytest.mark.asyncio
    async def test_401_with_www_authenticate(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = httpx.Headers(
            {"www-authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="read"'}
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com")
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes == ["read"]

    @pytest.mark.asyncio
    async def test_non_401_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com")
        assert url is None
        assert scopes is None

    @pytest.mark.asyncio
    async def test_401_without_header(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = httpx.Headers({})
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com")
        assert url is None
        assert scopes is None

    @pytest.mark.asyncio
    async def test_request_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("connection failed"))

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com")
        assert url is None
        assert scopes is None


class TestDiscoverFromWellKnown:
    @pytest.mark.asyncio
    async def test_found_with_path(self):
        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response_ok)

        result = await _discover_from_well_known(mock_client, "https://mcp.example.com/path/to/mcp")
        assert result == "https://mcp.example.com/.well-known/oauth-protected-resource/path/to/mcp"

    @pytest.mark.asyncio
    async def test_found_at_root(self):
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_response_404, mock_response_ok])

        result = await _discover_from_well_known(mock_client, "https://mcp.example.com/path")
        assert result == "https://mcp.example.com/.well-known/oauth-protected-resource"

    @pytest.mark.asyncio
    async def test_not_found(self):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _discover_from_well_known(mock_client, "https://mcp.example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_request_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("timeout"))

        result = await _discover_from_well_known(mock_client, "https://mcp.example.com")
        assert result is None


class TestFetchResourceMetadata:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "resource": "https://mcp.example.com",
            "authorization_servers": ["https://auth.example.com"],
            "scopes_supported": ["read"],
            "bearer_methods_supported": ["header"],
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        rm = await _fetch_resource_metadata(mock_client, "https://mcp.example.com/.well-known/oauth-protected-resource")
        assert rm.resource == "https://mcp.example.com"
        assert rm.authorization_servers == ["https://auth.example.com"]

    @pytest.mark.asyncio
    async def test_http_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("fail"))

        with pytest.raises(OAuthDiscoveryError, match="Failed to fetch resource metadata"):
            await _fetch_resource_metadata(mock_client, "https://mcp.example.com/.well-known/oauth-protected-resource")


class TestDiscoverAuthServerMetadata:
    @pytest.mark.asyncio
    async def test_success_no_path(self):
        metadata = {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "registration_endpoint": "https://auth.example.com/register",
            "scopes_supported": ["openid"],
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = metadata
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _discover_auth_server_metadata(mock_client, "https://auth.example.com")
        assert result.authorization_endpoint == "https://auth.example.com/authorize"
        assert result.token_endpoint == "https://auth.example.com/token"
        assert result.registration_endpoint == "https://auth.example.com/register"

    @pytest.mark.asyncio
    async def test_success_with_path(self):
        metadata = {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.json.return_value = metadata
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_response_404, mock_response_ok])

        result = await _discover_auth_server_metadata(mock_client, "https://auth.example.com/tenant1")
        assert result.authorization_endpoint == "https://auth.example.com/authorize"

    @pytest.mark.asyncio
    async def test_all_fail(self):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(OAuthDiscoveryError, match="Failed to discover authorization server metadata"):
            await _discover_auth_server_metadata(mock_client, "https://auth.example.com")


class TestDiscoverOAuthMetadata:
    @pytest.mark.asyncio
    @patch("app.services.oauth2.discovery._get_tls_verify", return_value=True)
    async def test_full_discovery_chain(self, _mock_tls):
        resource_metadata_response = MagicMock()
        resource_metadata_response.json.return_value = {
            "resource": "https://mcp.example.com",
            "authorization_servers": ["https://auth.example.com"],
            "scopes_supported": ["read"],
        }
        resource_metadata_response.raise_for_status = MagicMock()

        auth_server_response = MagicMock()
        auth_server_response.status_code = 200
        auth_server_response.json.return_value = {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        with patch("app.services.oauth2.discovery._discover_from_www_authenticate") as mock_www, \
             patch("app.services.oauth2.discovery._discover_from_well_known") as mock_wk, \
             patch("app.services.oauth2.discovery._fetch_resource_metadata") as mock_fetch_rm, \
             patch("app.services.oauth2.discovery._discover_auth_server_metadata") as mock_auth:

            mock_www.return_value = ("https://mcp.example.com/.well-known/oauth-protected-resource", ["read"])
            mock_fetch_rm.return_value = ResourceMetadata(
                resource="https://mcp.example.com",
                authorization_servers=["https://auth.example.com"],
                scopes_supported=["read"],
            )
            mock_auth.return_value = AuthorizationServerMetadata(
                issuer="https://auth.example.com",
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint="https://auth.example.com/token",
            )

            result = await discover_oauth_metadata("https://mcp.example.com")
            assert result.authorization_endpoint == "https://auth.example.com/authorize"
            assert result.token_endpoint == "https://auth.example.com/token"
            assert result.required_scopes == ["read"]
            mock_wk.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.oauth2.discovery._get_tls_verify", return_value=True)
    async def test_fallback_to_direct_auth_server(self, _mock_tls):
        with patch("app.services.oauth2.discovery._discover_from_www_authenticate") as mock_www, \
             patch("app.services.oauth2.discovery._discover_from_well_known") as mock_wk, \
             patch("app.services.oauth2.discovery._discover_auth_server_metadata") as mock_auth:

            mock_www.return_value = (None, None)
            mock_wk.return_value = None
            mock_auth.return_value = AuthorizationServerMetadata(
                issuer="https://mcp.example.com",
                authorization_endpoint="https://mcp.example.com/authorize",
                token_endpoint="https://mcp.example.com/token",
                scopes_supported=["read"],
            )

            result = await discover_oauth_metadata("https://mcp.example.com")
            assert result.authorization_endpoint == "https://mcp.example.com/authorize"
            assert result.resource_metadata is None

    @pytest.mark.asyncio
    @patch("app.services.oauth2.discovery._get_tls_verify", return_value=True)
    async def test_all_discovery_fails(self, _mock_tls):
        with patch("app.services.oauth2.discovery._discover_from_www_authenticate") as mock_www, \
             patch("app.services.oauth2.discovery._discover_from_well_known") as mock_wk, \
             patch("app.services.oauth2.discovery._discover_auth_server_metadata") as mock_auth:

            mock_www.return_value = (None, None)
            mock_wk.return_value = None
            mock_auth.side_effect = OAuthDiscoveryError("fail")

            with pytest.raises(OAuthDiscoveryError, match="Failed to discover OAuth metadata"):
                await discover_oauth_metadata("https://mcp.example.com")

    @pytest.mark.asyncio
    @patch("app.services.oauth2.discovery._get_tls_verify", return_value=True)
    async def test_resource_metadata_no_auth_servers(self, _mock_tls):
        with patch("app.services.oauth2.discovery._discover_from_www_authenticate") as mock_www, \
             patch("app.services.oauth2.discovery._discover_from_well_known") as mock_wk, \
             patch("app.services.oauth2.discovery._fetch_resource_metadata") as mock_fetch_rm:

            mock_www.return_value = ("https://mcp.example.com/.well-known/opr", None)
            mock_fetch_rm.return_value = ResourceMetadata(
                resource="https://mcp.example.com",
                authorization_servers=[],
            )

            with pytest.raises(OAuthDiscoveryError, match="did not include any authorization servers"):
                await discover_oauth_metadata("https://mcp.example.com")
