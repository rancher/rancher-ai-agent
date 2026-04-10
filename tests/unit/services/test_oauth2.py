"""Unit tests for MCP OAuth 2.0 discovery and client implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.services.oauth2 import (
    OAuthClient,
    OAuthDiscoveryError,
    OAuthDiscoveryResult,
    ResourceMetadata,
    AuthorizationServerMetadata,
    _parse_www_authenticate,
    _discover_from_www_authenticate,
    _discover_from_well_known,
    _fetch_resource_metadata,
    _discover_auth_server_metadata,
    discover_oauth_metadata,
    get_redirect_uri,
)


# ──────────────────────────────────────────────────────────────
# Tests for _parse_www_authenticate
# ──────────────────────────────────────────────────────────────

class TestParseWWWAuthenticate:
    def test_full_header_with_resource_metadata_and_scope(self):
        header = 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="files:read"'
        url, scopes = _parse_www_authenticate(header)
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes == ["files:read"]

    def test_multiple_scopes(self):
        header = 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="files:read files:write admin"'
        url, scopes = _parse_www_authenticate(header)
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes == ["files:read", "files:write", "admin"]

    def test_resource_metadata_only(self):
        header = 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
        url, scopes = _parse_www_authenticate(header)
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes is None

    def test_scope_only(self):
        header = 'Bearer scope="read:jira-work"'
        url, scopes = _parse_www_authenticate(header)
        assert url is None
        assert scopes == ["read:jira-work"]

    def test_empty_header(self):
        url, scopes = _parse_www_authenticate("")
        assert url is None
        assert scopes is None

    def test_bearer_only(self):
        url, scopes = _parse_www_authenticate("Bearer")
        assert url is None
        assert scopes is None

    def test_multiline_header(self):
        header = (
            'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource",\n'
            '                         scope="files:read"'
        )
        url, scopes = _parse_www_authenticate(header)
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes == ["files:read"]


# ──────────────────────────────────────────────────────────────
# Tests for _discover_from_www_authenticate
# ──────────────────────────────────────────────────────────────

class TestDiscoverFromWWWAuthenticate:
    @pytest.mark.asyncio
    async def test_401_with_www_authenticate_header(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {
            "www-authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="files:read"'
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/v1/mcp")
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes == ["files:read"]

    @pytest.mark.asyncio
    async def test_401_with_mixed_case_www_authenticate_header(self):
        """Header lookup must be case-insensitive (e.g. Www-Authenticate from some servers)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {
            "Www-Authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="files:read"'
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/v1/mcp")
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes == ["files:read"]

    @pytest.mark.asyncio
    async def test_401_with_uppercase_www_authenticate_header(self):
        """Header lookup must work for all-uppercase WWW-Authenticate."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {
            "WWW-Authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/v1/mcp")
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert scopes is None

    @pytest.mark.asyncio
    async def test_401_without_www_authenticate_header(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/v1/mcp")
        assert url is None
        assert scopes is None

    @pytest.mark.asyncio
    async def test_non_401_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/v1/mcp")
        assert url is None
        assert scopes is None

    @pytest.mark.asyncio
    async def test_connection_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("Connection refused"))

        url, scopes = await _discover_from_www_authenticate(mock_client, "https://mcp.example.com/v1/mcp")
        assert url is None
        assert scopes is None


# ──────────────────────────────────────────────────────────────
# Tests for _discover_from_well_known
# ──────────────────────────────────────────────────────────────

class TestDiscoverFromWellKnown:
    @pytest.mark.asyncio
    async def test_path_specific_well_known_found(self):
        """Should try path-specific well-known URI first."""
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response_200)

        url = await _discover_from_well_known(mock_client, "https://mcp.example.com/v1/mcp")
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource/v1/mcp"
        mock_client.get.assert_called_once_with("https://mcp.example.com/.well-known/oauth-protected-resource/v1/mcp")

    @pytest.mark.asyncio
    async def test_root_well_known_fallback(self):
        """If path-specific fails, should try root well-known URI."""
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_response_404, mock_response_200])

        url = await _discover_from_well_known(mock_client, "https://mcp.example.com/v1/mcp")
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"

    @pytest.mark.asyncio
    async def test_no_path_only_root(self):
        """URL without path should only try root well-known."""
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response_200)

        url = await _discover_from_well_known(mock_client, "https://mcp.example.com")
        assert url == "https://mcp.example.com/.well-known/oauth-protected-resource"
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_well_known_fail(self):
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response_404)

        url = await _discover_from_well_known(mock_client, "https://mcp.example.com/v1/mcp")
        assert url is None


# ──────────────────────────────────────────────────────────────
# Tests for _fetch_resource_metadata
# ──────────────────────────────────────────────────────────────

class TestFetchResourceMetadata:
    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "resource": "https://mcp.example.com",
            "authorization_servers": ["https://auth.example.com"],
            "scopes_supported": ["files:read", "files:write"],
            "bearer_methods_supported": ["header"],
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        metadata = await _fetch_resource_metadata(mock_client, "https://mcp.example.com/.well-known/oauth-protected-resource")
        assert metadata.resource == "https://mcp.example.com"
        assert metadata.authorization_servers == ["https://auth.example.com"]
        assert metadata.scopes_supported == ["files:read", "files:write"]

    @pytest.mark.asyncio
    async def test_http_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("Server error"))

        with pytest.raises(OAuthDiscoveryError, match="Failed to fetch resource metadata"):
            await _fetch_resource_metadata(mock_client, "https://mcp.example.com/.well-known/oauth-protected-resource")


# ──────────────────────────────────────────────────────────────
# Tests for _discover_auth_server_metadata
# ──────────────────────────────────────────────────────────────

class TestDiscoverAuthServerMetadata:
    @pytest.mark.asyncio
    async def test_issuer_without_path(self):
        """Should try oauth-authorization-server then openid-configuration."""
        auth_metadata = {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "registration_endpoint": "https://auth.example.com/register",
            "scopes_supported": ["openid", "files:read"],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value=auth_metadata)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await _discover_auth_server_metadata(mock_client, "https://auth.example.com")
        assert result.authorization_endpoint == "https://auth.example.com/authorize"
        assert result.token_endpoint == "https://auth.example.com/token"
        assert result.registration_endpoint == "https://auth.example.com/register"
        mock_client.get.assert_called_once_with("https://auth.example.com/.well-known/oauth-authorization-server")

    @pytest.mark.asyncio
    async def test_issuer_with_path_fallback_to_openid(self):
        """For issuer with path, should try oauth-authorization-server then openid-configuration."""
        auth_metadata = {
            "issuer": "https://auth.example.com/tenant1",
            "authorization_endpoint": "https://auth.example.com/tenant1/authorize",
            "token_endpoint": "https://auth.example.com/tenant1/token",
        }

        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json = MagicMock(return_value=auth_metadata)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_response_404, mock_response_200])

        result = await _discover_auth_server_metadata(mock_client, "https://auth.example.com/tenant1")
        assert result.authorization_endpoint == "https://auth.example.com/tenant1/authorize"
        assert result.token_endpoint == "https://auth.example.com/tenant1/token"
        # Second call should be openid-configuration with path insertion
        assert mock_client.get.call_args_list[1][0][0] == "https://auth.example.com/.well-known/openid-configuration/tenant1"

    @pytest.mark.asyncio
    async def test_issuer_with_path_appended_openid(self):
        """Should try path-appended openid-configuration as last resort."""
        auth_metadata = {
            "issuer": "https://auth.example.com/tenant1",
            "authorization_endpoint": "https://auth.example.com/tenant1/authorize",
            "token_endpoint": "https://auth.example.com/tenant1/token",
        }

        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json = MagicMock(return_value=auth_metadata)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_response_404, mock_response_404, mock_response_200])

        result = await _discover_auth_server_metadata(mock_client, "https://auth.example.com/tenant1")
        assert result.authorization_endpoint == "https://auth.example.com/tenant1/authorize"
        # Third call should be path-appended
        assert mock_client.get.call_args_list[2][0][0] == "https://auth.example.com/tenant1/.well-known/openid-configuration"

    @pytest.mark.asyncio
    async def test_issuer_with_path_tries_root_level_fallback(self):
        """When path-specific attempts fail, should fall back to root-level well-known URIs."""
        auth_metadata = {
            "issuer": "https://mcp.atlassian.com",
            "authorization_endpoint": "https://mcp.atlassian.com/v1/authorize",
            "token_endpoint": "https://cf.mcp.atlassian.com/v1/token",
            "registration_endpoint": "https://cf.mcp.atlassian.com/v1/register",
        }

        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json = MagicMock(return_value=auth_metadata)

        mock_client = AsyncMock()
        # path-insertion attempts all 404, then root-level /.well-known/oauth-authorization-server succeeds
        mock_client.get = AsyncMock(side_effect=[
            mock_response_404,  # /.well-known/oauth-authorization-server/v1/mcp
            mock_response_404,  # /.well-known/openid-configuration/v1/mcp
            mock_response_404,  # /v1/mcp/.well-known/openid-configuration
            mock_response_200,  # /.well-known/oauth-authorization-server  ← root fallback
        ])

        result = await _discover_auth_server_metadata(mock_client, "https://mcp.atlassian.com/v1/mcp")
        assert result.authorization_endpoint == "https://mcp.atlassian.com/v1/authorize"
        assert result.token_endpoint == "https://cf.mcp.atlassian.com/v1/token"
        assert mock_client.get.call_args_list[3][0][0] == "https://mcp.atlassian.com/.well-known/oauth-authorization-server"

    @pytest.mark.asyncio
    async def test_json_decode_error_is_skipped(self):
        """A non-JSON 200 response should be skipped, not raise an unhandled exception."""
        import json as _json

        mock_response_html = MagicMock()
        mock_response_html.status_code = 200
        mock_response_html.json = MagicMock(side_effect=_json.JSONDecodeError("Expecting value", "", 0))

        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404

        mock_client = AsyncMock()
        # First call returns HTML (non-JSON), rest 404
        mock_client.get = AsyncMock(side_effect=[mock_response_html, mock_response_404])

        with pytest.raises(OAuthDiscoveryError, match="Failed to discover authorization server metadata"):
            await _discover_auth_server_metadata(mock_client, "https://auth.example.com")

    @pytest.mark.asyncio
    async def test_all_discovery_attempts_fail(self):
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response_404)

        with pytest.raises(OAuthDiscoveryError, match="Failed to discover authorization server metadata"):
            await _discover_auth_server_metadata(mock_client, "https://auth.example.com")


# ──────────────────────────────────────────────────────────────
# Tests for discover_oauth_metadata (full chain)
# ──────────────────────────────────────────────────────────────

class TestDiscoverOAuthMetadata:
    @pytest.mark.asyncio
    async def test_full_discovery_via_www_authenticate(self):
        """Full chain: 401 → resource metadata → auth server metadata."""
        mcp_url = "https://mcp.example.com/v1/mcp"

        # Step 1: MCP server returns 401 with WWW-Authenticate header
        mcp_response = MagicMock()
        mcp_response.status_code = 401
        mcp_response.headers = {
            "www-authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="files:read"'
        }

        # Step 2: Resource metadata
        resource_response = MagicMock()
        resource_response.status_code = 200
        resource_response.raise_for_status = MagicMock()
        resource_response.json = MagicMock(return_value={
            "resource": "https://mcp.example.com",
            "authorization_servers": ["https://auth.example.com"],
            "scopes_supported": ["files:read", "files:write"],
        })

        # Step 3: Auth server metadata
        auth_response = MagicMock()
        auth_response.status_code = 200
        auth_response.json = MagicMock(return_value={
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "registration_endpoint": "https://auth.example.com/register",
            "scopes_supported": ["files:read", "files:write"],
            "code_challenge_methods_supported": ["S256"],
        })

        with patch("app.services.oauth2.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[mcp_response, resource_response, auth_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await discover_oauth_metadata(mcp_url)

        assert result.authorization_endpoint == "https://auth.example.com/authorize"
        assert result.token_endpoint == "https://auth.example.com/token"
        assert result.registration_endpoint == "https://auth.example.com/register"
        assert result.required_scopes == ["files:read"]  # From WWW-Authenticate, not resource metadata

    @pytest.mark.asyncio
    async def test_full_discovery_via_well_known_fallback(self):
        """Fall back to well-known URIs when 401 has no WWW-Authenticate."""
        mcp_url = "https://mcp.example.com/v1/mcp"

        # Step 1: MCP server returns 200 (no 401)
        mcp_response = MagicMock()
        mcp_response.status_code = 200
        mcp_response.headers = {}

        # Step 2: Path-specific well-known fails, root succeeds
        wellknown_404 = MagicMock()
        wellknown_404.status_code = 404
        wellknown_200 = MagicMock()
        wellknown_200.status_code = 200

        # Step 3: Resource metadata fetch
        resource_response = MagicMock()
        resource_response.status_code = 200
        resource_response.raise_for_status = MagicMock()
        resource_response.json = MagicMock(return_value={
            "resource": "https://mcp.example.com",
            "authorization_servers": ["https://auth.example.com"],
            "scopes_supported": ["files:read"],
        })

        # Step 4: Auth server metadata
        auth_response = MagicMock()
        auth_response.status_code = 200
        auth_response.json = MagicMock(return_value={
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        })

        with patch("app.services.oauth2.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            # mcp GET → 200, path well-known → 404, root well-known → 200, fetch resource → json, auth server → json
            mock_client.get = AsyncMock(side_effect=[
                mcp_response,       # 401 check
                wellknown_404,      # path-specific well-known
                wellknown_200,      # root well-known
                resource_response,  # fetch resource metadata
                auth_response,      # auth server metadata
            ])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await discover_oauth_metadata(mcp_url)

        assert result.authorization_endpoint == "https://auth.example.com/authorize"
        assert result.token_endpoint == "https://auth.example.com/token"
        # Scopes from resource metadata since WWW-Authenticate had none
        assert result.required_scopes == ["files:read"]

    @pytest.mark.asyncio
    async def test_fallback_direct_auth_server_discovery(self):
        """
        Atlassian-like scenario: WWW-Authenticate has no resource_metadata,
        well-known resource metadata returns 404, but RFC 8414 auth server
        metadata is available at the root-level well-known path.
        """
        mcp_url = "https://mcp.atlassian.com/v1/mcp"

        # Step 1: MCP returns 401 with no resource_metadata in WWW-Authenticate
        mcp_401 = MagicMock()
        mcp_401.status_code = 401
        mcp_401.headers = {"Www-Authenticate": 'Bearer realm="OAuth", error="invalid_token"'}

        # Steps 2: Both well-known resource metadata URLs return 404
        r_404 = MagicMock()
        r_404.status_code = 404

        # Step 4 fallback: path-specific auth server well-known URLs 404,
        # then root-level /.well-known/oauth-authorization-server returns 200
        auth_metadata = {
            "issuer": "https://cf.mcp.atlassian.com",
            "authorization_endpoint": "https://mcp.atlassian.com/v1/authorize",
            "token_endpoint": "https://cf.mcp.atlassian.com/v1/token",
            "registration_endpoint": "https://cf.mcp.atlassian.com/v1/register",
            "scopes_supported": ["read:jira-work", "write:jira-work"],
        }
        r_200 = MagicMock()
        r_200.status_code = 200
        r_200.json = MagicMock(return_value=auth_metadata)

        with patch("app.services.oauth2.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[
                mcp_401,   # 401 check (no resource_metadata)
                r_404,     # /.well-known/oauth-protected-resource/v1/mcp
                r_404,     # /.well-known/oauth-protected-resource (root)
                r_404,     # /.well-known/oauth-authorization-server/v1/mcp
                r_404,     # /.well-known/openid-configuration/v1/mcp
                r_404,     # /v1/mcp/.well-known/openid-configuration
                r_200,     # /.well-known/oauth-authorization-server (root fallback)
            ])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await discover_oauth_metadata(mcp_url)

        assert result.authorization_endpoint == "https://mcp.atlassian.com/v1/authorize"
        assert result.token_endpoint == "https://cf.mcp.atlassian.com/v1/token"
        assert result.registration_endpoint == "https://cf.mcp.atlassian.com/v1/register"
        # required_scopes falls back to auth server scopes_supported
        assert "read:jira-work" in result.required_scopes
        assert result.resource_metadata is None  # No resource metadata was found

    @pytest.mark.asyncio
    async def test_discovery_fails_when_all_strategies_exhausted(self):
        """Should raise error when no resource metadata AND no direct auth server metadata."""
        mcp_url = "https://mcp.example.com/v1/mcp"

        # MCP server returns 200 (no 401)
        mcp_response = MagicMock()
        mcp_response.status_code = 200
        mcp_response.headers = {}

        # All well-known URIs fail
        r_404 = MagicMock()
        r_404.status_code = 404

        with patch("app.services.oauth2.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            # Everything returns 404: 1 mcp + 2 resource well-known + 5 auth server well-known
            mock_client.get = AsyncMock(return_value=r_404)
            mock_client.get = AsyncMock(side_effect=[
                mcp_response,  # 401 check → 200, no resource_metadata
                r_404, r_404,  # resource well-known: path-specific + root
                r_404, r_404, r_404, r_404, r_404,  # auth server fallback: 3 path + 2 root
            ])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(OAuthDiscoveryError, match="Failed to discover OAuth metadata"):
                await discover_oauth_metadata(mcp_url)

    @pytest.mark.asyncio
    async def test_discovery_fails_no_authorization_servers(self):
        """Should raise error when resource metadata has no authorization_servers."""
        mcp_url = "https://mcp.example.com/v1/mcp"

        mcp_response = MagicMock()
        mcp_response.status_code = 401
        mcp_response.headers = {
            "www-authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
        }

        resource_response = MagicMock()
        resource_response.status_code = 200
        resource_response.raise_for_status = MagicMock()
        resource_response.json = MagicMock(return_value={
            "resource": "https://mcp.example.com",
            "authorization_servers": [],
        })

        with patch("app.services.oauth2.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[mcp_response, resource_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(OAuthDiscoveryError, match="did not include any authorization servers"):
                await discover_oauth_metadata(mcp_url)


# ──────────────────────────────────────────────────────────────
# Tests for get_redirect_uri
# ──────────────────────────────────────────────────────────────

class TestGetRedirectUri:
    def test_from_environment(self):
        with patch.dict("os.environ", {"OAUTH_REDIRECT_URI": "https://myapp.example.com/oauth/callback"}):
            assert get_redirect_uri() == "https://myapp.example.com/oauth/callback"

    def test_from_websocket_wss(self):
        with patch.dict("os.environ", {}, clear=True):
            mock_ws = MagicMock()
            mock_ws.url.scheme = "wss"
            mock_ws.url.hostname = "rancher.example.com"
            mock_ws.url.port = None
            assert get_redirect_uri(mock_ws) == "https://rancher.example.com/oauth/callback"

    def test_from_websocket_ws_with_port(self):
        with patch.dict("os.environ", {}, clear=True):
            mock_ws = MagicMock()
            mock_ws.url.scheme = "ws"
            mock_ws.url.hostname = "localhost"
            mock_ws.url.port = 8000
            assert get_redirect_uri(mock_ws) == "http://localhost:8000/oauth/callback"

    def test_from_websocket_standard_port_omitted(self):
        with patch.dict("os.environ", {}, clear=True):
            mock_ws = MagicMock()
            mock_ws.url.scheme = "wss"
            mock_ws.url.hostname = "rancher.example.com"
            mock_ws.url.port = 443
            assert get_redirect_uri(mock_ws) == "https://rancher.example.com/oauth/callback"

    def test_default_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_redirect_uri() == "http://localhost:8000/oauth/callback"

    def test_env_takes_precedence_over_websocket(self):
        with patch.dict("os.environ", {"OAUTH_REDIRECT_URI": "https://configured.example.com/callback"}):
            mock_ws = MagicMock()
            mock_ws.url.scheme = "ws"
            mock_ws.url.hostname = "localhost"
            mock_ws.url.port = 8000
            assert get_redirect_uri(mock_ws) == "https://configured.example.com/callback"


# ──────────────────────────────────────────────────────────────
# Tests for OAuthClient
# ──────────────────────────────────────────────────────────────

class TestOAuthClient:
    def test_init(self):
        client = OAuthClient(client_id="test-id", client_secret="test-secret", scope="read write")
        assert client.client_id == "test-id"
        assert client.client_secret == "test-secret"
        assert client.scope == "read write"

    def test_generate_pkce_pair(self):
        client = OAuthClient(client_id="test-id")
        verifier, challenge = client.generate_pkce_pair()
        assert len(verifier) > 0
        assert len(challenge) > 0
        assert verifier != challenge

    def test_pkce_pair_is_unique(self):
        client = OAuthClient(client_id="test-id")
        v1, c1 = client.generate_pkce_pair()
        v2, c2 = client.generate_pkce_pair()
        assert v1 != v2
        assert c1 != c2

    @pytest.mark.asyncio
    async def test_dynamic_registration_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "client_id": "registered-client-id",
            "client_secret": "registered-client-secret",
        })

        with patch("app.services.oauth2.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            client = await OAuthClient.from_dynamic_registration(
                registration_endpoint="https://auth.example.com/register",
                redirect_uri="http://localhost:8000/oauth/callback",
                scope="files:read",
            )

        assert client.client_id == "registered-client-id"
        assert client.client_secret == "registered-client-secret"
        assert client.scope == "files:read"

    @pytest.mark.asyncio
    async def test_dynamic_registration_failure(self):
        with patch("app.services.oauth2.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.RequestError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(OAuthDiscoveryError, match="Dynamic client registration failed"):
                await OAuthClient.from_dynamic_registration(
                    registration_endpoint="https://auth.example.com/register",
                    redirect_uri="http://localhost:8000/oauth/callback",
                )
