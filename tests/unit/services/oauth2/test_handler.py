"""Tests for app.services.oauth2.handler"""

import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.agent._constants import NoAgentAvailableError
from app.services.oauth2.handler import (
    _inject_oauth_cookie,
    _initiate_oauth_flow,
    _try_refresh_oauth_token,
    get_redirect_uri,
    handle_oauth_authentication,
)
from app.services.oauth2.models import AuthorizationServerMetadata, OAuthClientCredentials, OAuthDiscoveryResult, OAuthSecretError


class TestGetRedirectUri:
    def test_from_env_variable(self):
        with patch.dict(os.environ, {"OAUTH_REDIRECT_URI": "https://custom.example.com/callback"}, clear=True):
            assert get_redirect_uri() == "https://custom.example.com/callback"

    def test_env_takes_priority_over_url(self):
        with patch.dict(os.environ, {"OAUTH_REDIRECT_URI": "https://custom.example.com/callback"}, clear=True):
            assert get_redirect_uri("mcp.example.com") == "https://custom.example.com/callback"

    def test_from_hostname(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_redirect_uri("mcp.example.com")
            assert result == (
                "https://mcp.example.com/api/v1/namespaces/cattle-ai-agent-system"
                "/services/http:rancher-ai-agent:80/proxy/oauth/callback"
            )

    def test_none_url_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_redirect_uri()
            assert result == (
                "https://None/api/v1/namespaces/cattle-ai-agent-system"
                "/services/http:rancher-ai-agent:80/proxy/oauth/callback"
            )


def _make_agent_cfg(name="test-agent", mcp_url="https://mcp.example.com", authentication_secret=None):
    cfg = MagicMock()
    cfg.name = name
    cfg.mcp_url = mcp_url
    cfg.authentication_secret = authentication_secret
    return cfg


def _make_websocket(cookies=None, hostname="rancher.example.com"):
    ws = AsyncMock()
    ws.cookies = dict(cookies or {})
    ws.url = MagicMock()
    ws.url.hostname = hostname
    return ws


class TestTryRefreshOauthToken:
    @pytest.mark.asyncio
    async def test_returns_true_and_injects_when_refresh_token_exists(self):
        ws = _make_websocket(cookies={"mcp_oauth_rt_test-agent": "refresh-tok", "R_SESS": "sess"})
        ws.receive_text.return_value = "ok"
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler._inject_oauth_cookie", new_callable=AsyncMock) as mock_inject:
            result = await _try_refresh_oauth_token(cfg, ws)

        assert result is True
        ws.send_text.assert_called_once()
        assert "<token-refreshed>" in ws.send_text.call_args[0][0]
        mock_inject.assert_awaited_once_with(cfg, ws)

    @pytest.mark.asyncio
    async def test_returns_true_without_inject_when_response_not_ok(self):
        ws = _make_websocket(cookies={"mcp_oauth_rt_test-agent": "refresh-tok", "R_SESS": "sess"})
        ws.receive_text.return_value = "error"
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler._inject_oauth_cookie", new_callable=AsyncMock) as mock_inject:
            result = await _try_refresh_oauth_token(cfg, ws)

        assert result is True
        mock_inject.assert_not_awaited()

class TestInitiateOauthFlow:
    @pytest.mark.asyncio
    async def test_returns_true_when_refresh_succeeds(self):
        ws = _make_websocket()
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=True):
            result = await _initiate_oauth_flow(cfg, ws)

        assert result is True

    @pytest.mark.asyncio
    async def test_credentials_from_secret(self):
        ws = _make_websocket(cookies={"R_SESS": "sess"})
        cfg = _make_agent_cfg(authentication_secret="my-secret")

        discovery_result = OAuthDiscoveryResult(
            auth_server_metadata=AuthorizationServerMetadata(
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint="https://auth.example.com/token",
            ),
        )
        creds = OAuthClientCredentials(client_id="cid", client_secret="csecret", scopes="read")

        mock_oauth_client = MagicMock()
        mock_oauth_client.get_auth_url = AsyncMock(return_value=("https://auth-url", "verifier", "state123"))

        with (
            patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler.get_oauth_secret_data", return_value=(creds, discovery_result)),
            patch("app.services.oauth2.handler.OAuthClient", return_value=mock_oauth_client) as mock_cls,
        ):
            result = await _initiate_oauth_flow(cfg, ws)

        assert result is False
        mock_cls.assert_called_once_with(client_id="cid", client_secret="csecret", scope="read")
        ws.send_text.assert_called_once()
        assert "<authentication>" in ws.send_text.call_args[0][0]
        assert "https://auth-url" in ws.send_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_raises_when_no_authentication_secret(self):
        ws = _make_websocket()
        cfg = _make_agent_cfg(authentication_secret=None)

        with patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=False):
            with pytest.raises(NoAgentAvailableError, match="no authentication secret"):
                await _initiate_oauth_flow(cfg, ws)

    @pytest.mark.asyncio
    async def test_raises_when_no_client_credentials_in_secret(self):
        ws = _make_websocket()
        cfg = _make_agent_cfg(authentication_secret="my-secret")

        discovery_result = OAuthDiscoveryResult(
            auth_server_metadata=AuthorizationServerMetadata(
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint="https://auth.example.com/token",
            ),
        )
        creds = OAuthClientCredentials(client_id="", client_secret="")

        with (
            patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler.get_oauth_secret_data", return_value=(creds, discovery_result)),
        ):
            with pytest.raises(NoAgentAvailableError, match="clientID and clientSecret"):
                await _initiate_oauth_flow(cfg, ws)

    @pytest.mark.asyncio
    async def test_stores_oauth_state(self):
        ws = _make_websocket(cookies={"R_SESS": "sess-abc"})
        cfg = _make_agent_cfg(authentication_secret="my-secret")

        discovery_result = OAuthDiscoveryResult(
            auth_server_metadata=AuthorizationServerMetadata(
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint="https://auth.example.com/token",
            ),
        )
        creds = OAuthClientCredentials(client_id="cid", client_secret="csecret")
        mock_oauth_client = MagicMock()
        mock_oauth_client.get_auth_url = AsyncMock(return_value=("https://auth-url", "my-verifier", "state-xyz"))

        with (
            patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler.get_oauth_secret_data", return_value=(creds, discovery_result)),
            patch("app.services.oauth2.handler.OAuthClient", return_value=mock_oauth_client),
            patch("app.services.oauth2.handler.oauth_store") as mock_store,
        ):
            await _initiate_oauth_flow(cfg, ws)

        mock_store.set_state.assert_called_once_with(
            "state-xyz",
            {
                "verifier": "my-verifier",
                "oauth_client": mock_oauth_client,
                "token_endpoint": "https://auth.example.com/token",
                "cookie_key": "test-agent",
            },
            "sess-abc",
        )


class TestInjectOauthCookie:
    @pytest.mark.asyncio
    async def test_injects_token_when_found(self):
        ws = _make_websocket(cookies={"R_SESS": "sess"})
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler.oauth_store") as mock_store:
            mock_store.pop_token.return_value = "access-token-value"
            await _inject_oauth_cookie(cfg, ws)

        assert ws.cookies["mcp_oauth_at_test-agent"] == "access-token-value"
        mock_store.pop_token.assert_called_once_with("mcp_oauth_at_test-agent", "sess")

    @pytest.mark.asyncio
    async def test_no_injection_when_token_not_found(self):
        ws = _make_websocket(cookies={"R_SESS": "sess"})
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler.oauth_store") as mock_store:
            mock_store.pop_token.return_value = None
            await _inject_oauth_cookie(cfg, ws)

        assert "mcp_oauth_at_test-agent" not in ws.cookies


class TestHandleOauthAuthentication:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_token_refreshed(self):
        ws = _make_websocket()
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler._initiate_oauth_flow", new_callable=AsyncMock, return_value=True) as mock_flow:
            await handle_oauth_authentication(cfg, ws)

        mock_flow.assert_awaited_once_with(cfg, ws)
        ws.receive_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_waits_and_injects_when_not_refreshed(self):
        ws = _make_websocket()
        cfg = _make_agent_cfg()

        with (
            patch("app.services.oauth2.handler._initiate_oauth_flow", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler._inject_oauth_cookie", new_callable=AsyncMock) as mock_inject,
        ):
            await handle_oauth_authentication(cfg, ws)

        ws.receive_text.assert_awaited_once()
        mock_inject.assert_awaited_once_with(cfg, ws)
