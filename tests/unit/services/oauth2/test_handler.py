"""Tests for app.services.oauth2.handler"""

import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.agent._constants import NoAgentAvailableError
from app.services.oauth2.models import OAuth2Canceled
from app.services.oauth2.handler import (
    _inject_oauth_cookie,
    _initiate_oauth_flow,
    _try_refresh_oauth_token,
    get_redirect_uri,
    handle_oauth_authentication,
)
from app.services.oauth2.client import OAuthClientManager


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
                "/services/http:rancher-ai-agent:80/proxy/v1/api/oauth2/callback"
            )

    def test_none_url_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_redirect_uri()
            assert result == (
                "https://None/api/v1/namespaces/cattle-ai-agent-system"
                "/services/http:rancher-ai-agent:80/proxy/v1/api/oauth2/callback"
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
        ws.receive_text.return_value = "token_refresh_confirmed"
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler._inject_oauth_cookie", new_callable=AsyncMock) as mock_inject:
            result = await _try_refresh_oauth_token(cfg.name, ws)

        assert result is True
        ws.send_text.assert_called_once()
        assert "<token-refresh>" in ws.send_text.call_args[0][0]
        mock_inject.assert_awaited_once_with(cfg.name, ws)

    @pytest.mark.asyncio
    async def test_returns_true_without_inject_when_response_not_ok(self):
        ws = _make_websocket(cookies={"mcp_oauth_rt_test-agent": "refresh-tok", "R_SESS": "sess"})
        ws.receive_text.return_value = "error"
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler._inject_oauth_cookie", new_callable=AsyncMock) as mock_inject:
            result = await _try_refresh_oauth_token(cfg, ws)

        assert result is False
        mock_inject.assert_not_awaited()

class TestInitiateOauthFlow:
    def setup_method(self):
        """Reset singleton between tests."""
        OAuthClientManager._instance = None

    @pytest.mark.asyncio
    async def test_returns_true_when_refresh_succeeds(self):
        ws = _make_websocket()
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=True):
            result = await _initiate_oauth_flow(cfg.name, ws)

        assert result is True

    @pytest.mark.asyncio
    async def test_uses_registered_client(self):
        ws = _make_websocket(cookies={"R_SESS": "sess"})
        cfg = _make_agent_cfg(authentication_secret="my-secret")

        mock_client = MagicMock()
        mock_client.create_authorization_url = AsyncMock(return_value={
            "url": "https://auth-url",
            "state": "state123",
            "code_verifier": "verifier123",
        })

        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_client

        with (
            patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler.OAuthClientManager.get_instance", return_value=mock_manager),
        ):
            result = await _initiate_oauth_flow(cfg.name, ws)

        assert result is False
        mock_manager.get_client.assert_called_once_with("test-agent")
        ws.send_text.assert_called_once()
        sent = ws.send_text.call_args[0][0]
        assert "<authentication>" in sent
        assert "<message>" not in sent
        assert "https://auth-url" in ws.send_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_raises_when_no_client_registered(self):
        ws = _make_websocket()
        cfg = _make_agent_cfg(authentication_secret="my-secret")

        mock_manager = MagicMock()
        mock_manager.get_client.return_value = None

        with (
            patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler.OAuthClientManager.get_instance", return_value=mock_manager),
        ):
            with pytest.raises(NoAgentAvailableError, match="no OAuth client is registered"):
                await _initiate_oauth_flow(cfg.name, ws)

    @pytest.mark.asyncio
    async def test_stores_oauth_state(self):
        ws = _make_websocket(cookies={"R_SESS": "sess-abc"})
        cfg = _make_agent_cfg(authentication_secret="my-secret")

        mock_client = MagicMock()
        mock_client.create_authorization_url = AsyncMock(return_value={
            "url": "https://auth-url",
            "state": "state-xyz",
            "code_verifier": "my-verifier",
        })

        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_client

        with (
            patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler.OAuthClientManager.get_instance", return_value=mock_manager),
            patch("app.services.oauth2.handler.oauth_store") as mock_store,
        ):
            await _initiate_oauth_flow(cfg.name, ws)

        mock_store.set_state.assert_called_once_with(
            "state-xyz",
            {
                "code_verifier": "my-verifier",
                "agent_name": "test-agent",
                "redirect_uri": mock_store.set_state.call_args[0][1]["redirect_uri"],
            },
            "sess-abc",
        )

    @pytest.mark.asyncio
    async def test_wraps_message_in_message_tag_when_add_message_tag_true(self):
        ws = _make_websocket(cookies={"R_SESS": "sess"})
        cfg = _make_agent_cfg(authentication_secret="my-secret")

        mock_client = MagicMock()
        mock_client.create_authorization_url = AsyncMock(return_value={
            "url": "https://auth-url",
            "state": "state123",
            "code_verifier": "verifier123",
        })

        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_client

        with (
            patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler.OAuthClientManager.get_instance", return_value=mock_manager),
            patch("app.services.oauth2.handler.oauth_store"),
        ):
            await _initiate_oauth_flow(cfg.name, ws)

        sent = ws.send_text.call_args[0][0]        
        assert sent.startswith("<authentication>") and sent.endswith("</authentication>")
        assert '{"type": "oauth2", "url": "https://auth-url", "agent": "test-agent"}' in sent

    @pytest.mark.asyncio
    async def test_no_message_tag_when_add_message_tag_false(self):
        ws = _make_websocket(cookies={"R_SESS": "sess"})
        cfg = _make_agent_cfg(authentication_secret="my-secret")

        mock_client = MagicMock()
        mock_client.create_authorization_url = AsyncMock(return_value={
            "url": "https://auth-url",
            "state": "state456",
            "code_verifier": "verifier456",
        })

        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_client

        with (
            patch("app.services.oauth2.handler._try_refresh_oauth_token", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler.OAuthClientManager.get_instance", return_value=mock_manager),
            patch("app.services.oauth2.handler.oauth_store"),
        ):
            await _initiate_oauth_flow(cfg.name, ws)

        sent = ws.send_text.call_args[0][0]
        assert "<message>" not in sent
        assert sent.startswith("<authentication>") and sent.endswith("</authentication>")


class TestInjectOauthCookie:
    @pytest.mark.asyncio
    async def test_injects_token_when_found(self):
        ws = _make_websocket(cookies={"R_SESS": "sess"})
        cfg = _make_agent_cfg()

        with patch("app.services.oauth2.handler.oauth_store") as mock_store:
            mock_store.pop_token.return_value = "access-token-value"
            await _inject_oauth_cookie(cfg.name, ws)

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
            await handle_oauth_authentication(cfg.name, ws)

        mock_flow.assert_awaited_once_with(cfg.name, ws)
        ws.receive_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_waits_and_injects_when_not_refreshed(self):
        ws = _make_websocket()
        ws.receive_text.return_value = "authentication_confirmed"
        cfg = _make_agent_cfg()

        with (
            patch("app.services.oauth2.handler._initiate_oauth_flow", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler._inject_oauth_cookie", new_callable=AsyncMock) as mock_inject,
        ):
            await handle_oauth_authentication(cfg.name, ws)

        ws.receive_text.assert_awaited_once()
        mock_inject.assert_awaited_once_with(cfg.name, ws)

    @pytest.mark.asyncio
    async def test_raises_when_authentication_not_confirmed(self):
        ws = _make_websocket()
        ws.receive_text.return_value = "unexpected_response"
        cfg = _make_agent_cfg()

        with (
            patch("app.services.oauth2.handler._initiate_oauth_flow", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler._inject_oauth_cookie", new_callable=AsyncMock) as mock_inject,
        ):
            with pytest.raises(Exception, match="OAuth2 authentication failed"):
                await handle_oauth_authentication(cfg.name, ws)

        ws.receive_text.assert_awaited_once()
        mock_inject.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_oauth2_cancelled_when_user_cancels(self):
        ws = _make_websocket()
        ws.receive_text.return_value = "authentication_canceled"
        cfg = _make_agent_cfg()

        with (
            patch("app.services.oauth2.handler._initiate_oauth_flow", new_callable=AsyncMock, return_value=False),
            patch("app.services.oauth2.handler._inject_oauth_cookie", new_callable=AsyncMock) as mock_inject,
        ):
            with pytest.raises(OAuth2Canceled):
                await handle_oauth_authentication(cfg.name, ws)

        ws.receive_text.assert_awaited_once()
        mock_inject.assert_not_awaited()
