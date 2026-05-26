"""Tests for app.services.oauth2.utils"""

import os
from unittest.mock import MagicMock, patch

from app.services.oauth2.utils import (
    AGENT_NAMESPACE,
    _get_tls_verify,
    get_redirect_uri,
)


def test_agent_namespace_value():
    assert AGENT_NAMESPACE == "cattle-ai-agent-system"


class TestGetTlsVerify:
    def test_defaults_to_true(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _get_tls_verify() is True

    def test_insecure_skip_tls_true(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "true"}, clear=True):
            assert _get_tls_verify() is False

    def test_insecure_skip_tls_true_uppercase(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "TRUE"}, clear=True):
            assert _get_tls_verify() is False

    def test_insecure_skip_tls_false(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "false"}, clear=True):
            assert _get_tls_verify() is True

    def test_insecure_skip_tls_other_value(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "yes"}, clear=True):
            assert _get_tls_verify() is True


class TestGetRedirectUri:
    def test_from_env_variable(self):
        with patch.dict(os.environ, {"OAUTH_REDIRECT_URI": "https://custom.example.com/callback"}, clear=True):
            assert get_redirect_uri() == "https://custom.example.com/callback"

    def test_env_takes_priority_over_websocket(self):
        ws = MagicMock()
        with patch.dict(os.environ, {"OAUTH_REDIRECT_URI": "https://custom.example.com/callback"}, clear=True):
            assert get_redirect_uri(websocket=ws) == "https://custom.example.com/callback"

    def test_from_wss_websocket(self):
        ws = MagicMock()
        ws.url.scheme = "wss"
        ws.url.hostname = "mcp.example.com"
        ws.url.port = None
        with patch.dict(os.environ, {}, clear=True):
            assert get_redirect_uri(websocket=ws) == "https://mcp.example.com/oauth/callback"

    def test_from_ws_websocket_with_port(self):
        ws = MagicMock()
        ws.url.scheme = "ws"
        ws.url.hostname = "localhost"
        ws.url.port = 8080
        with patch.dict(os.environ, {}, clear=True):
            assert get_redirect_uri(websocket=ws) == "http://localhost:8080/oauth/callback"

    def test_from_wss_websocket_standard_port(self):
        ws = MagicMock()
        ws.url.scheme = "wss"
        ws.url.hostname = "mcp.example.com"
        ws.url.port = 443
        with patch.dict(os.environ, {}, clear=True):
            assert get_redirect_uri(websocket=ws) == "https://mcp.example.com/oauth/callback"

    def test_default_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_redirect_uri() == "http://localhost:8000/oauth/callback"
