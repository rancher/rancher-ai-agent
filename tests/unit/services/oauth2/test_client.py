"""Tests for app.services.oauth2.client"""

import os

import pytest
from unittest.mock import patch

from app.services.oauth2.client import OAuthClientManager, get_tls_verify


class TestGetTlsVerify:
    def test_defaults_to_true(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_tls_verify() is True

    def test_insecure_skip_tls_true(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "true"}, clear=True):
            assert get_tls_verify() is False

    def test_insecure_skip_tls_true_uppercase(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "TRUE"}, clear=True):
            assert get_tls_verify() is False

    def test_insecure_skip_tls_false(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "false"}, clear=True):
            assert get_tls_verify() is True

    def test_insecure_skip_tls_other_value(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "yes"}, clear=True):
            assert get_tls_verify() is True


class TestOAuthClientManager:
    def setup_method(self):
        """Reset singleton between tests."""
        OAuthClientManager._instance = None

    def test_singleton(self):
        m1 = OAuthClientManager.get_instance()
        m2 = OAuthClientManager.get_instance()
        assert m1 is m2

    def test_register_client(self):
        manager = OAuthClientManager.get_instance()
        manager.register_client(
            name="test-agent",
            client_id="cid",
            client_secret="csecret",
            scope="openid",
            server_metadata_url="https://auth.example.com/.well-known/openid-configuration",
        )
        assert manager.has_client("test-agent")

    def test_get_client_returns_none_for_unknown(self):
        manager = OAuthClientManager.get_instance()
        assert manager.get_client("nonexistent") is None

    def test_get_client_returns_starlette_app(self):
        manager = OAuthClientManager.get_instance()
        manager.register_client(
            name="test-agent",
            client_id="cid",
            client_secret="csecret",
            scope="openid",
            server_metadata_url="https://auth.example.com/.well-known/openid-configuration",
        )
        client = manager.get_client("test-agent")
        assert client is not None
        assert client.client_id == "cid"
        assert client.client_secret == "csecret"

    def test_register_client_overwrites_existing(self):
        manager = OAuthClientManager.get_instance()
        manager.register_client(
            name="test-agent",
            client_id="old-id",
            client_secret="old-secret",
            scope="openid",
            server_metadata_url="https://auth.example.com/.well-known/openid-configuration",
        )
        manager.register_client(
            name="test-agent",
            client_id="new-id",
            client_secret="new-secret",
            scope="profile",
            server_metadata_url="https://auth2.example.com/.well-known/openid-configuration",
        )
        client = manager.get_client("test-agent")
        assert client.client_id == "new-id"
        assert client.client_secret == "new-secret"

    def test_remove_client(self):
        manager = OAuthClientManager.get_instance()
        manager.register_client(
            name="test-agent",
            client_id="cid",
            client_secret="csecret",
            scope="openid",
            server_metadata_url="https://auth.example.com/.well-known/openid-configuration",
        )
        assert manager.has_client("test-agent")
        manager.remove_client("test-agent")
        assert not manager.has_client("test-agent")

    def test_has_client_false_when_not_registered(self):
        manager = OAuthClientManager.get_instance()
        assert not manager.has_client("unknown")
