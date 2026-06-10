"""Tests for app.services.oauth2.credentials"""

import base64
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.rest import ApiException

from app.services.oauth2.credentials import (
    AGENT_NAMESPACE,
    _decode_key,
    _read_secret,
    get_oauth_secret_data,
)
from app.services.oauth2.models import OAuthClientCredentials, OAuthSecretError


def _b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("utf-8")


def _make_secret(data: dict | None = None):
    secret = MagicMock()
    secret.data = data
    return secret



class TestReadSecret:
    @patch("app.services.oauth2.credentials._load_kube_config")
    @patch("app.services.oauth2.credentials.client.CoreV1Api")
    def test_returns_secret(self, mock_core_api, mock_kube_config):
        mock_secret = _make_secret({"clientID": _b64("id")})
        mock_core_api.return_value.read_namespaced_secret.return_value = mock_secret

        result = _read_secret("my-secret")

        assert result is mock_secret
        mock_core_api.return_value.read_namespaced_secret.assert_called_once_with(
            "my-secret", AGENT_NAMESPACE
        )

    @patch("app.services.oauth2.credentials._load_kube_config")
    @patch("app.services.oauth2.credentials.client.CoreV1Api")
    def test_raises_oauth_secret_error_on_404(self, mock_core_api, mock_kube_config):
        mock_core_api.return_value.read_namespaced_secret.side_effect = ApiException(
            status=404, reason="Not Found"
        )

        with pytest.raises(OAuthSecretError, match="not found"):
            _read_secret("missing-secret")

    @patch("app.services.oauth2.credentials._load_kube_config")
    @patch("app.services.oauth2.credentials.client.CoreV1Api")
    def test_reraises_non_404_api_exception(self, mock_core_api, mock_kube_config):
        mock_core_api.return_value.read_namespaced_secret.side_effect = ApiException(
            status=403, reason="Forbidden"
        )

        with pytest.raises(ApiException):
            _read_secret("forbidden-secret")


class TestGetOAuthSecretData:
    @patch("app.services.oauth2.credentials._read_secret")
    def test_returns_credentials_with_metadata_endpoint(self, mock_read_secret):
        mock_read_secret.return_value = _make_secret(
            {
                "clientID": _b64("cid"),
                "clientSecret": _b64("csecret"),
                "scope": _b64("openid"),
                "metadata_endpoint": _b64("https://auth.example.com/.well-known/openid-configuration"),
            }
        )

        result = get_oauth_secret_data("test-secret")

        assert result.client_id == "cid"
        assert result.client_secret == "csecret"
        assert result.scope == "openid"
        assert result.metadata_endpoint == "https://auth.example.com/.well-known/openid-configuration"

    @patch("app.services.oauth2.credentials._read_secret")
    def test_raises_when_secret_has_no_data(self, mock_read_secret):
        mock_read_secret.return_value = _make_secret(data=None)

        with pytest.raises(OAuthSecretError, match="does not have data"):
            get_oauth_secret_data("empty-secret")
