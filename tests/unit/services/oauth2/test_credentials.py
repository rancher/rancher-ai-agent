"""Tests for app.services.oauth2.credentials"""

import base64

import pytest
from unittest.mock import patch, MagicMock

from app.services.oauth2.credentials import AGENT_NAMESPACE, get_oauth_client_credentials
from app.services.oauth2.models import OAuthClientCredentials


class TestGetOAuthClientCredentials:
    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_success_with_all_fields(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_secret = MagicMock()
        mock_secret.data = {
            "clientID": base64.b64encode(b"my-client-id").decode(),
            "clientSecret": base64.b64encode(b"my-client-secret").decode(),
            "scopes": base64.b64encode(b"read write").decode(),
        }
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = mock_secret
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        creds = get_oauth_client_credentials("my-secret")
        assert creds.client_id == "my-client-id"
        assert creds.client_secret == "my-client-secret"
        assert creds.scopes == "read write"
        mock_v1.read_namespaced_secret.assert_called_once_with("my-secret", "cattle-ai-agent-system")

    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_success_client_id_only(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_secret = MagicMock()
        mock_secret.data = {
            "clientID": base64.b64encode(b"my-client-id").decode(),
        }
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = mock_secret
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        creds = get_oauth_client_credentials("my-secret")
        assert creds.client_id == "my-client-id"
        assert creds.client_secret == ""
        assert creds.scopes == ""

    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_empty_secret_raises(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_secret = MagicMock()
        mock_secret.data = None
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = mock_secret
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        with pytest.raises(RuntimeError, match="is empty"):
            get_oauth_client_credentials("my-secret")

    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_falls_back_to_kube_config(self, mock_k8s_client, mock_k8s_config):
        from kubernetes.config import ConfigException
        mock_k8s_config.load_incluster_config.side_effect = ConfigException("not in cluster")
        mock_k8s_config.ConfigException = ConfigException
        mock_k8s_config.load_kube_config.return_value = None
        mock_secret = MagicMock()
        mock_secret.data = {
            "clientID": base64.b64encode(b"local-id").decode(),
        }
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = mock_secret
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        creds = get_oauth_client_credentials("my-secret")
        assert creds.client_id == "local-id"
        mock_k8s_config.load_kube_config.assert_called_once()

    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_scopes_stripped(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_secret = MagicMock()
        mock_secret.data = {
            "clientID": base64.b64encode(b"id").decode(),
            "scopes": base64.b64encode(b"  read write  ").decode(),
        }
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = mock_secret
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        creds = get_oauth_client_credentials("my-secret")
        assert creds.scopes == "read write"
