"""Tests for app.services.oauth2.credentials"""

import base64
import json

import pytest
from unittest.mock import patch, MagicMock

from kubernetes.client.rest import ApiException

from app.services.oauth2.credentials import (
    AGENT_NAMESPACE,
    get_oauth_client_credentials,
    get_oauth_secret_data,
    create_oauth_secret,
    update_oauth_secret_credentials,
)
from app.services.oauth2.models import AuthorizationServerMetadata, OAuthClientCredentials, OAuthDiscoveryResult, OAuthSecretError


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

        with pytest.raises(OAuthSecretError, match="does not have data"):
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


def _make_metadata_dict():
    return {
        "auth_server_metadata": {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "registration_endpoint": "https://auth.example.com/register",
            "scopes_supported": ["read", "write"],
        },
        "required_scopes": [],
    }


def _make_secret_with_metadata(client_id="cid", client_secret="csecret", scopes="read", metadata_dict=None):
    metadata_dict = metadata_dict or _make_metadata_dict()
    mock_secret = MagicMock()
    mock_secret.data = {
        "clientID": base64.b64encode(client_id.encode()).decode(),
        "clientSecret": base64.b64encode(client_secret.encode()).decode(),
        "scopes": base64.b64encode(scopes.encode()).decode(),
        "metadata": base64.b64encode(json.dumps(metadata_dict).encode()).decode(),
    }
    return mock_secret


class TestGetOAuthSecretData:
    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_success_with_all_fields(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = _make_secret_with_metadata()
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        creds, metadata = get_oauth_secret_data("my-secret")
        assert creds.client_id == "cid"
        assert creds.client_secret == "csecret"
        assert metadata.auth_server_metadata.authorization_endpoint == "https://auth.example.com/authorize"
        assert metadata.auth_server_metadata.token_endpoint == "https://auth.example.com/token"

    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_missing_metadata_key_raises(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_secret = MagicMock()
        mock_secret.data = {
            "clientID": base64.b64encode(b"cid").decode(),
        }
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = mock_secret
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        with pytest.raises(OAuthSecretError, match="metadata"):
            get_oauth_secret_data("my-secret")

    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_invalid_metadata_json_raises(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_secret = MagicMock()
        mock_secret.data = {
            "metadata": base64.b64encode(b"not-json").decode(),
        }
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = mock_secret
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        with pytest.raises(OAuthSecretError, match="invalid metadata"):
            get_oauth_secret_data("my-secret")

    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_secret_not_found_raises(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.side_effect = ApiException(status=404)
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        with pytest.raises(OAuthSecretError, match="not found"):
            get_oauth_secret_data("missing-secret")


class TestCreateOAuthSecret:
    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_creates_new_secret(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.side_effect = ApiException(status=404)
        mock_k8s_client.CoreV1Api.return_value = mock_v1
        mock_k8s_client.V1Secret.return_value = MagicMock()
        mock_k8s_client.V1ObjectMeta.return_value = MagicMock()

        metadata = OAuthDiscoveryResult(
            auth_server_metadata=AuthorizationServerMetadata(
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint="https://auth.example.com/token",
            ),
        )
        create_oauth_secret("test-secret", metadata)
        mock_v1.create_namespaced_secret.assert_called_once()

    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_updates_existing_secret(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        existing_secret = MagicMock()
        existing_secret.data = {"clientID": base64.b64encode(b"existing").decode()}
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = existing_secret
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        metadata = OAuthDiscoveryResult(
            auth_server_metadata=AuthorizationServerMetadata(
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint="https://auth.example.com/token",
            ),
        )
        create_oauth_secret("test-secret", metadata)
        mock_v1.patch_namespaced_secret.assert_called_once()
        mock_v1.create_namespaced_secret.assert_not_called()


class TestUpdateOAuthSecretCredentials:
    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_patches_secret_with_credentials(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        existing_secret = MagicMock()
        existing_secret.data = {"metadata": base64.b64encode(b"{}").decode()}
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.return_value = existing_secret
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        update_oauth_secret_credentials("test-secret", "new-id", "new-secret", "read write")
        mock_v1.patch_namespaced_secret.assert_called_once()
        patched_data = existing_secret.data
        assert base64.b64decode(patched_data["clientID"]).decode() == "new-id"
        assert base64.b64decode(patched_data["clientSecret"]).decode() == "new-secret"
        assert base64.b64decode(patched_data["scopes"]).decode() == "read write"

    @patch("app.services.oauth2.credentials.config")
    @patch("app.services.oauth2.credentials.client")
    def test_not_found_raises(self, mock_k8s_client, mock_k8s_config):
        mock_k8s_config.load_incluster_config.return_value = None
        mock_v1 = MagicMock()
        mock_v1.read_namespaced_secret.side_effect = ApiException(status=404)
        mock_k8s_client.CoreV1Api.return_value = mock_v1

        with pytest.raises(OAuthSecretError, match="not found"):
            update_oauth_secret_credentials("missing", "id", "secret", "read")
