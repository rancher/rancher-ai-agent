"""
Unit tests for the agent loader module.

Tests agent configuration loading and secret retrieval.
"""
import pytest
import base64
from unittest.mock import MagicMock, patch
from kubernetes.client.rest import ApiException

from app.services.agent.loader import get_basic_auth_credentials, NAMESPACE


# ============================================================================
# get_basic_auth_credentials Tests
# ============================================================================

@patch('app.services.agent.loader.config')
@patch('app.services.agent.loader.client')
def test_get_basic_auth_credentials_success(mock_k8s_client, mock_config):
    """Verify get_basic_auth_credentials successfully retrieves and encodes credentials."""
    # Setup
    secret_name = "test-secret"
    username = "testuser"
    password = "testpass"
    
    # Mock Kubernetes client
    mock_v1 = MagicMock()
    mock_k8s_client.CoreV1Api.return_value = mock_v1
    
    # Create mock secret with base64-encoded username and password
    mock_secret = MagicMock()
    mock_secret.data = {
        'username': base64.b64encode(username.encode('utf-8')).decode('utf-8'),
        'password': base64.b64encode(password.encode('utf-8')).decode('utf-8')
    }
    mock_v1.read_namespaced_secret.return_value = mock_secret
    
    # Execute
    result = get_basic_auth_credentials(secret_name)
    
    # Verify
    expected_credentials = base64.b64encode(f"{username}:{password}".encode('utf-8')).decode('utf-8')
    assert result == expected_credentials
    mock_v1.read_namespaced_secret.assert_called_once_with(secret_name, NAMESPACE)


@patch('app.services.agent.loader.config')
@patch('app.services.agent.loader.client')
def test_get_basic_auth_credentials_empty_secret(mock_k8s_client, mock_config):
    """Verify get_basic_auth_credentials raises RuntimeError for empty secret."""
    # Setup
    secret_name = "empty-secret"
    
    # Mock Kubernetes client
    mock_v1 = MagicMock()
    mock_k8s_client.CoreV1Api.return_value = mock_v1
    
    # Create mock empty secret
    mock_secret = MagicMock()
    mock_secret.data = None
    mock_v1.read_namespaced_secret.return_value = mock_secret
    
    # Execute & Verify
    with pytest.raises(RuntimeError) as exc_info:
        get_basic_auth_credentials(secret_name)
    
    assert f"Authentication secret '{secret_name}'" in str(exc_info.value)
    assert "is empty" in str(exc_info.value)


@patch('app.services.agent.loader.config')
@patch('app.services.agent.loader.client')
def test_get_basic_auth_credentials_missing_username(mock_k8s_client, mock_config):
    """Verify get_basic_auth_credentials raises RuntimeError when username key is missing."""
    # Setup
    secret_name = "incomplete-secret"
    password = "testpass"
    
    # Mock Kubernetes client
    mock_v1 = MagicMock()
    mock_k8s_client.CoreV1Api.return_value = mock_v1
    
    # Create mock secret missing username
    mock_secret = MagicMock()
    mock_secret.data = {
        'password': base64.b64encode(password.encode('utf-8')).decode('utf-8')
    }
    mock_v1.read_namespaced_secret.return_value = mock_secret
    
    # Execute & Verify
    with pytest.raises(RuntimeError) as exc_info:
        get_basic_auth_credentials(secret_name)
    
    assert f"Authentication secret '{secret_name}'" in str(exc_info.value)
    assert "does not contain 'username' and 'password' keys" in str(exc_info.value)


@patch('app.services.agent.loader.config')
@patch('app.services.agent.loader.client')
def test_get_basic_auth_credentials_missing_password(mock_k8s_client, mock_config):
    """Verify get_basic_auth_credentials raises RuntimeError when password key is missing."""
    # Setup
    secret_name = "incomplete-secret"
    username = "testuser"
    
    # Mock Kubernetes client
    mock_v1 = MagicMock()
    mock_k8s_client.CoreV1Api.return_value = mock_v1
    
    # Create mock secret missing password
    mock_secret = MagicMock()
    mock_secret.data = {
        'username': base64.b64encode(username.encode('utf-8')).decode('utf-8')
    }
    mock_v1.read_namespaced_secret.return_value = mock_secret
    
    # Execute & Verify
    with pytest.raises(RuntimeError) as exc_info:
        get_basic_auth_credentials(secret_name)
    
    assert f"Authentication secret '{secret_name}'" in str(exc_info.value)
    assert "does not contain 'username' and 'password' keys" in str(exc_info.value)


@patch('app.services.agent.loader.config')
@patch('app.services.agent.loader.client')
def test_get_basic_auth_credentials_api_exception(mock_k8s_client, mock_config):
    """Verify get_basic_auth_credentials propagates ApiException from Kubernetes."""
    # Setup
    secret_name = "nonexistent-secret"
    
    # Mock Kubernetes client
    mock_v1 = MagicMock()
    mock_k8s_client.CoreV1Api.return_value = mock_v1
    
    # Mock API exception (e.g., secret not found)
    mock_v1.read_namespaced_secret.side_effect = ApiException(status=404, reason="Not Found")
    
    # Execute & Verify
    with pytest.raises(ApiException) as exc_info:
        get_basic_auth_credentials(secret_name)
    
    assert exc_info.value.status == 404

