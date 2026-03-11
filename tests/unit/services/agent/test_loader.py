"""
Unit tests for the agent loader module.

Tests agent configuration loading and secret retrieval.
"""
import pytest
import base64
from unittest.mock import MagicMock, patch
from kubernetes.client.rest import ApiException

from app.services.agent.loader import (
    _get_default_ai_agent_config_crds,
    _update_default_ai_agent_config_crds,
    NAMESPACE,
)

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
    
def test_update_creates_missing_builtin_crd():
    """Verify _update_default_ai_agent_config_crds creates a built-in CRD that is missing from the cluster."""
    mock_api = MagicMock()

    # Only one existing CRD — the others are missing
    existing_items = [
        {
            "metadata": {"name": "rancher"},
            "spec": _get_default_ai_agent_config_crds()[0]["spec"],
        }
    ]

    _update_default_ai_agent_config_crds(mock_api, existing_items)

    # Should have created the missing fleet and provisioning CRDs
    assert mock_api.create_namespaced_custom_object.call_count == 2
    created_names = [
        c.kwargs["body"]["metadata"]["name"]
        for c in mock_api.create_namespaced_custom_object.call_args_list
    ]
    assert "fleet" in created_names
    assert "provisioning" in created_names


def test_update_patches_changed_builtin_crd():
    """Verify _update_default_ai_agent_config_crds patches a built-in CRD whose spec has drifted."""
    mock_api = MagicMock()

    defaults = _get_default_ai_agent_config_crds()
    # Build existing items from defaults but change the rancher description
    existing_items = []
    for d in defaults:
        item = {
            "metadata": {"name": d["metadata"]["name"]},
            "spec": {**d["spec"]},
        }
        existing_items.append(item)

    # Mutate the rancher description to simulate drift
    existing_items[0]["spec"]["description"] = "outdated description"

    _update_default_ai_agent_config_crds(mock_api, existing_items)

    # Should patch only the drifted rancher CRD
    mock_api.patch_namespaced_custom_object.assert_called_once()
    patch_call = mock_api.patch_namespaced_custom_object.call_args
    assert patch_call.kwargs["name"] == "rancher"
    assert mock_api.create_namespaced_custom_object.call_count == 0


def test_update_skips_up_to_date_builtin_crds():
    """Verify _update_default_ai_agent_config_crds does nothing when all built-in CRDs match."""
    mock_api = MagicMock()

    defaults = _get_default_ai_agent_config_crds()
    existing_items = [
        {
            "metadata": {"name": d["metadata"]["name"]},
            "spec": {**d["spec"]},
        }
        for d in defaults
    ]

    _update_default_ai_agent_config_crds(mock_api, existing_items)

    mock_api.create_namespaced_custom_object.assert_not_called()
    mock_api.patch_namespaced_custom_object.assert_not_called()


def test_update_skips_non_builtin_crds():
    """Verify _update_default_ai_agent_config_crds does not update a CRD that is not marked builtIn."""
    mock_api = MagicMock()

    defaults = _get_default_ai_agent_config_crds()
    existing_items = []
    for d in defaults:
        spec = {**d["spec"]}
        # Mark rancher as not built-in and change its description
        if d["metadata"]["name"] == "rancher":
            spec["builtIn"] = False
            spec["description"] = "user-customized"
        existing_items.append({
            "metadata": {"name": d["metadata"]["name"]},
            "spec": spec,
        })

    _update_default_ai_agent_config_crds(mock_api, existing_items)

    # rancher should NOT be patched (not built-in), others are up-to-date
    mock_api.patch_namespaced_custom_object.assert_not_called()
    mock_api.create_namespaced_custom_object.assert_not_called()


def test_update_patches_multiple_drifted_crds():
    """Verify _update_default_ai_agent_config_crds patches all built-in CRDs that have drifted."""
    mock_api = MagicMock()

    defaults = _get_default_ai_agent_config_crds()
    existing_items = []
    for d in defaults:
        item = {
            "metadata": {"name": d["metadata"]["name"]},
            "spec": {**d["spec"], "description": "stale"},
        }
        existing_items.append(item)

    _update_default_ai_agent_config_crds(mock_api, existing_items)

    # All three built-in CRDs should be patched
    assert mock_api.patch_namespaced_custom_object.call_count == len(defaults)
    patched_names = sorted(
        c.kwargs["name"]
        for c in mock_api.patch_namespaced_custom_object.call_args_list
    )
    expected_names = sorted(d["metadata"]["name"] for d in defaults)
    assert patched_names == expected_names
    mock_api.create_namespaced_custom_object.assert_not_called()


def test_update_detects_drift_in_human_validation_tools():
    """Verify a change in humanValidationTools triggers a patch."""
    mock_api = MagicMock()

    defaults = _get_default_ai_agent_config_crds()
    existing_items = []
    for d in defaults:
        spec = {**d["spec"]}
        # Remove a tool from rancher's validation list
        if d["metadata"]["name"] == "rancher":
            spec["humanValidationTools"] = ["createKubernetesResource"]
        existing_items.append({
            "metadata": {"name": d["metadata"]["name"]},
            "spec": spec,
        })

    _update_default_ai_agent_config_crds(mock_api, existing_items)

    # Only rancher should be patched
    mock_api.patch_namespaced_custom_object.assert_called_once()
    assert mock_api.patch_namespaced_custom_object.call_args.kwargs["name"] == "rancher"


def test_update_detects_drift_in_system_prompt():
    """Verify a change in systemPrompt triggers a patch."""
    mock_api = MagicMock()

    defaults = _get_default_ai_agent_config_crds()
    existing_items = []
    for d in defaults:
        spec = {**d["spec"]}
        if d["metadata"]["name"] == "fleet":
            spec["systemPrompt"] = "old prompt"
        existing_items.append({
            "metadata": {"name": d["metadata"]["name"]},
            "spec": spec,
        })

    _update_default_ai_agent_config_crds(mock_api, existing_items)

    mock_api.patch_namespaced_custom_object.assert_called_once()
    assert mock_api.patch_namespaced_custom_object.call_args.kwargs["name"] == "fleet"


def test_update_ignores_enabled_field_change():
    """Verify that a change to spec.enabled does not trigger a patch."""
    mock_api = MagicMock()

    defaults = _get_default_ai_agent_config_crds()
    existing_items = []
    for d in defaults:
        spec = {**d["spec"]}
        # Simulate user disabling all agents
        spec["enabled"] = False
        existing_items.append({
            "metadata": {"name": d["metadata"]["name"]},
            "spec": spec,
        })

    _update_default_ai_agent_config_crds(mock_api, existing_items)

    # enabled change must not cause any patch or create
    mock_api.patch_namespaced_custom_object.assert_not_called()
    mock_api.create_namespaced_custom_object.assert_not_called()
