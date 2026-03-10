"""
Unit tests for the agent loader module.

Tests agent configuration loading and secret retrieval.
"""
import pytest
import base64
from unittest.mock import MagicMock, patch, call
from kubernetes.client.rest import ApiException

from app.services.agent.loader import (
    _get_default_ai_agent_config_crds,
    _update_default_ai_agent_config_crds,
    ensure_default_ai_agent_config_crds,
    NAMESPACE,
    CRD_GROUP,
    CRD_VERSION,
    CRD_PLURAL,
)

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
