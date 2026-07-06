"""Kubernetes-based OAuth client credential retrieval."""

import base64
import json
import logging

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .models import OAuthClientCredentials, OAuthSecretError

AGENT_NAMESPACE = "cattle-ai-agent-system"

logger = logging.getLogger(__name__)


def _load_kube_config():
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def _read_secret(secret_name: str) -> client.V1Secret:
    _load_kube_config()
    v1 = client.CoreV1Api()
    try:
        return v1.read_namespaced_secret(secret_name, AGENT_NAMESPACE)
    except ApiException as e:
        if e.status == 404:
            raise OAuthSecretError(
                f"OAuth secret '{secret_name}' not found in namespace '{AGENT_NAMESPACE}'."
            ) from e
        raise


def _decode_key(data: dict, key: str, default: str = "") -> str:
    if key in data:
        return base64.b64decode(data[key]).decode("utf-8").strip()
    return default



def get_oauth_secret_data(secret_name: str) -> OAuthClientCredentials:
    """
    Read both client credentials and discovery metadata from a Kubernetes secret.
    Args:
        secret_name: Name of the secret in the agent namespace.
    Returns:
        OAuthClientCredentials with client_id, client_secret, scope, and metadata_endpoint.
    Raises:
        OAuthSecretError: If the secret does not exist or has no metadata key.
    """
    secret = _read_secret(secret_name)

    if not secret.data:
        raise OAuthSecretError(
            f"OAuth secret '{secret_name}' does not have data."
        )

    return OAuthClientCredentials(
        client_id=_decode_key(secret.data, "clientID"),
        client_secret=_decode_key(secret.data, "clientSecret"),
        scope=_decode_key(secret.data, "scope"),
        metadata_endpoint=_decode_key(secret.data, "metadataEndpoint"),
    )


