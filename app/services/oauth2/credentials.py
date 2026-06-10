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


def get_oauth_client_credentials(secret_name: str) -> OAuthClientCredentials:
    """
    Retrieve OAuth2 client credentials from a Kubernetes secret.
    The secret must contain a 'clientId' key (or legacy 'client_id') and
    optionally a 'clientSecret' key (or legacy 'client_secret') and
    a 'scopes' key containing a space-separated string of OAuth scopes.
    Key names follow the AIAgentConfig CRD oauthSecret convention:
        clientId, clientSecret, scopes
    Args:
        secret_name: Name of the secret in the agent namespace.
    Returns:
        OAuthClientCredentials with client_id, client_secret, and scopes.
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
    )


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
        metadata_endpoint=_decode_key(secret.data, "metadata_endpoint"),
    )



def create_oauth_secret(secret_name: str) -> None:
    """
    Create or update a Kubernetes secret with OAuth discovery metadata.
    Args:
        secret_name: Name for the secret.
        metadata: The discovery result to persist.
    """
    _load_kube_config()
    v1 = client.CoreV1Api()

    secret_data = {
       # "metadata": base64.b64encode(metadata_json.encode("utf-8")).decode("utf-8"),
    }

    try:
        existing = v1.read_namespaced_secret(secret_name, AGENT_NAMESPACE)
        if existing.data:
            existing.data["metadata"] = secret_data["metadata"]
        else:
            existing.data = secret_data
        v1.patch_namespaced_secret(secret_name, AGENT_NAMESPACE, existing)
        logger.info(f"Updated OAuth secret '{secret_name}' with metadata")
    except ApiException as e:
        if e.status != 404:
            raise
        secret = client.V1Secret(
            metadata=client.V1ObjectMeta(name=secret_name, namespace=AGENT_NAMESPACE),
            data=secret_data,
            type="Opaque",
        )
        v1.create_namespaced_secret(AGENT_NAMESPACE, secret)
        logger.info(f"Created OAuth secret '{secret_name}' with metadata")


def update_oauth_secret_credentials(
    secret_name: str, client_id: str, client_secret: str, scopes: str
) -> None:
    """
    Update an existing Kubernetes secret with OAuth client credentials.
    Args:
        secret_name: Name of the existing secret.
        client_id: The registered client ID.
        client_secret: The registered client secret.
        scopes: Space-separated scopes.
    Raises:
        OAuthSecretError: If the secret does not exist.
    """
    secret = _read_secret(secret_name)

    patch_data = {
        "clientID": base64.b64encode(client_id.encode("utf-8")).decode("utf-8"),
        "clientSecret": base64.b64encode(client_secret.encode("utf-8")).decode("utf-8"),
    }
    if scopes:
        patch_data["scopes"] = base64.b64encode(scopes.encode("utf-8")).decode("utf-8")

    if secret.data:
        secret.data.update(patch_data)
    else:
        secret.data = patch_data

    _load_kube_config()
    v1 = client.CoreV1Api()
    v1.patch_namespaced_secret(secret_name, AGENT_NAMESPACE, secret)
    logger.info(f"Updated OAuth secret '{secret_name}' with client credentials")
