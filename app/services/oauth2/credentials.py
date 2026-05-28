"""Kubernetes-based OAuth client credential retrieval."""

import base64
import logging

from kubernetes import client, config

from .models import OAuthClientCredentials
from .utils import AGENT_NAMESPACE

logger = logging.getLogger(__name__)


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

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    v1 = client.CoreV1Api()
    secret = v1.read_namespaced_secret(secret_name, AGENT_NAMESPACE)

    if not secret.data:
        #TODO dont break the whole agent if the secret is misconfigured, just skip oauth and log an error
        raise RuntimeError(
            f"OAuth secret '{secret_name}' in namespace '{AGENT_NAMESPACE}' is empty."
        )

    client_id = ""
    client_secret = ""
    scopes = ""
    if "clientID" in secret.data:
        client_id = base64.b64decode(secret.data["clientID"]).decode('utf-8')
    if "clientSecret" in secret.data:
        client_secret = base64.b64decode(secret.data["clientSecret"]).decode('utf-8')
    if 'scopes' in secret.data:
        scopes = base64.b64decode(secret.data['scopes']).decode('utf-8').strip()

    return OAuthClientCredentials(client_id=client_id, client_secret=client_secret, scopes=scopes)
