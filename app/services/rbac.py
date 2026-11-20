"""
RBAC — resolve access to AI resources using native Rancher authorization.

Rather than re-implementing role matching, access is delegated to Rancher: we
ask Rancher, *as the requesting user*, for the resources (``AIAgentConfig``,
``AITask``, ``AITaskDefinition``) they can see. Rancher resolves the user's
full identity (user **and** group memberships) and applies its access rules. Access is configured with
ordinary Rancher GlobalRole rules (e.g. ``get`` scoped by ``resourceNames``) —
no custom permission model, and cluster admins naturally see everything.

This is a single call per request (no per-object SubjectAccessReview) and, unlike
a raw Kubernetes ``list``, honors ``resourceNames`` because Rancher filters the
collection to what the user may access.
"""

import os

import httpx

from .agent.loader import AgentConfig, _crd_to_agent_config
from .auth import _load_rancher_url, _get_tls_verify

# Rancher (Steve) aggregated-API type ids.
AGENT_STEVE_TYPE = "ai.cattle.io.aiagentconfig"
AITASK_STEVE_TYPE = "ai.cattle.io.aitask"
AITASK_DEFINITION_STEVE_TYPE = "ai.cattle.io.aitaskdefinition"

# Feature flag. When disabled, agent filtering is skipped and every authenticated
# user receives the full agent list (legacy behavior). Evaluated once at import
# time. It does not affect AITask/AITaskDefinition access checks, which are
# always enforced.
RBAC_ENABLED = os.environ.get("RBAC_ENABLED", "true").lower() == "true"


class RBACError(Exception):
    """Raised when the user's access cannot be resolved (fail closed)."""
    pass


class AccessDenied(RBACError):
    """Raised when Rancher denies the user access to a resource (401/403)."""
    pass


class NotFound(RBACError):
    """Raised when the requested resource does not exist (404)."""
    pass


def rbac_enabled() -> bool:
    """Return whether RBAC agent filtering is enabled for this deployment."""
    return RBAC_ENABLED


def _rancher_url() -> str | None:
    """Resolve the Rancher server URL (env override, else cluster setting)."""
    return os.environ.get("RANCHER_URL", "").strip() or _load_rancher_url()


async def _steve_get(token: str, path: str, params: dict | None = None) -> httpx.Response:
    """
    GET ``/v1/<path>`` from Rancher's Steve API as the holder of *token*.

    Raises:
        RBACError: If the token/URL is missing or the request fails.
    """
    if not token:
        raise RBACError("Missing Rancher session token")

    base = _rancher_url()
    if not base:
        raise RBACError("Rancher URL is not configured and could not be resolved")

    url = f"{base}/v1/{path}"
    headers = {"Cookie": f"R_SESS={token}", "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=5.0, verify=_get_tls_verify()) as http_client:
            return await http_client.get(url, headers=headers, params=params)
    except Exception as e:
        raise RBACError(f"Request to Rancher /v1/{path} failed: {e}") from e


async def accessible_agent_configs(token: str) -> list[AgentConfig]:
    """
    Fetch the enabled ``AIAgentConfig``s the holder of *token* may access.

    Queries Rancher's aggregated API with the user's session cookie, so the
    result reflects the user's full identity (user + groups) and honors
    ``resourceNames``-scoped access — a single call that returns the already
    scoped objects, so there's no need to separately list every config.

    Raises:
        RBACError: If the accessible agents cannot be resolved (fail closed).
    """
    resp = await _steve_get(token, AGENT_STEVE_TYPE)
    if resp.status_code != 200:
        raise RBACError(f"Rancher AIAgentConfig list returned HTTP {resp.status_code}")

    data = resp.json().get("data", [])
    return [
        _crd_to_agent_config(item)
        for item in data
        if item.get("spec", {}).get("enabled", True)
    ]


async def accessible_resources(
    token: str, steve_type: str, namespace: str, label_selector: str | None = None
) -> list[dict]:
    """
    List the *steve_type* objects in *namespace* the holder of *token* may access.

    Rancher resolves the user's roles and group memberships and returns only the
    objects they are allowed to see (honoring ``resourceNames``).

    Raises:
        AccessDenied: If Rancher denies the user listing the resource.
        RBACError: If access cannot be resolved (fail closed).
    """
    params = {"labelSelector": label_selector} if label_selector else None
    resp = await _steve_get(token, f"{steve_type}/{namespace}", params)
    if resp.status_code in (401, 403):
        raise AccessDenied(f"Access denied listing {steve_type} (HTTP {resp.status_code})")
    if resp.status_code != 200:
        raise RBACError(f"Rancher {steve_type} list returned HTTP {resp.status_code}")
    return resp.json().get("data", [])


async def accessible_resource(token: str, steve_type: str, namespace: str, name: str) -> dict:
    """
    Get the *steve_type* object *name* in *namespace* as the holder of *token*.

    Raises:
        NotFound: If the object does not exist (or is hidden from the user).
        AccessDenied: If Rancher denies the user access to the object.
        RBACError: If access cannot be resolved (fail closed).
    """
    resp = await _steve_get(token, f"{steve_type}/{namespace}/{name}")
    if resp.status_code in (401, 403):
        raise AccessDenied(f"Access denied to {steve_type} {name} (HTTP {resp.status_code})")
    if resp.status_code == 404:
        raise NotFound(f"{steve_type} {name} not found")
    if resp.status_code != 200:
        raise RBACError(f"Rancher {steve_type} {name} request returned HTTP {resp.status_code}")
    return resp.json()
