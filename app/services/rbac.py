"""
RBAC — filter AI agents using native Rancher authorization.

Rather than re-implementing role matching, access is delegated to Rancher: we
ask Rancher, *as the requesting user*, for the ``AIAgentConfig`` resources they
can see. Rancher resolves the user's full identity (user **and** group
memberships) and applies its access rules, so an agent is visible iff the user is
allowed to access the corresponding ``AIAgentConfig``. Access is configured with
ordinary Rancher GlobalRole rules (e.g. ``get`` scoped by ``resourceNames``) —
no custom permission model, and cluster admins naturally see all agents.

This is a single call per request (no per-agent SubjectAccessReview) and, unlike
a raw Kubernetes ``list``, honors ``resourceNames`` because Rancher filters the
collection to what the user may access.
"""

import os

import httpx2

from .agent.loader import AgentConfig, _crd_to_agent_config
from .auth import _load_rancher_url, _get_tls_verify

# Rancher (Steve) aggregated-API type id for AIAgentConfig.
AGENT_STEVE_TYPE = "ai.cattle.io.aiagentconfig"

# Feature flag. When disabled, filtering is skipped and every authenticated user
# receives the full agent list (legacy behavior). Evaluated once at import time.
RBAC_ENABLED = os.environ.get("RBAC_ENABLED", "true").lower() == "true"


class RBACError(Exception):
    """Raised when the user's accessible agents cannot be resolved (fail closed)."""
    pass


def rbac_enabled() -> bool:
    """Return whether RBAC agent filtering is enabled for this deployment."""
    return RBAC_ENABLED


def _rancher_url() -> str | None:
    """Resolve the Rancher server URL (env override, else cluster setting)."""
    return os.environ.get("RANCHER_URL", "").strip() or _load_rancher_url()


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
    if not token:
        raise RBACError("Missing Rancher session token")

    base = _rancher_url()
    if not base:
        raise RBACError("Rancher URL is not configured and could not be resolved")

    url = f"{base}/v1/{AGENT_STEVE_TYPE}"
    headers = {"Cookie": f"R_SESS={token}", "Accept": "application/json"}

    try:
        async with httpx2.AsyncClient(timeout=5.0, verify=_get_tls_verify()) as http_client:
            resp = await http_client.get(url, headers=headers)
    except Exception as e:
        raise RBACError(f"Failed to list AIAgentConfigs from Rancher: {e}") from e

    if resp.status_code != 200:
        raise RBACError(f"Rancher AIAgentConfig list returned HTTP {resp.status_code}")

    data = resp.json().get("data", [])
    return [
        _crd_to_agent_config(item)
        for item in data
        if item.get("spec", {}).get("enabled", True)
    ]
