"""
Unit tests for RBAC agent filtering in app/services/rbac.py.

Access is resolved by asking Rancher (as the user) for the AIAgentConfigs they
can see — a single authenticated call that returns the already-scoped objects.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.rbac import accessible_agent_configs, RBACError


def _steve_item(name: str, enabled: bool = True) -> dict:
    return {"metadata": {"name": name}, "spec": {"displayName": name.capitalize(), "enabled": enabled}}


def _mock_http(status_code=200, items=None):
    """Patch httpx2.AsyncClient so GET returns a Rancher list of *items*."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"data": items or []}
    client = AsyncMock()
    client.get.return_value = resp
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return patch("app.services.rbac.httpx2.AsyncClient", return_value=ctx)


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_returns_scoped_agent_configs(_url, _tls):
    items = [_steve_item("rancher"), _steve_item("fleet")]
    with _mock_http(items=items):
        result = await accessible_agent_configs("tok")
    assert {c.name for c in result} == {"rancher", "fleet"}
    assert result[0].displayName == "Rancher"


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_disabled_agents_excluded(_url, _tls):
    items = [_steve_item("rancher"), _steve_item("legacy", enabled=False)]
    with _mock_http(items=items):
        result = await accessible_agent_configs("tok")
    assert {c.name for c in result} == {"rancher"}


@pytest.mark.asyncio
async def test_missing_token_raises():
    with pytest.raises(RBACError):
        await accessible_agent_configs("")


@pytest.mark.asyncio
@patch("app.services.rbac._rancher_url", return_value=None)
async def test_no_rancher_url_raises(_url):
    with pytest.raises(RBACError):
        await accessible_agent_configs("tok")


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_non_200_raises(_url, _tls):
    with _mock_http(status_code=403):
        with pytest.raises(RBACError):
            await accessible_agent_configs("tok")


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_request_error_raises(_url, _tls):
    client = AsyncMock()
    client.get.side_effect = Exception("boom")
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch("app.services.rbac.httpx2.AsyncClient", return_value=ctx):
        with pytest.raises(RBACError):
            await accessible_agent_configs("tok")
