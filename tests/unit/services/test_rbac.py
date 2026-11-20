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
    """Patch httpx.AsyncClient so GET returns a Rancher list of *items*."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"data": items or []}
    client = AsyncMock()
    client.get.return_value = resp
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return patch("app.services.rbac.httpx.AsyncClient", return_value=ctx)


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
    with patch("app.services.rbac.httpx.AsyncClient", return_value=ctx):
        with pytest.raises(RBACError):
            await accessible_agent_configs("tok")


# --- AITask / AITaskDefinition access -----------------------------------------------------------

from app.services.rbac import (  # noqa: E402
    accessible_resource,
    accessible_resources,
    AccessDenied,
    NotFound,
    AITASK_STEVE_TYPE,
    AITASK_DEFINITION_STEVE_TYPE,
)


def _mock_http_json(status_code=200, payload=None):
    """Patch httpx.AsyncClient so GET returns *payload*; yields the client mock."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload if payload is not None else {}
    client = AsyncMock()
    client.get.return_value = resp
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return client, patch("app.services.rbac.httpx.AsyncClient", return_value=ctx)


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_aitasks_list_as_user(_url, _tls):
    client, p = _mock_http_json(payload={"data": [{"metadata": {"name": "t1"}}]})
    with p:
        result = await accessible_resources("tok", AITASK_STEVE_TYPE, "ns", "ai.cattle.io/task-definition=daily")
    assert result == [{"metadata": {"name": "t1"}}]
    args, kwargs = client.get.call_args
    assert args[0] == "https://rancher.test/v1/ai.cattle.io.aitask/ns"
    assert kwargs["headers"]["Cookie"] == "R_SESS=tok"
    assert kwargs["params"] == {"labelSelector": "ai.cattle.io/task-definition=daily"}


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_aitasks_list_without_selector(_url, _tls):
    client, p = _mock_http_json(payload={"data": []})
    with p:
        assert await accessible_resources("tok", AITASK_STEVE_TYPE, "ns") == []
    assert client.get.call_args.kwargs["params"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [401, 403])
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_aitasks_list_denied(_url, _tls, code):
    _, p = _mock_http_json(status_code=code)
    with p, pytest.raises(AccessDenied):
        await accessible_resources("tok", AITASK_STEVE_TYPE, "ns")


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_aitasks_list_other_error(_url, _tls):
    _, p = _mock_http_json(status_code=500)
    with p, pytest.raises(RBACError) as exc:
        await accessible_resources("tok", AITASK_STEVE_TYPE, "ns")
    assert not isinstance(exc.value, AccessDenied)


@pytest.mark.asyncio
async def test_aitasks_missing_token_raises():
    with pytest.raises(RBACError):
        await accessible_resources("", AITASK_STEVE_TYPE, "ns")
    with pytest.raises(RBACError):
        await accessible_resource("", AITASK_STEVE_TYPE, "ns", "t1")


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_aitask_get_as_user(_url, _tls):
    task = {"metadata": {"name": "t1", "uid": "u1"}}
    client, p = _mock_http_json(payload=task)
    with p:
        assert await accessible_resource("tok", AITASK_STEVE_TYPE, "ns", "t1") == task
    assert client.get.call_args.args[0] == "https://rancher.test/v1/ai.cattle.io.aitask/ns/t1"


@pytest.mark.asyncio
@pytest.mark.parametrize("code,exc_type", [
    (401, AccessDenied),
    (403, AccessDenied),
    (404, NotFound),
    (500, RBACError),
])
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_aitask_get_errors(_url, _tls, code, exc_type):
    _, p = _mock_http_json(status_code=code)
    with p, pytest.raises(exc_type):
        await accessible_resource("tok", AITASK_STEVE_TYPE, "ns", "t1")


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_aitask_request_error_raises(_url, _tls):
    client = AsyncMock()
    client.get.side_effect = Exception("boom")
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch("app.services.rbac.httpx.AsyncClient", return_value=ctx):
        with pytest.raises(RBACError):
            await accessible_resource("tok", AITASK_STEVE_TYPE, "ns", "t1")


@pytest.mark.asyncio
@patch("app.services.rbac._get_tls_verify", return_value=True)
@patch("app.services.rbac._rancher_url", return_value="https://rancher.test")
async def test_aitaskdefinitions_list_as_user(_url, _tls):
    client, p = _mock_http_json(payload={"data": [{"metadata": {"name": "daily"}}]})
    with p:
        result = await accessible_resources("tok", AITASK_DEFINITION_STEVE_TYPE, "ns")
    assert result == [{"metadata": {"name": "daily"}}]
    assert client.get.call_args.args[0] == "https://rancher.test/v1/ai.cattle.io.aitaskdefinition/ns"
