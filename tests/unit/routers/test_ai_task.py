import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from langgraph.store.memory import InMemoryStore

from app.routers import ai_task as ai_task_router
from app.services.rbac import (
    AITASK_DEFINITION_STEVE_TYPE,
    AITASK_STEVE_TYPE,
    AccessDenied,
    NotFound,
    RBACError,
)


def _task(name, created, uid=None, phase=None, definition="daily"):
    return {
        "metadata": {"name": name, "uid": uid or f"uid-{name}", "creationTimestamp": created},
        "spec": {"taskDefinitionRef": definition, "prompt": "check", "agents": ["rancher"]},
        "status": {"phase": phase} if phase else {},
    }


def _definition(name, display_name, schedule="@daily", status_=None, **spec):
    return {
        "metadata": {"name": name, "creationTimestamp": "2026-01-01T00:00:00Z"},
        "spec": {"displayName": display_name, "prompt": "check", "schedule": schedule, **spec},
        "status": status_ or {},
    }


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def mock_request(store, monkeypatch):
    monkeypatch.delenv("RANCHER_API_TOKEN", raising=False)
    req = MagicMock()
    req.app.memory_manager.get_store.return_value = store
    req.cookies = {"R_SESS": "token"}
    return req


@pytest.fixture
def list_resources():
    with patch.object(ai_task_router, "accessible_resources", AsyncMock(return_value=[])) as m:
        yield m


@pytest.fixture
def get_resource():
    with patch.object(ai_task_router, "accessible_resource", AsyncMock()) as m:
        yield m


# --- GET /aitaskdefinitions ---------------------------------------------------

@pytest.mark.asyncio
async def test_definitions_requires_token(mock_request, list_resources):
    mock_request.cookies = {}
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_task_definitions(mock_request)
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
    list_resources.assert_not_called()


@pytest.mark.asyncio
async def test_definitions_listed_as_user_sorted(mock_request, list_resources):
    list_resources.return_value = [
        _definition("weekly", "Weekly report", schedule="@weekly", suspend=True, timeZone="Europe/Madrid",
                    agents=["fleet"]),
        _definition("daily", "daily check", status_={"nextRunTime": "2026-01-02T00:00:00Z"}),
    ]
    resp = await ai_task_router.get_ai_task_definitions(mock_request)

    list_resources.assert_awaited_once_with("token", AITASK_DEFINITION_STEVE_TYPE, ai_task_router.NAMESPACE)
    assert resp.status_code == status.HTTP_200_OK
    body = json.loads(resp.body)
    assert [d["name"] for d in body] == ["daily", "weekly"]
    daily, weekly = body
    assert daily["timeZone"] == "UTC"
    assert daily["suspend"] is False
    assert daily["agents"] == []
    assert daily["nextRunTime"] == "2026-01-02T00:00:00Z"
    assert weekly["suspend"] is True
    assert weekly["timeZone"] == "Europe/Madrid"
    assert weekly["agents"] == ["fleet"]


@pytest.mark.asyncio
async def test_definitions_forbidden(mock_request, list_resources):
    list_resources.side_effect = AccessDenied("no")
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_task_definitions(mock_request)
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_definitions_rbac_failure_fails_closed(mock_request, list_resources):
    list_resources.side_effect = RBACError("rancher down")
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_task_definitions(mock_request)
    assert exc.value.status_code == status.HTTP_502_BAD_GATEWAY


# --- GET /aitasks?definition= -------------------------------------------------

@pytest.mark.asyncio
async def test_list_requires_token(mock_request, list_resources):
    mock_request.cookies = {}
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_tasks(mock_request, definition="daily")
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
    list_resources.assert_not_called()


@pytest.mark.asyncio
async def test_list_filters_by_definition(mock_request, list_resources):
    await ai_task_router.get_ai_tasks(mock_request, definition="daily")
    list_resources.assert_awaited_once_with(
        "token", AITASK_STEVE_TYPE, ai_task_router.NAMESPACE, "ai.cattle.io/task-definition=daily"
    )


@pytest.mark.asyncio
async def test_list_sorted_newest_first(mock_request, list_resources):
    list_resources.return_value = [
        _task("old", "2026-01-01T00:00:00Z", phase="Succeeded"),
        _task("new", "2026-02-01T00:00:00Z"),
    ]
    resp = await ai_task_router.get_ai_tasks(mock_request, definition="daily")

    assert resp.status_code == status.HTTP_200_OK
    body = json.loads(resp.body)
    assert [t["name"] for t in body] == ["new", "old"]
    assert body[0]["phase"] == "Pending"
    assert body[1]["phase"] == "Succeeded"


@pytest.mark.asyncio
async def test_list_forbidden(mock_request, list_resources):
    list_resources.side_effect = AccessDenied("no")
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_tasks(mock_request, definition="daily")
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_list_rbac_failure_fails_closed(mock_request, list_resources):
    list_resources.side_effect = RBACError("rancher down")
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_tasks(mock_request, definition="daily")
    assert exc.value.status_code == status.HTTP_502_BAD_GATEWAY


@pytest.fixture
def http_client(list_resources, monkeypatch):
    monkeypatch.delenv("RANCHER_API_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(ai_task_router.router)
    client = TestClient(app)
    client.cookies.set("R_SESS", "token")
    return client


def test_list_definition_is_required(http_client, list_resources):
    resp = http_client.get("/v1/api/aitasks")
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    list_resources.assert_not_called()


@pytest.mark.parametrize("bad", ["daily,foo!=bar", "Daily", "-daily", "a" * 64, "daily=x"])
def test_list_rejects_invalid_definition(http_client, list_resources, bad):
    resp = http_client.get("/v1/api/aitasks", params={"definition": bad})
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    list_resources.assert_not_called()


def test_list_valid_definition_over_http(http_client, list_resources):
    resp = http_client.get("/v1/api/aitasks", params={"definition": "daily-check.v2"})
    assert resp.status_code == status.HTTP_200_OK
    assert list_resources.call_args.args[3] == "ai.cattle.io/task-definition=daily-check.v2"


# --- GET /aitasks/{name} ------------------------------------------------------

@pytest.mark.asyncio
async def test_get_includes_stored_result(mock_request, get_resource, store):
    get_resource.return_value = _task("t1", "2026-01-01T00:00:00Z", uid="u1", phase="Succeeded")
    await store.aput(("aitasks",), "u1", {"output": "report", "error": None})

    resp = await ai_task_router.get_ai_task(mock_request, name="t1")

    body = json.loads(resp.body)
    assert resp.status_code == status.HTTP_200_OK
    assert body["output"] == "report"
    assert body["prompt"] == "check"
    assert body["agents"] == ["rancher"]
    get_resource.assert_awaited_once_with("token", AITASK_STEVE_TYPE, ai_task_router.NAMESPACE, "t1")


@pytest.mark.asyncio
async def test_get_without_stored_result(mock_request, get_resource):
    get_resource.return_value = _task("t1", "2026-01-01T00:00:00Z")
    body = json.loads((await ai_task_router.get_ai_task(mock_request, name="t1")).body)
    assert body["output"] is None
    assert body["error"] is None


@pytest.mark.asyncio
async def test_get_requires_token(mock_request, get_resource):
    mock_request.cookies = {}
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_task(mock_request, name="t1")
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
    get_resource.assert_not_called()


@pytest.mark.asyncio
async def test_get_not_found(mock_request, get_resource):
    get_resource.side_effect = NotFound("missing")
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_task(mock_request, name="missing")
    assert exc.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_get_forbidden_does_not_read_store(mock_request, get_resource):
    get_resource.side_effect = AccessDenied("no")
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_task(mock_request, name="t1")
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN
    mock_request.app.memory_manager.get_store.assert_not_called()


@pytest.mark.asyncio
async def test_get_rbac_failure_fails_closed(mock_request, get_resource):
    get_resource.side_effect = RBACError("rancher down")
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_task(mock_request, name="t1")
    assert exc.value.status_code == status.HTTP_502_BAD_GATEWAY


# --- Run now ----------------------------------------------------------------------------------

@pytest.fixture
def run_mocks(get_resource):
    with patch.object(ai_task_router, "has_active_task", AsyncMock(return_value=False)) as active, \
         patch.object(ai_task_router, "patch_resource", AsyncMock(return_value={})) as patch_, \
         patch.object(ai_task_router, "_k8s_api", MagicMock()):
        yield get_resource, active, patch_


@pytest.mark.asyncio
async def test_run_now_patches_annotation_as_user(mock_request, run_mocks):
    get_resource, active, patch_ = run_mocks
    resp = await ai_task_router.run_ai_task_definition(mock_request, "daily")

    assert resp.status_code == status.HTTP_202_ACCEPTED
    content = json.loads(resp.body)
    get_resource.assert_awaited_once_with("token", AITASK_DEFINITION_STEVE_TYPE, "cattle-ai-agent-system", "daily")
    token, steve_type, ns, name, body = patch_.call_args.args
    assert (token, steve_type, ns, name) == ("token", AITASK_DEFINITION_STEVE_TYPE, "cattle-ai-agent-system", "daily")
    requested_at = body["metadata"]["annotations"]["ai.cattle.io/run-requested-at"]
    assert content == {"name": ai_task_router.manual_task_name("daily", requested_at), "requestedAt": requested_at}


@pytest.mark.asyncio
@pytest.mark.parametrize("exc,code", [
    (NotFound("x"), status.HTTP_404_NOT_FOUND),
    (AccessDenied("x"), status.HTTP_403_FORBIDDEN),
    (RBACError("x"), status.HTTP_502_BAD_GATEWAY),
])
async def test_run_now_definition_errors(mock_request, run_mocks, exc, code):
    get_resource, active, patch_ = run_mocks
    get_resource.side_effect = exc
    with pytest.raises(HTTPException) as e:
        await ai_task_router.run_ai_task_definition(mock_request, "daily")
    assert e.value.status_code == code
    assert not active.called
    assert not patch_.called


@pytest.mark.asyncio
async def test_run_now_conflict_when_active(mock_request, run_mocks):
    _, active, patch_ = run_mocks
    active.return_value = True
    with pytest.raises(HTTPException) as e:
        await ai_task_router.run_ai_task_definition(mock_request, "daily")
    assert e.value.status_code == status.HTTP_409_CONFLICT
    assert not patch_.called


@pytest.mark.asyncio
async def test_run_now_patch_forbidden(mock_request, run_mocks):
    _, _, patch_ = run_mocks
    patch_.side_effect = AccessDenied("x")
    with pytest.raises(HTTPException) as e:
        await ai_task_router.run_ai_task_definition(mock_request, "daily")
    assert e.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_run_now_requires_token(mock_request, run_mocks):
    mock_request.cookies = {}
    with pytest.raises(HTTPException) as e:
        await ai_task_router.run_ai_task_definition(mock_request, "daily")
    assert e.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.parametrize("bad", ["Daily", "-daily", "a" * 64, "da_ily"])
def test_run_now_rejects_invalid_name(http_client, run_mocks, bad):
    resp = http_client.post(f"/v1/api/aitaskdefinitions/{bad}/run")
    assert resp.status_code == 422
    assert not run_mocks[2].called


def test_run_now_over_http(http_client, run_mocks):
    resp = http_client.post("/v1/api/aitaskdefinitions/daily/run")
    assert resp.status_code == 202
    assert resp.json()["name"].startswith("daily-")


def test_summary_includes_trigger():
    task = _task("t1", "2026-01-01T00:00:00Z")
    task["metadata"]["labels"] = {"ai.cattle.io/trigger": "manual"}
    assert ai_task_router._summary(task)["trigger"] == "manual"
