import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status
from kubernetes.client.rest import ApiException
from langgraph.store.memory import InMemoryStore

from app.routers import ai_task as ai_task_router


def _task(name, created, uid=None, phase=None, definition="daily"):
    return {
        "metadata": {"name": name, "uid": uid or f"uid-{name}", "creationTimestamp": created},
        "spec": {"taskDefinitionRef": definition, "prompt": "check", "agents": ["rancher"]},
        "status": {"phase": phase} if phase else {},
    }


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def mock_request(store):
    req = MagicMock()
    req.app.memory_manager.get_store.return_value = store
    req.cookies = {"R_SESS": "token"}
    return req


@pytest.fixture
def api():
    api = MagicMock()
    with patch.object(ai_task_router, "_custom_objects_api", return_value=api), \
         patch.object(ai_task_router, "get_user_id_from_token", AsyncMock(return_value="u")):
        yield api


@pytest.mark.asyncio
async def test_list_requires_user(mock_request):
    with patch.object(ai_task_router, "get_user_id_from_token", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await ai_task_router.get_ai_tasks(mock_request)
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_list_sorted_newest_first(mock_request, api):
    api.list_namespaced_custom_object.return_value = {"items": [
        _task("old", "2026-01-01T00:00:00Z", phase="Succeeded"),
        _task("new", "2026-02-01T00:00:00Z"),
    ]}
    resp = await ai_task_router.get_ai_tasks(mock_request)

    assert resp.status_code == status.HTTP_200_OK
    body = json.loads(resp.body)
    assert [t["name"] for t in body] == ["new", "old"]
    assert body[0]["phase"] == "Pending"
    assert body[1]["phase"] == "Succeeded"
    assert "label_selector" not in api.list_namespaced_custom_object.call_args.kwargs


@pytest.mark.asyncio
async def test_list_filters_by_definition(mock_request, api):
    api.list_namespaced_custom_object.return_value = {"items": []}
    await ai_task_router.get_ai_tasks(mock_request, definition="daily")
    assert api.list_namespaced_custom_object.call_args.kwargs["label_selector"] == "ai.cattle.io/task-definition=daily"


@pytest.mark.asyncio
async def test_get_includes_stored_result(mock_request, api, store):
    api.get_namespaced_custom_object.return_value = _task("t1", "2026-01-01T00:00:00Z", uid="u1", phase="Succeeded")
    await store.aput(("aitasks",), "u1", {"output": "report", "error": None})

    resp = await ai_task_router.get_ai_task(mock_request, name="t1")

    body = json.loads(resp.body)
    assert resp.status_code == status.HTTP_200_OK
    assert body["output"] == "report"
    assert body["prompt"] == "check"
    assert body["agents"] == ["rancher"]


@pytest.mark.asyncio
async def test_get_without_stored_result(mock_request, api):
    api.get_namespaced_custom_object.return_value = _task("t1", "2026-01-01T00:00:00Z")
    body = json.loads((await ai_task_router.get_ai_task(mock_request, name="t1")).body)
    assert body["output"] is None
    assert body["error"] is None


@pytest.mark.asyncio
async def test_get_not_found(mock_request, api):
    api.get_namespaced_custom_object.side_effect = ApiException(status=404)
    with pytest.raises(HTTPException) as exc:
        await ai_task_router.get_ai_task(mock_request, name="missing")
    assert exc.value.status_code == status.HTTP_404_NOT_FOUND
