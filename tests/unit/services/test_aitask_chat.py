from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from app.constants import AITASK_SEED_KWARG, AITASK_STORE_NAMESPACE
from app.services import aitask_chat
from app.services.aitask_chat import (
    TaskNotFinished,
    aitask_chat_id,
    build_seed_messages,
    resolve_aitask_chat,
    session_token,
)
from app.services.rbac import AITASK_STEVE_TYPE, AccessDenied, NotFound


def _task(phase="Succeeded", name="daily-1", uid="u1"):
    return {
        "metadata": {"name": name, "uid": uid},
        "spec": {"taskDefinitionRef": "daily", "prompt": "check the clusters"},
        "status": {"phase": phase, "startTime": "2026-01-01T00:00:00Z", "completionTime": "2026-01-01T00:05:00Z"},
    }


@pytest.fixture
def get_resource():
    with patch.object(aitask_chat, "accessible_resource", AsyncMock()) as m:
        yield m


def test_chat_id_is_deterministic_and_per_user():
    assert aitask_chat_id("u1", "user-a") == aitask_chat_id("u1", "user-a")
    assert aitask_chat_id("u1", "user-a") != aitask_chat_id("u1", "user-b")
    assert aitask_chat_id("u1", "user-a") != aitask_chat_id("u2", "user-a")
    assert aitask_chat_id("u1", "user-a").startswith("aitask-u1-")


def test_session_token(monkeypatch):
    monkeypatch.delenv("RANCHER_API_TOKEN", raising=False)
    assert session_token({"R_SESS": "t"}) == "t"
    assert session_token({}) == ""
    monkeypatch.setenv("RANCHER_API_TOKEN", "env")
    assert session_token({"R_SESS": "t"}) == "env"


def test_seed_messages_for_succeeded_task():
    human, ai = build_seed_messages(_task(), {"output": "all good\nRESULT: Healthy"})

    assert isinstance(human, HumanMessage) and isinstance(ai, AIMessage)
    assert human.content == "check the clusters"
    assert ai.content == "all good\nRESULT: Healthy"
    assert human.id and ai.id and human.id != ai.id
    for msg in (human, ai):
        assert msg.additional_kwargs[AITASK_SEED_KWARG] is True
        metadata = msg.additional_kwargs["request_metadata"]
        assert metadata["tags"] == ["aitask"]
        assert metadata["labels"] == {"summary": "Task daily: daily-1", "aitask": "daily-1"}
    assert human.additional_kwargs["created_at"] == "2026-01-01T00:00:00Z"
    assert ai.additional_kwargs["created_at"] == "2026-01-01T00:05:00Z"


def test_seed_messages_for_failed_task():
    _, ai = build_seed_messages(_task(phase="Failed"), {"phase": "Failed", "error": "boom"})
    assert ai.content == "The task failed: boom"


def test_seed_messages_without_stored_record():
    _, ai = build_seed_messages(_task(), {})
    assert ai.content == ""


@pytest.mark.asyncio
async def test_resolve_reads_task_as_user_and_store(get_resource):
    store = InMemoryStore()
    await store.aput(AITASK_STORE_NAMESPACE, "u1", {"output": "report"})
    get_resource.return_value = _task()

    uid, seed = await resolve_aitask_chat("token", store, "daily-1")

    assert uid == "u1"
    assert [m.content for m in seed] == ["check the clusters", "report"]
    get_resource.assert_awaited_once_with("token", AITASK_STEVE_TYPE, aitask_chat.NAMESPACE, "daily-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [None, "Pending", "Running"])
async def test_resolve_rejects_unfinished_task(get_resource, phase):
    task = _task(phase=phase)
    if phase is None:
        task["status"] = {}
    get_resource.return_value = task
    with pytest.raises(TaskNotFinished):
        await resolve_aitask_chat("token", InMemoryStore(), "daily-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [AccessDenied("no"), NotFound("missing")])
async def test_resolve_propagates_access_errors(get_resource, error):
    get_resource.side_effect = error
    store = AsyncMock()
    with pytest.raises(type(error)):
        await resolve_aitask_chat("token", store, "daily-1")
    store.aget.assert_not_called()
