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
    submit_validation,
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

    chat = await resolve_aitask_chat("token", store, "daily-1")

    assert chat.uid == "u1"
    assert chat.pending is None
    assert [m.content for m in chat.seed_messages] == ["check the clusters", "report"]
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


# --- human validation ---

_PENDING = {"id": "v1", "message": '<confirmation-response>{"type": "update"}</confirmation-response>', "requestedAt": "t0"}


@pytest.mark.asyncio
async def test_resolve_task_waiting_for_validation(get_resource):
    store = InMemoryStore()
    await store.aput(AITASK_STORE_NAMESPACE, "u1", {"pendingValidation": _PENDING})
    task = _task(phase="Running")
    task["status"]["pendingValidation"] = True
    get_resource.return_value = task

    chat = await resolve_aitask_chat("token", store, "daily-1")

    assert chat.pending == _PENDING
    assert chat.seed_messages == []
    assert chat.name == "daily-1"


@pytest.mark.asyncio
async def test_resolve_running_task_without_stored_validation_is_not_finished(get_resource):
    task = _task(phase="Running")
    task["status"]["pendingValidation"] = True
    get_resource.return_value = task
    with pytest.raises(TaskNotFinished):
        await resolve_aitask_chat("token", InMemoryStore(), "daily-1")


def test_seed_messages_include_validations():
    record = {
        "output": "scaled it",
        "validations": [{"message": _PENDING["message"], "response": "yes", "requestedAt": "t1", "answeredAt": "t2"}],
    }

    seed = build_seed_messages(_task(), record)

    assert [type(m) for m in seed] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    request = seed[1].content
    assert "asked for validation" in request
    assert '"type": "update"' in request
    assert "confirmation-response" not in request
    assert seed[2].content == "yes"
    assert seed[2].additional_kwargs["created_at"] == "t2"
    assert seed[3].content == "scaled it"
    assert all(m.additional_kwargs[AITASK_SEED_KWARG] for m in seed)
    assert len({m.id for m in seed}) == 4


@pytest.mark.asyncio
async def test_submit_validation_patches_annotation_as_user():
    import json
    from app.constants import ANNOTATION_VALIDATION_RESPONSE
    with patch.object(aitask_chat, "patch_resource", AsyncMock()) as patch_resource:
        answer = await submit_validation("token", "daily-1", "v1", "yes", "u-1")

    args = patch_resource.call_args.args
    assert args[:4] == ("token", AITASK_STEVE_TYPE, aitask_chat.NAMESPACE, "daily-1")
    value = json.loads(args[4]["metadata"]["annotations"][ANNOTATION_VALIDATION_RESPONSE])
    assert value == answer
    assert value["validationId"] == "v1"
    assert value["response"] == "yes"
    assert value["userId"] == "u-1"
    assert value["via"] == "chat"


@pytest.mark.asyncio
async def test_submit_validation_propagates_access_denied():
    with patch.object(aitask_chat, "patch_resource", AsyncMock(side_effect=AccessDenied("no"))):
        with pytest.raises(AccessDenied):
            await submit_validation("token", "daily-1", "v1", "yes", "u-1")
