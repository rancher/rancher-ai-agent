"""Unit tests for the AITask kopf controller."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from app.controllers import ai_task
from app.services import aitask_runner
from app.services.agent.loader import AuthenticationType


def _cfg(name="rancher", auth=AuthenticationType.RANCHER):
    return SimpleNamespace(name=name, authentication=auth)


def _patch():
    p = MagicMock()
    p.status = {}
    return p


@pytest.fixture
def store():
    s = InMemoryStore()
    with patch.object(ai_task, "_store", s):
        yield s


@pytest.fixture
def status_patches():
    """Status patches made with the agent service account (resume path)."""
    calls = []

    async def record(namespace, name, status):
        calls.append(status)

    with patch.object(aitask_runner, "patch_task_status", side_effect=record):
        yield calls


def _interrupt(message="<confirmation-response>{}</confirmation-response>", id="int-1"):
    return SimpleNamespace(value={"message": message}, id=id)


def _graph(output="All good\nRESULT: Healthy", interrupts=()):
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content=output)]})
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(interrupts=list(interrupts)))
    return graph


async def _run(graph, configs, store, spec=None, status=None):
    patch_obj = _patch()
    spec = spec or {"taskDefinitionRef": "daily", "prompt": "check clusters", "agents": []}
    with patch.object(aitask_runner, "load_agent_configs", return_value=configs), \
         patch.object(aitask_runner, "build_agent_from_configs", AsyncMock(return_value=(graph, [{"name": c.name} for c in configs]))) as build, \
         patch.object(ai_task.LLMManager, "get_instance", return_value=MagicMock()):
        await ai_task.run_ai_task(spec=spec, status=status, name="t1", namespace="ns", uid="uid-1",
                                  logger=MagicMock(), patch=patch_obj)
    item = await store.aget(ai_task.AITASK_STORE_NAMESPACE, "uid-1")
    return patch_obj.status, (item.value if item else None), build


@pytest.mark.asyncio
async def test_success_stores_output_and_result(store):
    graph = _graph()
    status, stored, build = await _run(graph, [_cfg("a"), _cfg("b")], store)

    assert status["phase"] == "Succeeded"
    assert status["result"] == "Healthy"
    assert stored["output"] == "All good\nRESULT: Healthy"
    assert stored["phase"] == "Succeeded"
    assert stored["error"] is None
    assert build.call_args.kwargs["allow_planner"] is False
    configurable = graph.ainvoke.call_args.args[1]["configurable"]
    assert configurable["aitask"] is True
    assert "agent" not in configurable  # supervisor run


@pytest.mark.asyncio
async def test_single_agent_sets_agent_in_config(store):
    graph = _graph()
    await _run(graph, [_cfg("rancher")], store)
    assert graph.ainvoke.call_args.args[1]["configurable"]["agent"] == "rancher"


@pytest.mark.asyncio
async def test_interrupt_pauses_task_for_validation(store):
    status, stored, _ = await _run(_graph(interrupts=[_interrupt()]), [_cfg()], store)

    assert status["phase"] == "Running"
    assert status["pendingValidation"] is True
    assert "completionTime" not in status
    assert "result" not in status
    assert stored["phase"] == "Running"
    assert stored["error"] is None
    assert stored["output"] is None
    pending = stored["pendingValidation"]
    assert pending["message"] == "<confirmation-response>{}</confirmation-response>"
    assert pending["interruptId"] == "int-1"
    assert pending["id"]


@pytest.mark.asyncio
async def test_pending_task_fails_on_restart(store):
    graph = _graph()
    pending = {"id": "v1", "message": "m", "requestedAt": "t"}
    await store.aput(ai_task.AITASK_STORE_NAMESPACE, "uid-1", {
        **aitask_runner.new_record("t1", {"prompt": "check clusters"}), "pendingValidation": pending,
    })

    status, stored, _ = await _run(graph, [_cfg()], store, status={"phase": "Running", "pendingValidation": True})

    graph.ainvoke.assert_not_called()
    assert status["phase"] == "Failed"
    assert status["pendingValidation"] is False
    assert stored["error"] == aitask_runner.LOST_RUN_ERROR
    assert stored["pendingValidation"] is None


@pytest.mark.asyncio
async def test_no_usable_agents_fails_task(store):
    status, stored, _ = await _run(_graph(), [_cfg(auth=AuthenticationType.OAUTH2)], store)

    assert status["phase"] == "Failed"
    assert "No usable agents" in stored["error"]
    assert stored["completionTime"]


@pytest.mark.asyncio
async def test_finished_task_is_not_rerun(store):
    graph = _graph()
    status, stored, _ = await _run(graph, [_cfg()], store, status={"phase": "Succeeded"})

    graph.ainvoke.assert_not_called()
    assert status == {}
    assert stored is None


@pytest.mark.parametrize("output,expected", [
    ("report\nRESULT: Warning", "Warning"),
    ("report\n**RESULT:** Critical\n\n", "Critical"),
    ("report\nresult: healthy", "Healthy"),
    ("RESULT: Healthy\nmore text after", None),
    ("no verdict", None),
    ("", None),
])
def test_parse_result(output, expected):
    assert ai_task.parse_result(output) == expected


@pytest.mark.asyncio
async def test_startup_uses_sqlite_store_and_cleanup_closes_it(tmp_path, monkeypatch):
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    monkeypatch.setenv("DB_ENABLED", "false")
    monkeypatch.setenv("SQLITE_ENABLED", "true")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "store.db"))

    await ai_task.on_startup(logger=MagicMock())
    try:
        assert isinstance(ai_task._store, AsyncSqliteStore)
        assert ai_task._sqlite_conn is not None
    finally:
        await ai_task.on_cleanup()
    assert ai_task._store is None
    assert ai_task._sqlite_conn is None


# --- validation answers with the annotation (kubectl) ---

async def _pending_store(store, validation_id="v1"):
    record = aitask_runner.new_record("t1", {"taskDefinitionRef": "daily", "prompt": "check clusters"})
    record["pendingValidation"] = {"id": validation_id, "interruptId": "int-1", "message": "<confirmation-response>{}</confirmation-response>", "requestedAt": "t0"}
    await store.aput(ai_task.AITASK_STORE_NAMESPACE, "uid-1", record)
    return record


def _resumable_graph(output="Scaled\nRESULT: Healthy", interrupts_after=()):
    """A graph waiting on an interrupt, and whose state after the resume has *interrupts_after*."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content=output)]})
    graph.aget_state = AsyncMock(side_effect=[
        SimpleNamespace(interrupts=[_interrupt()], values={}),
        SimpleNamespace(interrupts=list(interrupts_after), values={}),
    ])
    return graph


async def _answer(graph, store, value, status=None):
    with patch.object(aitask_runner, "load_agent_configs", return_value=[_cfg()]), \
         patch.object(aitask_runner, "build_agent_from_configs", AsyncMock(return_value=(graph, [{"name": "rancher"}]))), \
         patch.object(ai_task.LLMManager, "get_instance", return_value=MagicMock()):
        await ai_task.answer_validation(
            new=value, spec={"prompt": "check clusters", "agents": []},
            status=status if status is not None else {"phase": "Running", "pendingValidation": True},
            name="t1", namespace="ns", uid="uid-1", logger=MagicMock(),
        )
    item = await store.aget(ai_task.AITASK_STORE_NAMESPACE, "uid-1")
    return item.value


@pytest.fixture(autouse=True)
def _reset_claims():
    aitask_runner._claimed_validations.clear()
    yield
    aitask_runner._claimed_validations.clear()


@pytest.mark.asyncio
async def test_kubectl_answer_resumes_task(store, status_patches):
    await _pending_store(store)
    graph = _resumable_graph()

    stored = await _answer(graph, store, '{"response": "yes"}')

    command = graph.ainvoke.call_args.args[0]
    assert command.resume == "yes"
    assert graph.ainvoke.call_args.args[1]["configurable"]["thread_id"] == "aitask-uid-1"
    assert stored["phase"] == "Succeeded"
    assert stored["result"] == "Healthy"
    assert stored["pendingValidation"] is None
    assert stored["validations"][0]["response"] == "yes"
    assert stored["validations"][0]["via"] == "kubectl"
    assert status_patches[0] == {"pendingValidation": False}
    assert status_patches[-1]["phase"] == "Succeeded"
    assert status_patches[-1]["pendingValidation"] is False


@pytest.mark.asyncio
async def test_kubectl_answer_matching_validation_id(store, status_patches):
    await _pending_store(store, "v1")
    stored = await _answer(_resumable_graph(), store, '{"response": "yes", "validationId": "v1"}')
    assert stored["phase"] == "Succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("value, status", [
    ('{"response": "yes", "via": "chat"}', None),
    ('{"response": "yes", "validationId": "old"}', None),
    ('not json', None),
    ('{"answer": "yes"}', None),
    ('{"response": "yes"}', {"phase": "Succeeded"}),
])
async def test_answer_ignored(store, status_patches, value, status):
    await _pending_store(store)
    graph = _resumable_graph()

    stored = await _answer(graph, store, value, status)

    graph.ainvoke.assert_not_called()
    assert stored["pendingValidation"]["id"] == "v1"
    assert status_patches == []


@pytest.mark.asyncio
async def test_already_claimed_answer_ignored(store, status_patches):
    await _pending_store(store)
    assert aitask_runner.claim_validation("v1")
    graph = _resumable_graph()

    await _answer(graph, store, '{"response": "yes"}')

    graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_second_interrupt_keeps_task_pending(store, status_patches):
    await _pending_store(store)
    graph = _resumable_graph(interrupts_after=[_interrupt(message="second")])

    stored = await _answer(graph, store, '{"response": "yes"}')

    assert stored["phase"] == "Running"
    assert stored["pendingValidation"]["message"] == "second"
    assert stored["pendingValidation"]["id"] != "v1"
    assert len(stored["validations"]) == 1
    assert status_patches[-1] == {"phase": "Running", "pendingValidation": True, "startTime": stored["startTime"]}


@pytest.mark.asyncio
async def test_lost_checkpoint_fails_task(store, status_patches):
    await _pending_store(store)
    graph = _graph()  # not waiting on an interrupt

    stored = await _answer(graph, store, '{"response": "yes"}')

    graph.ainvoke.assert_not_called()
    assert stored["phase"] == "Failed"
    assert stored["error"] == aitask_runner.LOST_RUN_ERROR
    assert status_patches[-1]["phase"] == "Failed"
