"""Unit tests for the AITask kopf controller."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from app.controllers import ai_task
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


def _graph(output="All good\nRESULT: Healthy", interrupts=()):
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content=output)]})
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(interrupts=list(interrupts)))
    return graph


async def _run(graph, configs, store, spec=None, status=None):
    patch_obj = _patch()
    spec = spec or {"taskDefinitionRef": "daily", "prompt": "check clusters", "agents": []}
    with patch.object(ai_task, "load_agent_configs", return_value=configs), \
         patch.object(ai_task, "build_agent_from_configs", AsyncMock(return_value=(graph, [{"name": c.name} for c in configs]))) as build, \
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
async def test_interrupt_fails_task(store):
    interrupt = SimpleNamespace(value={"message": "<confirmation-response>{}</confirmation-response>"})
    status, stored, _ = await _run(_graph(interrupts=[interrupt]), [_cfg()], store)

    assert status["phase"] == "Failed"
    assert "result" not in status
    assert "human validation" in stored["error"]
    assert stored["output"] is None


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
