"""Unit tests for the shared AITask runner."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.services import aitask_runner


def _record():
    return aitask_runner.new_record("t1", {"taskDefinitionRef": "daily", "prompt": "check"})


def _interrupt(message="<confirmation-response>{}</confirmation-response>"):
    return SimpleNamespace(value={"message": message}, id="int-1")


def _graph(*states):
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="done\nRESULT: Warning")]})
    graph.aget_state = AsyncMock(side_effect=list(states))
    return graph


@pytest.fixture(autouse=True)
def _reset_claims():
    aitask_runner._claimed_validations.clear()
    yield
    aitask_runner._claimed_validations.clear()


@pytest.mark.asyncio
async def test_finish_run_interrupt_marks_pending():
    graph = _graph(SimpleNamespace(interrupts=[_interrupt()], values={}))

    record = await aitask_runner.finish_run(graph, {}, _record(), {"messages": []})

    assert record["phase"] == "Running"
    assert record["completionTime"] is None
    assert record["pendingValidation"]["message"] == "<confirmation-response>{}</confirmation-response>"
    assert record["pendingValidation"]["interruptId"] == "int-1"


@pytest.mark.asyncio
async def test_finish_run_reads_state_without_result():
    state = SimpleNamespace(interrupts=[], values={"messages": [AIMessage(content="ok\nRESULT: Critical")]})

    record = await aitask_runner.finish_run(_graph(state), {}, _record())

    assert record["phase"] == "Succeeded"
    assert record["result"] == "Critical"
    assert record["output"] == "ok\nRESULT: Critical"
    assert record["completionTime"]


def test_claim_validation_only_once():
    assert aitask_runner.claim_validation("v1") is True
    assert aitask_runner.claim_validation("v1") is False
    assert aitask_runner.claim_validation("v2") is True


def test_status_patch():
    record = _record()
    assert aitask_runner.status_patch(record) == {
        "phase": "Running", "pendingValidation": False, "startTime": record["startTime"],
    }
    record.update(phase="Succeeded", result="Healthy", completionTime="t1")
    assert aitask_runner.status_patch(record)["result"] == "Healthy"
    assert aitask_runner.status_patch(record)["completionTime"] == "t1"


@pytest.mark.parametrize("value, expected", [
    ('{"response": "yes"}', {"response": "yes"}),
    ('{"response": "no", "via": "chat"}', {"response": "no", "via": "chat"}),
    ('{"response": 1}', None),
    ('[]', None),
    ('nope', None),
    (None, None),
])
def test_parse_validation_answer(value, expected):
    assert aitask_runner.parse_validation_answer(value) == expected


@pytest.mark.asyncio
async def test_resume_task_streams_and_records_answer():
    graph = _graph(
        SimpleNamespace(interrupts=[_interrupt()], values={}),
        SimpleNamespace(interrupts=[], values={"messages": [AIMessage(content="scaled\nRESULT: Healthy")]}),
    )
    record = _record()
    record["pendingValidation"] = {"id": "v1", "message": "m", "requestedAt": "t0"}
    stream = AsyncMock()
    answer = aitask_runner.build_validation_answer("v1", "yes", "u-1")

    record = await aitask_runner.resume_task(graph, {"configurable": {}}, record, answer, stream)

    graph.ainvoke.assert_not_called()
    assert stream.call_args.args[1].resume == "yes"
    assert record["phase"] == "Succeeded"
    assert record["output"] == "scaled\nRESULT: Healthy"
    assert record["validations"] == [{
        "message": "m", "response": "yes", "userId": "u-1", "via": "chat",
        "requestedAt": "t0", "answeredAt": answer["answeredAt"],
    }]


@pytest.mark.asyncio
async def test_resume_task_lost_checkpoint_fails():
    graph = _graph(SimpleNamespace(interrupts=[], values={}))

    record = await aitask_runner.resume_task(graph, {}, _record(), {"response": "yes"})

    assert record["phase"] == "Failed"
    assert record["error"] == aitask_runner.LOST_RUN_ERROR
    graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_resume_task_error_fails():
    graph = _graph(SimpleNamespace(interrupts=[_interrupt()], values={}))
    graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    record = await aitask_runner.resume_task(graph, {}, _record(), {"response": "yes"})

    assert record["phase"] == "Failed"
    assert record["error"] == "boom"
    assert len(record["validations"]) == 1
