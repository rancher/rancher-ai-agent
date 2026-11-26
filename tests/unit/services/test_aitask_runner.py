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


# --- credentials ---

import base64
from unittest.mock import patch

from kubernetes.client.rest import ApiException

from app.constants import LABEL_TASK_CREDENTIALS
from app.services.agent.loader import AuthenticationType
from app.services.rbac import RBACError


def _secret(data=None, labels=None):
    return SimpleNamespace(
        metadata=SimpleNamespace(labels={LABEL_TASK_CREDENTIALS: "true"} if labels is None else labels),
        data=data if data is not None else {"token": base64.b64encode(b"token-abc:secret\n").decode()},
    )


@pytest.fixture
def core_api(monkeypatch):
    monkeypatch.delenv("RANCHER_API_TOKEN", raising=False)
    api = MagicMock()
    with patch.object(aitask_runner, "_load_k8s_config"), \
         patch.object(aitask_runner.client, "CoreV1Api", return_value=api):
        yield api


@pytest.mark.asyncio
async def test_token_from_labelled_secret(core_api):
    core_api.read_namespaced_secret.return_value = _secret()

    token = await aitask_runner.resolve_task_token("ns", {"name": "liz-task-token"})

    assert token == "token-abc:secret"
    core_api.read_namespaced_secret.assert_called_once_with("liz-task-token", "ns")


@pytest.mark.asyncio
async def test_token_custom_key(core_api):
    core_api.read_namespaced_secret.return_value = _secret(data={"rancher": base64.b64encode(b"t2").decode()})
    assert await aitask_runner.resolve_task_token("ns", {"name": "s", "key": "rancher"}) == "t2"


@pytest.mark.asyncio
async def test_env_token_wins(core_api, monkeypatch):
    monkeypatch.setenv("RANCHER_API_TOKEN", "dev-token")

    assert await aitask_runner.resolve_task_token("ns", {"name": "s"}) == "dev-token"
    assert await aitask_runner.resolve_task_token("ns", None) == "dev-token"
    core_api.read_namespaced_secret.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("ref, secret, error, message", [
    (None, None, None, "no credentialsRef"),
    ({}, None, None, "no credentialsRef"),
    ({"name": "s"}, None, ApiException(status=404), "not found"),
    ({"name": "s"}, None, ApiException(status=403), "HTTP 403"),
    ({"name": "s"}, _secret(labels={}), None, "is not labelled"),
    ({"name": "s"}, _secret(labels={LABEL_TASK_CREDENTIALS: "false"}), None, "is not labelled"),
    ({"name": "s"}, _secret(data={}), None, "has no 'token' key"),
    ({"name": "s", "key": "other"}, _secret(), None, "has no 'other' key"),
    ({"name": "s"}, _secret(data={"token": base64.b64encode(b"  ").decode()}), None, "is empty"),
])
async def test_token_errors(core_api, ref, secret, error, message):
    if error:
        core_api.read_namespaced_secret.side_effect = error
    else:
        core_api.read_namespaced_secret.return_value = secret

    with pytest.raises(aitask_runner.TaskCredentialsError, match=message):
        await aitask_runner.resolve_task_token("ns", ref)


@pytest.mark.asyncio
async def test_task_token_reads_spec_credentials_ref():
    with patch.object(aitask_runner, "resolve_task_token", AsyncMock(return_value="t")) as resolve:
        assert await aitask_runner.task_token("ns", {"credentialsRef": {"name": "s"}}) == "t"
    resolve.assert_awaited_once_with("ns", {"name": "s"})


def _cfg(name, auth=AuthenticationType.RANCHER):
    return SimpleNamespace(name=name, authentication=auth)


@pytest.mark.asyncio
async def test_agents_scoped_to_token_user_with_rbac():
    accessible = AsyncMock(return_value=[_cfg("a"), _cfg("o", AuthenticationType.OAUTH2)])
    with patch.object(aitask_runner, "rbac_enabled", return_value=True), \
         patch.object(aitask_runner, "accessible_agent_configs", accessible), \
         patch.object(aitask_runner, "load_agent_configs") as load_all:
        configs = await aitask_runner.resolve_agent_configs(["a", "b", "o"], "task-token")

    assert [c.name for c in configs] == ["a"]
    accessible.assert_awaited_once_with("task-token")
    load_all.assert_not_called()


@pytest.mark.asyncio
async def test_agents_without_rbac_use_all_enabled():
    with patch.object(aitask_runner, "rbac_enabled", return_value=False), \
         patch.object(aitask_runner, "accessible_agent_configs", AsyncMock()) as accessible, \
         patch.object(aitask_runner, "load_agent_configs", return_value=[_cfg("a"), _cfg("b")]):
        configs = await aitask_runner.resolve_agent_configs([], "task-token")

    assert [c.name for c in configs] == ["a", "b"]
    accessible.assert_not_called()


@pytest.mark.asyncio
async def test_agents_rbac_error_fails():
    with patch.object(aitask_runner, "rbac_enabled", return_value=True), \
         patch.object(aitask_runner, "accessible_agent_configs", AsyncMock(side_effect=RBACError("down"))):
        with pytest.raises(RuntimeError, match="Unable to verify"):
            await aitask_runner.resolve_agent_configs([], "task-token")


@pytest.mark.asyncio
async def test_build_task_graph_passes_token():
    build = AsyncMock(return_value=(MagicMock(), [{"name": "a"}]))
    with patch.object(aitask_runner, "rbac_enabled", return_value=False), \
         patch.object(aitask_runner, "load_agent_configs", return_value=[_cfg("a")]), \
         patch.object(aitask_runner, "build_agent_from_configs", build):
        _, config, _ = await aitask_runner.build_task_graph(MagicMock(), [], "uid-1", "task-token")

    assert build.call_args.kwargs["rancher_token"] == "task-token"
    assert build.call_args.kwargs["allow_planner"] is False
    assert config["configurable"] == {"thread_id": "aitask-uid-1", "user_id": "aitask", "aitask": True, "agent": "a"}
