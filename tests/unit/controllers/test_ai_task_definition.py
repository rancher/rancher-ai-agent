"""Unit tests for the AITaskDefinition scheduler."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from kubernetes.client.rest import ApiException

from app.controllers import ai_task_definition as sched
from app.controllers.ai_task_definition import (
    ScheduleError,
    build_task_body,
    latest_due,
    next_run,
    parse_schedule,
    task_name,
)

UTC = ZoneInfo("UTC")


def _dt(*args, tz=timezone.utc):
    return datetime(*args, tzinfo=tz)


# --- pure helpers ---------------------------------------------------------------

def test_latest_due_returns_latest_past_time():
    due = latest_due("*/10 * * * *", UTC, _dt(2026, 1, 1), _dt(2026, 1, 1, 10, 25))
    assert due == _dt(2026, 1, 1, 10, 20)


def test_latest_due_none_when_already_handled():
    assert latest_due("0 * * * *", UTC, _dt(2026, 1, 1, 10), _dt(2026, 1, 1, 10, 30)) is None


def test_latest_due_none_before_first_occurrence():
    assert latest_due("0 9 * * *", UTC, _dt(2026, 1, 1, 8), _dt(2026, 1, 1, 8, 59)) is None


def test_latest_due_uses_time_zone():
    tz = ZoneInfo("Europe/Madrid")  # UTC+1 in winter
    due = latest_due("0 9 * * *", tz, _dt(2026, 1, 1), _dt(2026, 1, 1, 8, 30))
    assert due == _dt(2026, 1, 1, 8)  # 09:00 Madrid
    assert next_run("0 9 * * *", tz, _dt(2026, 1, 1, 8, 30)) == _dt(2026, 1, 2, 8)


def test_latest_due_supports_macros():
    assert latest_due("@hourly", UTC, _dt(2026, 1, 1), _dt(2026, 1, 1, 10, 5)) == _dt(2026, 1, 1, 10)


@pytest.mark.parametrize("schedule,time_zone,reason", [
    ("not a cron", None, sched.REASON_INVALID_SCHEDULE),
    ("0 9 * * *", "Mars/Olympus", sched.REASON_UNKNOWN_TIME_ZONE),
])
def test_parse_schedule_errors(schedule, time_zone, reason):
    with pytest.raises(ScheduleError) as exc:
        parse_schedule(schedule, time_zone)
    assert exc.value.reason == reason


def test_task_name_and_body():
    due = _dt(2026, 1, 1, 10)
    body = build_task_body("daily", "uid-1", {"prompt": "check", "agents": ["rancher"], "schedule": "x"}, due)

    assert body["metadata"]["name"] == task_name("daily", due) == f"daily-{int(due.timestamp() // 60)}"
    assert body["metadata"]["labels"] == {"ai.cattle.io/task-definition": "daily", "ai.cattle.io/trigger": "scheduled"}
    assert body["metadata"]["annotations"] == {"ai.cattle.io/scheduled-at": "2026-01-01T10:00:00Z"}
    owner = body["metadata"]["ownerReferences"][0]
    assert (owner["kind"], owner["name"], owner["uid"], owner["controller"]) == ("AITaskDefinition", "daily", "uid-1", True)
    assert body["spec"] == {"taskDefinitionRef": "daily", "prompt": "check", "agents": ["rancher"]}


def test_body_omits_empty_agents():
    body = build_task_body("daily", "uid-1", {"prompt": "check"}, _dt(2026, 1, 1))
    assert "agents" not in body["spec"]
    assert "credentialsRef" not in body["spec"]


def test_body_copies_credentials_ref():
    ref = {"name": "liz-task-token", "key": "token"}
    body = build_task_body("daily", "uid-1", {"prompt": "check", "credentialsRef": ref}, _dt(2026, 1, 1))
    assert body["spec"]["credentialsRef"] == ref
    manual = sched.build_manual_task_body("daily", "uid-1", {"prompt": "check", "credentialsRef": ref}, "2026-01-01T10:00:00.000Z")
    assert manual["spec"]["credentialsRef"] == ref


# --- scheduler ------------------------------------------------------------------

def _api(active_phases=()):
    api = MagicMock()
    api.list_namespaced_custom_object.return_value = {
        "items": [{"status": {"phase": p}} for p in active_phases]
    }
    return api


def _scheduler(api, last="2026-01-01T00:00:00Z", conditions=None):
    status = {"lastScheduleTime": last} if last else {}
    if conditions:
        status["conditions"] = conditions
    meta = {"creationTimestamp": "2025-12-31T00:00:00Z"}
    return sched._Scheduler(api, "daily", "ns", "uid-1", status, meta, MagicMock())


def _status_patches(api) -> dict:
    merged = {}
    for call in api.patch_namespaced_custom_object_status.call_args_list:
        merged.update(call.kwargs["body"]["status"])
    return merged


SPEC = {"schedule": "0 * * * *", "prompt": "check"}


@pytest.mark.asyncio
async def test_tick_creates_task_when_due():
    api = _api()
    s = _scheduler(api, last="2026-01-01T09:00:00Z")
    sleep = await s.tick(SPEC, 1, _dt(2026, 1, 1, 10, 4))

    api.create_namespaced_custom_object.assert_called_once()
    assert api.create_namespaced_custom_object.call_args.kwargs["body"]["metadata"]["name"] == task_name("daily", _dt(2026, 1, 1, 10))
    status = _status_patches(api)
    assert status["lastScheduleTime"] == "2026-01-01T10:00:00Z"
    assert status["nextRunTime"] == "2026-01-01T11:00:00Z"
    assert {c["type"]: c["status"] for c in status["conditions"]} == {"ScheduleValid": "True", "Ready": "True"}
    assert sleep == sched.MAX_SLEEP_SECONDS


@pytest.mark.asyncio
async def test_tick_skips_missed_run_past_deadline():
    api = _api()
    s = _scheduler(api, last="2026-01-01T09:00:00Z")
    await s.tick(SPEC, 1, _dt(2026, 1, 1, 10, 6))

    api.create_namespaced_custom_object.assert_not_called()
    assert _status_patches(api)["lastScheduleTime"] == "2026-01-01T10:00:00Z"


@pytest.mark.asyncio
async def test_tick_runs_only_latest_missed_time():
    api = _api()
    s = _scheduler(api, last="2026-01-01T05:00:00Z")
    await s.tick(SPEC, 1, _dt(2026, 1, 1, 10, 2))

    api.create_namespaced_custom_object.assert_called_once()
    assert _status_patches(api)["lastScheduleTime"] == "2026-01-01T10:00:00Z"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["", "Pending", "Running"])
async def test_tick_skips_when_task_active(phase):
    api = _api(active_phases=["Succeeded", phase])
    s = _scheduler(api, last="2026-01-01T09:00:00Z")
    await s.tick(SPEC, 1, _dt(2026, 1, 1, 10, 1))

    api.create_namespaced_custom_object.assert_not_called()
    assert _status_patches(api)["lastScheduleTime"] == "2026-01-01T10:00:00Z"


@pytest.mark.asyncio
async def test_tick_ignores_conflict_on_create():
    api = _api()
    api.create_namespaced_custom_object.side_effect = ApiException(status=409)
    s = _scheduler(api, last="2026-01-01T09:00:00Z")
    await s.tick(SPEC, 1, _dt(2026, 1, 1, 10, 1))

    assert _status_patches(api)["lastScheduleTime"] == "2026-01-01T10:00:00Z"


@pytest.mark.asyncio
async def test_tick_uses_creation_time_first_time():
    api = _api()
    s = _scheduler(api, last=None)  # created 2025-12-31T00:00:00Z
    await s.tick(SPEC, 1, _dt(2026, 1, 1, 10, 1))
    api.create_namespaced_custom_object.assert_called_once()


@pytest.mark.asyncio
async def test_tick_invalid_schedule_sets_condition():
    api = _api()
    s = _scheduler(api)
    await s.tick({"schedule": "bogus", "prompt": "p"}, 3, _dt(2026, 1, 1, 10))

    api.create_namespaced_custom_object.assert_not_called()
    conditions = _status_patches(api)["conditions"]
    assert all(c["status"] == "False" and c["reason"] == "InvalidSchedule" for c in conditions)
    assert all(c["observedGeneration"] == 3 for c in conditions)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec,reason", [
    ({"schedule": "0 * * * *", "suspend": True, "prompt": "p"}, "Suspended"),
    ({"prompt": "p"}, "ManualOnly"),
])
async def test_tick_suspended_or_manual(spec, reason):
    api = _api()
    s = _scheduler(api, last="2026-01-01T09:00:00Z")
    s.next_run = "2026-01-01T10:00:00Z"
    await s.tick(spec, 1, _dt(2026, 1, 1, 10, 1))

    api.create_namespaced_custom_object.assert_not_called()
    status = _status_patches(api)
    assert status["nextRunTime"] is None
    assert status["conditions"][0]["reason"] == reason


@pytest.mark.asyncio
async def test_status_not_rewritten_when_unchanged():
    api = _api()
    ready = [{"type": "Ready", "status": "True", "reason": "ScheduleValid"}]
    s = _scheduler(api, last="2026-01-01T10:00:00Z", conditions=ready)
    s.next_run = "2026-01-01T11:00:00Z"
    await s.tick(SPEC, 1, _dt(2026, 1, 1, 10, 30))

    api.patch_namespaced_custom_object_status.assert_not_called()


@pytest.mark.asyncio
async def test_daemon_runs_one_iteration_and_stops():
    api = _api()
    stopped = MagicMock()
    stopped.__bool__.side_effect = [False, True]
    stopped.wait = AsyncMock()
    with patch.object(sched, "_api", return_value=api):
        await sched.schedule_daemon(
            spec=SPEC, status={}, meta={"creationTimestamp": "2025-12-31T00:00:00Z", "generation": 1},
            name="daily", namespace="ns", uid="uid-1", stopped=stopped, logger=MagicMock(),
        )

    stopped.wait.assert_awaited_once()
    assert api.patch_namespaced_custom_object_status.called


@pytest.mark.asyncio
async def test_daemon_survives_errors():
    api = _api()
    api.patch_namespaced_custom_object_status.side_effect = RuntimeError("boom")
    stopped = MagicMock()
    stopped.__bool__.side_effect = [False, True]
    stopped.wait = AsyncMock()
    with patch.object(sched, "_api", return_value=api):
        await sched.schedule_daemon(
            spec=SPEC, status={}, meta={"creationTimestamp": "2025-12-31T00:00:00Z"},
            name="daily", namespace="ns", uid="uid-1", stopped=stopped, logger=MagicMock(),
        )
    stopped.wait.assert_awaited_once_with(sched.MAX_SLEEP_SECONDS)


# --- Run now --------------------------------------------------------------------

def test_manual_task_name_is_deterministic_and_fits():
    name = "a" * 52
    assert sched.manual_task_name(name, "t1") == sched.manual_task_name(name, "t1")
    assert sched.manual_task_name(name, "t1") != sched.manual_task_name(name, "t2")
    assert len(sched.manual_task_name(name, "2026-01-01T00:00:00.000Z")) == 63


def test_manual_task_body():
    body = sched.build_manual_task_body("daily", "uid-1", {"prompt": "check"}, "2026-01-01T10:00:00.000Z")
    assert body["metadata"]["name"] == sched.manual_task_name("daily", "2026-01-01T10:00:00.000Z")
    assert body["metadata"]["labels"] == {"ai.cattle.io/task-definition": "daily", "ai.cattle.io/trigger": "manual"}
    assert body["metadata"]["annotations"] == {"ai.cattle.io/run-requested-at": "2026-01-01T10:00:00.000Z"}
    assert body["metadata"]["ownerReferences"][0]["uid"] == "uid-1"
    assert body["spec"] == {"taskDefinitionRef": "daily", "prompt": "check"}


async def _run_now(api, new):
    logger = MagicMock()
    with patch.object(sched, "_api", return_value=api):
        await sched.run_now(
            new=new, spec={"prompt": "check", "suspend": True}, name="daily", namespace="ns",
            uid="uid-1", logger=logger,
        )
    return logger


@pytest.mark.asyncio
async def test_run_now_creates_manual_task():
    api = _api()
    await _run_now(api, "2026-01-01T10:00:00.000Z")
    body = api.create_namespaced_custom_object.call_args.kwargs["body"]
    assert body["metadata"]["labels"]["ai.cattle.io/trigger"] == "manual"
    assert not api.patch_namespaced_custom_object_status.called


@pytest.mark.asyncio
@pytest.mark.parametrize("new", [None, "", "  "])
async def test_run_now_ignores_empty_value(new):
    api = _api()
    await _run_now(api, new)
    assert not api.create_namespaced_custom_object.called


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["", "Pending", "Running"])
async def test_run_now_skips_when_task_active(phase):
    api = _api(active_phases=[phase])
    logger = await _run_now(api, "t1")
    assert not api.create_namespaced_custom_object.called
    logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_run_now_ignores_conflict():
    api = _api()
    api.create_namespaced_custom_object.side_effect = ApiException(status=409)
    await _run_now(api, "t1")
