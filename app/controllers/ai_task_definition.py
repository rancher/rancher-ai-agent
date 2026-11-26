"""
Scheduler for AITaskDefinition custom resources.

A kopf daemon runs for every AITaskDefinition and creates an AITask each time its
cron ``schedule`` (evaluated in ``timeZone``, UTC by default) fires. The AITask
executor (``ai_task.py``) then runs it.

Rules:
* ``suspend: true`` or an empty schedule: no AITasks are created.
* Overlap: if an AITask of the same definition is still Pending/Running, the run is skipped.
* Missed runs (e.g. pod downtime): only the most recent missed time is run, and only if
  it is at most ``STARTING_DEADLINE`` late. Every scheduled time is handled once
  (``status.lastScheduleTime``), whether it was run or skipped.
* AITask names are deterministic (``<definition>-<unix minutes>``), so creation is
  idempotent across restarts.

Run now: setting the ``ai.cattle.io/run-requested-at`` annotation to a new value creates one
AITask with the ``manual`` trigger (``<definition>-<hash of the value>``), even if the
definition is suspended or manual only. It is skipped if an AITask is still active.
"""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import kopf
from croniter import croniter
from kubernetes import client
from kubernetes.client.rest import ApiException

from ..constants import (
    ANNOTATION_RUN_REQUESTED_AT,
    ANNOTATION_SCHEDULED_AT,
    LABEL_TASK_DEFINITION,
    LABEL_TRIGGER,
    TRIGGER_MANUAL,
    TRIGGER_SCHEDULED,
)
from ..services.agent.loader import _load_k8s_config

GROUP = "ai.cattle.io"
VERSION = "v1alpha1"

STARTING_DEADLINE = timedelta(seconds=300)
MAX_SLEEP_SECONDS = 60.0

CONDITION_SCHEDULE_VALID = "ScheduleValid"
CONDITION_READY = "Ready"
REASON_VALID = "ScheduleValid"
REASON_INVALID_SCHEDULE = "InvalidSchedule"
REASON_UNKNOWN_TIME_ZONE = "UnknownTimeZone"
REASON_SUSPENDED = "Suspended"
REASON_MANUAL_ONLY = "ManualOnly"

_ACTIVE_PHASES = ("", "Pending", "Running")


class ScheduleError(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# --- helpers ---------------------------------------------------------------

def format_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_schedule(schedule: str, time_zone: str | None) -> ZoneInfo:
    """Validate the schedule and return its time zone. Raises ScheduleError."""
    try:
        tz = ZoneInfo(time_zone or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        raise ScheduleError(REASON_UNKNOWN_TIME_ZONE, f"Unknown time zone '{time_zone}'")
    if not croniter.is_valid(schedule):
        raise ScheduleError(REASON_INVALID_SCHEDULE, f"Invalid cron schedule '{schedule}'")
    return tz


def latest_due(schedule: str, tz: ZoneInfo, last: datetime, now: datetime) -> datetime | None:
    """Latest scheduled time in (last, now], or None if there is none."""
    due = croniter(schedule, now.astimezone(tz)).get_prev(datetime)
    return due if due > last else None


def next_run(schedule: str, tz: ZoneInfo, now: datetime) -> datetime:
    return croniter(schedule, now.astimezone(tz)).get_next(datetime)


def task_name(definition_name: str, due: datetime) -> str:
    return f"{definition_name}-{int(due.timestamp() // 60)}"


def manual_task_name(definition_name: str, requested_at: str) -> str:
    """Name of the AITask created for a Run now request. At most 52 + 1 + 10 = 63 chars."""
    return f"{definition_name}-{hashlib.sha1(requested_at.encode()).hexdigest()[:10]}"


def build_task_body(definition_name: str, definition_uid: str, spec: dict, due: datetime) -> dict:
    """AITask for the scheduled run at ``due``."""
    return _task_body(
        definition_name, definition_uid, spec, task_name(definition_name, due),
        TRIGGER_SCHEDULED, {ANNOTATION_SCHEDULED_AT: format_time(due)},
    )


def build_manual_task_body(definition_name: str, definition_uid: str, spec: dict, requested_at: str) -> dict:
    """AITask for the Run now request ``requested_at``."""
    return _task_body(
        definition_name, definition_uid, spec, manual_task_name(definition_name, requested_at),
        TRIGGER_MANUAL, {ANNOTATION_RUN_REQUESTED_AT: requested_at},
    )


def _task_body(definition_name: str, definition_uid: str, spec: dict, name: str, trigger: str, annotations: dict) -> dict:
    return {
        "apiVersion": f"{GROUP}/{VERSION}",
        "kind": "AITask",
        "metadata": {
            "name": name,
            "labels": {
                LABEL_TASK_DEFINITION: definition_name,
                LABEL_TRIGGER: trigger,
            },
            "annotations": annotations,
            "ownerReferences": [{
                "apiVersion": f"{GROUP}/{VERSION}",
                "kind": "AITaskDefinition",
                "name": definition_name,
                "uid": definition_uid,
                "controller": True,
                "blockOwnerDeletion": False,
            }],
        },
        "spec": {
            "taskDefinitionRef": definition_name,
            "prompt": spec["prompt"],
            **({"agents": list(spec["agents"])} if spec.get("agents") else {}),
            **({"credentialsRef": dict(spec["credentialsRef"])} if spec.get("credentialsRef") else {}),
        },
    }


# --- Kubernetes calls -----------------------------------------------------------

def _api() -> client.CustomObjectsApi:
    _load_k8s_config()
    return client.CustomObjectsApi()


async def has_active_task(api, namespace: str, definition_name: str) -> bool:
    tasks = await asyncio.to_thread(
        api.list_namespaced_custom_object,
        group=GROUP, version=VERSION, namespace=namespace, plural="aitasks",
        label_selector=f"{LABEL_TASK_DEFINITION}={definition_name}",
    )
    return any((t.get("status") or {}).get("phase", "") in _ACTIVE_PHASES for t in tasks.get("items", []))


async def _create_task(api, namespace: str, body: dict, logger) -> None:
    try:
        await asyncio.to_thread(
            api.create_namespaced_custom_object,
            group=GROUP, version=VERSION, namespace=namespace, plural="aitasks", body=body,
        )
        logger.info(f"Created AITask {namespace}/{body['metadata']['name']}")
    except ApiException as e:
        if e.status != 409:
            raise
        logger.info(f"AITask {namespace}/{body['metadata']['name']} already exists")


async def _patch_status(api, namespace: str, name: str, status: dict) -> None:
    await asyncio.to_thread(
        api.patch_namespaced_custom_object_status,
        group=GROUP, version=VERSION, namespace=namespace, plural="aitaskdefinitions",
        name=name, body={"status": status},
    )


def _conditions(ok: bool, reason: str, message: str, generation: int | None) -> list[dict]:
    now = format_time(datetime.now(timezone.utc))
    return [
        {
            "type": condition_type,
            "status": "True" if ok else "False",
            "reason": reason,
            "message": message,
            "lastTransitionTime": now,
            "observedGeneration": generation or 0,
        }
        for condition_type in (CONDITION_SCHEDULE_VALID, CONDITION_READY)
    ]


# --- daemon ---------------------------------------------------------------------

class _Scheduler:
    """State for one AITaskDefinition daemon. Writes status only when values change."""

    def __init__(self, api, name: str, namespace: str, uid: str, status: dict, meta: dict, logger):
        self.api = api
        self.name = name
        self.namespace = namespace
        self.uid = uid
        self.logger = logger
        self.last = parse_time(status.get("lastScheduleTime")) or parse_time(meta.get("creationTimestamp"))
        self.next_run = status.get("nextRunTime")
        ready = next((c for c in status.get("conditions") or [] if c.get("type") == CONDITION_READY), None)
        self.condition = (ready.get("status"), ready.get("reason")) if ready else None

    async def set_condition(self, ok: bool, reason: str, message: str, generation: int | None) -> None:
        key = ("True" if ok else "False", reason)
        if key == self.condition:
            return
        await _patch_status(self.api, self.namespace, self.name, {"conditions": _conditions(ok, reason, message, generation)})
        self.condition = key
        if not ok:
            self.logger.warning(f"AITaskDefinition {self.namespace}/{self.name}: {message}")

    async def set_next_run(self, value: str | None) -> None:
        if value == self.next_run:
            return
        await _patch_status(self.api, self.namespace, self.name, {"nextRunTime": value})
        self.next_run = value

    async def tick(self, spec: dict, generation: int | None, now: datetime) -> float:
        """Handle one iteration. Returns how many seconds to sleep."""
        schedule = (spec.get("schedule") or "").strip()
        if not schedule or spec.get("suspend"):
            reason = REASON_SUSPENDED if schedule else REASON_MANUAL_ONLY
            await self.set_condition(True, reason, "No AITasks are scheduled", generation)
            await self.set_next_run(None)
            # Don't run missed times when resumed: schedule from now on.
            self.last = max(self.last, now) if self.last else now
            return MAX_SLEEP_SECONDS

        try:
            tz = parse_schedule(schedule, spec.get("timeZone"))
        except ScheduleError as e:
            await self.set_condition(False, e.reason, str(e), generation)
            await self.set_next_run(None)
            return MAX_SLEEP_SECONDS
        await self.set_condition(True, REASON_VALID, "Schedule is valid", generation)

        due = latest_due(schedule, tz, self.last or now, now)
        if due:
            await self.handle_due(spec, due, now)

        upcoming = next_run(schedule, tz, now)
        await self.set_next_run(format_time(upcoming))
        # +1s so we wake up just after the boundary (latest_due excludes `now` itself).
        return max(1.0, min((upcoming - now).total_seconds() + 1.0, MAX_SLEEP_SECONDS))

    async def handle_due(self, spec: dict, due: datetime, now: datetime) -> None:
        if now - due > STARTING_DEADLINE:
            self.logger.warning(f"AITaskDefinition {self.namespace}/{self.name}: missed run at {format_time(due)}, skipping")
        elif await has_active_task(self.api, self.namespace, self.name):
            self.logger.info(f"AITaskDefinition {self.namespace}/{self.name}: previous AITask still active, skipping run at {format_time(due)}")
        else:
            await _create_task(self.api, self.namespace, build_task_body(self.name, self.uid, spec, due), self.logger)
        await _patch_status(self.api, self.namespace, self.name, {"lastScheduleTime": format_time(due)})
        self.last = due


@kopf.daemon(GROUP, VERSION, "aitaskdefinitions", cancellation_timeout=10.0)
async def schedule_daemon(spec, status, meta, name, namespace, uid, stopped, logger, **kwargs):
    """Create AITasks for an AITaskDefinition according to its schedule."""
    scheduler = _Scheduler(_api(), name, namespace, uid, dict(status or {}), dict(meta), logger)
    while not stopped:
        try:
            # spec/meta are live views: changes to the definition are picked up here.
            sleep_for = await scheduler.tick(dict(spec), meta.get("generation"), datetime.now(timezone.utc))
        except Exception:
            logger.error(f"AITaskDefinition {namespace}/{name}: scheduling failed, retrying")
            sleep_for = MAX_SLEEP_SECONDS
        await stopped.wait(sleep_for)


@kopf.on.field(GROUP, VERSION, "aitaskdefinitions", field=("metadata", "annotations", ANNOTATION_RUN_REQUESTED_AT))
async def run_now(new, spec, name, namespace, uid, logger, **kwargs):
    """Create a manual AITask when the run-requested-at annotation gets a new value."""
    requested_at = (new or "").strip()
    if not requested_at:
        return
    api = _api()
    if await has_active_task(api, namespace, name):
        logger.warning(f"AITaskDefinition {namespace}/{name}: Run now ({requested_at}) skipped, previous AITask still active")
        return
    await _create_task(api, namespace, build_manual_task_body(name, uid, dict(spec), requested_at), logger)
