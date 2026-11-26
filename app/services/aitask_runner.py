"""
Running AITask graphs.

Shared by the AITask controller (first run, and validation answers given with
kubectl) and the follow-up websocket (validation answers given in the chat).

A run that reaches a tool needing human validation is paused: the record gets a
``pendingValidation`` entry and the AITask ``status.pendingValidation=true``
(the phase stays ``Running``). An answer resumes the interrupted graph with
``Command(resume=<answer>)`` under the task identity, not the identity of the
user who answered.

The graphs are checkpointed in the process-wide in-memory ``checkpointer``, so a
paused run can be resumed from either event loop (kopf or FastAPI), but not after
a restart.
"""

import asyncio
import json
import logging
import re
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from uuid import uuid4

from kubernetes import client
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.types import Command

from ..constants import AITASK_STORE_NAMESPACE, VALIDATION_VIA_CHAT
from .agent.factory import build_agent_from_configs
from .agent.loader import AgentConfig, AuthenticationType, _load_k8s_config, load_agent_configs
from .agent.system_prompts import AITASK_RESULT_VALUES

GROUP = "ai.cattle.io"
VERSION = "v1alpha1"

LOST_RUN_ERROR = "The run waiting for validation can no longer be resumed (the agent restarted)."

_RESULT_PATTERN = re.compile(r"^\s*\**RESULT:\**\s*\**(" + "|".join(AITASK_RESULT_VALUES) + r")\b", re.IGNORECASE)

# TODO: use a persistent checkpointer, so paused runs survive a restart.
checkpointer = InMemorySaver()

_claimed_validations: set[str] = set()
_claim_lock = threading.Lock()

# Streams a graph run (graph, input, config) to the client.
StreamFn = Callable[[object, object, dict], Awaitable[None]]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_record(name: str, spec: dict) -> dict:
    """The store record of an AITask run that starts now."""
    return {
        "name": name,
        "taskDefinitionRef": spec.get("taskDefinitionRef"),
        "prompt": spec["prompt"],
        "agents": list(spec.get("agents") or []),
        "phase": "Running",
        "result": None,
        "output": None,
        "error": None,
        "startTime": now(),
        "completionTime": None,
        "pendingValidation": None,
        "validations": [],
    }


def resolve_agent_configs(agent_names: list[str]) -> list[AgentConfig]:
    """Select the enabled AIAgentConfigs named by the task (all of them if the list is empty)."""
    configs = load_agent_configs()
    if agent_names:
        known = {c.name for c in configs}
        for name in agent_names:
            if name not in known:
                logging.warning(f"AITask references unknown or disabled agent '{name}'")
        configs = [c for c in configs if c.name in agent_names]

    usable = []
    for cfg in configs:
        if cfg.authentication == AuthenticationType.OAUTH2:
            logging.warning(f"Agent '{cfg.name}' requires interactive OAuth2 login, skipping for AITask")
            continue
        usable.append(cfg)
    return usable


async def build_task_graph(llm, agent_names: list[str], uid: str) -> tuple[object, dict, list[dict]]:
    """
    Build the graph of the AITask with uid *uid* and its run config.

    One agent gives a single agent, several give a supervisor. The thread id is
    derived from the uid, so building it again resumes the same run.
    """
    configs = resolve_agent_configs(agent_names)
    if not configs:
        raise RuntimeError("No usable agents available for this task")

    graph, agents_metadata = await build_agent_from_configs(llm, configs, checkpointer, allow_planner=False)

    configurable = {"thread_id": f"aitask-{uid}", "user_id": "aitask", "aitask": True}
    if len(configs) == 1:
        # Direct child call: identity_preamble_middleware adds the Liz identity.
        configurable["agent"] = configs[0].name
    return graph, {"configurable": configurable}, agents_metadata


def parse_result(output: str) -> str | None:
    """Return the Healthy/Warning/Critical verdict from the last RESULT line of *output*, if any."""
    for line in reversed(output.strip().splitlines()):
        match = _RESULT_PATTERN.match(line)
        if match:
            return match.group(1).capitalize()
        if line.strip():
            return None
    return None


def interrupt_message(interrupts) -> str:
    value = interrupts[0].value
    if isinstance(value, dict):
        return str(value.get("message", ""))
    return str(value)


def fail_record(record: dict, error: str) -> dict:
    record["phase"] = "Failed"
    record["error"] = error
    record["pendingValidation"] = None
    record["completionTime"] = now()
    return record


async def finish_run(graph, config: dict, record: dict, result: dict | None = None) -> dict:
    """
    Update *record* after a graph run (first run or resume).

    If the graph stopped on a human-validation interrupt, the record is marked as
    pending validation and the phase stays ``Running``. Otherwise the output and
    the verdict of the last message are stored and the phase is ``Succeeded``.
    *result* is the return value of ``ainvoke``. Without it (streamed runs), the
    messages are read from the checkpointed state.
    """
    state = await graph.aget_state(config)
    if state and state.interrupts:
        record["phase"] = "Running"
        record["pendingValidation"] = {
            "id": str(uuid4()),
            "interruptId": getattr(state.interrupts[0], "id", None),
            "message": interrupt_message(state.interrupts),
            "requestedAt": now(),
        }
        return record

    if result is not None:
        messages = result.get("messages") or []
    else:
        messages = (getattr(state, "values", None) or {}).get("messages") or []
    output = str(messages[-1].text) if messages else ""
    record["output"] = output
    record["result"] = parse_result(output)
    record["phase"] = "Succeeded"
    record["pendingValidation"] = None
    record["completionTime"] = now()
    return record


def claim_validation(validation_id: str) -> bool:
    """
    Claim the right to resume the run waiting on validation *validation_id*.

    Only the first caller in the process wins, so an answer from the chat and one
    from kubectl (or two chats) never resume the same interrupt twice.
    """
    with _claim_lock:
        if validation_id in _claimed_validations:
            return False
        _claimed_validations.add(validation_id)
        return True


def parse_validation_answer(value: str | None) -> dict | None:
    """
    Parse the validation-response annotation value.

    It is JSON with at least ``response``, and optionally ``validationId``,
    ``userId``, ``answeredAt`` and ``via``. Returns None if it is not valid.
    """
    try:
        answer = json.loads(value or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(answer, dict) or not isinstance(answer.get("response"), str):
        return None
    return answer


def build_validation_answer(validation_id: str, response: str, user_id: str | None, via: str = VALIDATION_VIA_CHAT) -> dict:
    return {
        "validationId": validation_id,
        "response": response,
        "userId": user_id,
        "answeredAt": now(),
        "via": via,
    }


async def resume_task(graph, config: dict, record: dict, answer: dict, stream: StreamFn | None = None) -> dict:
    """
    Resume the run of *record* with the validation *answer*.

    The answer is appended to ``record.validations``. With *stream* the resumed run
    is streamed (the follow-up websocket), otherwise it is invoked. If the graph is
    not waiting on an interrupt (in-memory state lost after a restart), the record
    is marked as failed.
    """
    state = await graph.aget_state(config)
    if not (state and state.interrupts):
        return fail_record(record, LOST_RUN_ERROR)

    pending = record.get("pendingValidation") or {}
    record.setdefault("validations", []).append({
        "message": pending.get("message") or interrupt_message(state.interrupts),
        "response": answer["response"],
        "userId": answer.get("userId"),
        "via": answer.get("via") or "kubectl",
        "requestedAt": pending.get("requestedAt"),
        "answeredAt": answer.get("answeredAt") or now(),
    })
    record["pendingValidation"] = None

    try:
        command = Command(resume=answer["response"])
        result = None
        if stream is not None:
            await stream(graph, command, config)
        else:
            result = await graph.ainvoke(command, config)
        return await finish_run(graph, config, record, result)
    except Exception as e:
        logging.exception(f"Resumed run of AITask {record.get('name')} failed")
        return fail_record(record, str(e))


def status_patch(record: dict) -> dict:
    """The AITask status matching *record*."""
    status = {
        "phase": record["phase"],
        "pendingValidation": bool(record.get("pendingValidation")),
    }
    if record.get("startTime"):
        status["startTime"] = record["startTime"]
    if record.get("completionTime"):
        status["completionTime"] = record["completionTime"]
    if record.get("result"):
        status["result"] = record["result"]
    return status


async def patch_task_status(namespace: str, name: str, status: dict) -> None:
    """Patch the status of AITask *name* with the agent service account."""
    def _patch():
        _load_k8s_config()
        client.CustomObjectsApi().patch_namespaced_custom_object_status(
            group=GROUP, version=VERSION, namespace=namespace, plural="aitasks",
            name=name, body={"status": status},
        )
    await asyncio.to_thread(_patch)


async def load_record(store: BaseStore, uid: str) -> dict | None:
    item = await store.aget(AITASK_STORE_NAMESPACE, uid)
    return dict(item.value) if item else None


async def save_record(store: BaseStore | None, uid: str, record: dict) -> None:
    if store is None:
        logging.error("AITask store is not initialized, result not stored")
        return
    try:
        await store.aput(AITASK_STORE_NAMESPACE, uid, record)
    except Exception:
        logging.exception(f"Failed to store result of AITask {record.get('name')}")


async def continue_task(
    store: BaseStore,
    llm,
    namespace: str,
    name: str,
    uid: str,
    agent_names: list[str],
    record: dict,
    answer: dict,
    stream: StreamFn | None = None,
) -> dict:
    """
    Resume a paused AITask run with a validation *answer* that has been claimed
    (``claim_validation``), then store the record and patch the AITask status.

    ``status.pendingValidation`` is cleared first so the run shows as in progress.
    Errors end the task as failed, so it is never left waiting.
    """
    try:
        await patch_task_status(namespace, name, {"pendingValidation": False})
        graph, config, _ = await build_task_graph(llm, agent_names, uid)
        record = await resume_task(graph, config, record, answer, stream)
    except Exception as e:
        logging.exception(f"Failed to resume AITask {namespace}/{name}")
        record = fail_record(record, str(e))

    await save_record(store, uid, record)
    try:
        await patch_task_status(namespace, name, status_patch(record))
    except Exception:
        logging.exception(f"Failed to patch the status of AITask {namespace}/{name}")
    return record
