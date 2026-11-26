"""
Follow-up conversations on AITask runs.

A user who can read an AITask can open a chat that continues from the task's
output (websocket ``/v1/ws/messages?aitask=<name>``). Each user gets their own
conversation per AITask run, with a deterministic thread id so reconnecting
reopens it. The follow-up runs as that user (their agents, token and RBAC),
not as the task identity.

The task output is read from the LangGraph store (written by the executor) and
added as the first two messages of the conversation (task prompt and task
output) on the user's first turn. From then on it is a normal chat thread.

A task waiting for human validation can be opened too: the chat shows the
confirmation prompt, and the user's answer resumes the task run (see
``routers/websocket.py``). The answers are part of the seed of the
conversation once the task has finished.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.store.base import BaseStore

from ..constants import AITASK_SEED_KWARG, AITASK_STORE_NAMESPACE, ANNOTATION_VALIDATION_RESPONSE
from .aitask_runner import build_validation_answer
from .rbac import AITASK_STEVE_TYPE, accessible_resource, patch_resource

NAMESPACE = "cattle-ai-agent-system"

AITASK_CHAT_TAG = "aitask"

_FINISHED_PHASES = ("Succeeded", "Failed")


_CONFIRMATION_TAG = "confirmation-response"


class TaskNotFinished(Exception):
    """Raised when a follow-up is requested for an AITask that has no outcome yet."""
    pass


@dataclass
class AITaskChat:
    """An AITask as seen by its follow-up chat."""
    name: str
    uid: str
    task: dict
    record: dict
    # Messages the conversation starts with (empty while the task waits for validation).
    seed_messages: list[BaseMessage] = field(default_factory=list)
    # The record's pendingValidation ({id, message, ...}) if the task waits for validation.
    pending: dict | None = None


def session_token(cookies) -> str:
    """Rancher session token of the caller (``RANCHER_API_TOKEN`` env, for dev, or the ``R_SESS`` cookie)."""
    return os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))


def aitask_chat_id(task_uid: str, user_id: str) -> str:
    """Thread id of *user_id*'s follow-up conversation on the AITask with uid *task_uid*."""
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
    return f"aitask-{task_uid}-{user_hash}"


def build_seed_messages(task: dict, record: dict) -> list[BaseMessage]:
    """
    The messages a follow-up conversation starts with: the task prompt and its
    output (or the error of a failed run).

    They have ids so they are matched (not duplicated) when added to both
    ``messages`` and ``messages_history``.
    """
    metadata = task.get("metadata", {})
    spec = task.get("spec", {})
    task_status = task.get("status") or {}
    name = metadata.get("name", "")
    definition = spec.get("taskDefinitionRef") or name
    prompt = spec.get("prompt") or record.get("prompt") or ""

    if task_status.get("phase") == "Failed" or record.get("phase") == "Failed":
        output = f"The task failed: {record.get('error') or 'unknown error'}"
    else:
        output = record.get("output") or ""

    request_metadata = {
        "agent": "",
        "user_input": prompt,
        "context": {},
        "tags": [AITASK_CHAT_TAG],
        "labels": {"summary": f"Task {definition}: {name}", "aitask": name},
    }
    start_time = task_status.get("startTime") or record.get("startTime")
    completion_time = task_status.get("completionTime") or record.get("completionTime") or start_time

    def seed_kwargs(created_at):
        return {AITASK_SEED_KWARG: True, "created_at": created_at, "request_metadata": request_metadata}

    validation_messages: list[BaseMessage] = []
    for validation in record.get("validations") or []:
        validation_messages += [
            AIMessage(
                id=str(uuid4()),
                content=_validation_request_text(validation.get("message") or ""),
                additional_kwargs=seed_kwargs(validation.get("requestedAt") or start_time),
            ),
            HumanMessage(
                id=str(uuid4()),
                content=validation.get("response") or "",
                additional_kwargs=seed_kwargs(validation.get("answeredAt") or start_time),
            ),
        ]

    return [
        HumanMessage(
            id=str(uuid4()),
            content=prompt,
            additional_kwargs={
                AITASK_SEED_KWARG: True,
                "created_at": start_time,
                "request_metadata": request_metadata,
            },
        ),
        *validation_messages,
        AIMessage(
            id=str(uuid4()),
            content=output,
            additional_kwargs={
                AITASK_SEED_KWARG: True,
                "created_at": completion_time,
                "request_metadata": request_metadata,
            },
        ),
    ]


def _validation_request_text(message: str) -> str:
    """
    Readable text of a validation request for the conversation. The
    ``<confirmation-response>`` tag is removed so the UI doesn't show a live
    confirmation for an answered request.
    """
    payload = message.strip()
    prefix, suffix = f"<{_CONFIRMATION_TAG}>", f"</{_CONFIRMATION_TAG}>"
    if payload.startswith(prefix) and payload.endswith(suffix):
        payload = payload[len(prefix):-len(suffix)]
    try:
        payload = json.dumps(json.loads(payload), indent=2)
    except (json.JSONDecodeError, TypeError):
        pass
    return f"The task asked for validation before running this action:\n```json\n{payload}\n```"


def _pending_validation(task: dict, record: dict) -> dict | None:
    """The record's pending validation, if the AITask is waiting for one."""
    task_status = task.get("status") or {}
    if task_status.get("phase") != "Running" or not task_status.get("pendingValidation"):
        return None
    return record.get("pendingValidation") or None


async def resolve_aitask_chat(token: str, store: BaseStore, name: str) -> AITaskChat:
    """
    Read AITask *name* as the holder of *token* and return what its follow-up
    conversation needs: the seed messages of a finished task, or the pending
    validation of a task waiting for one.

    Raises:
        AccessDenied / NotFound / RBACError: From the Rancher access check.
        TaskNotFinished: If the task is still running (or pending) and not waiting
            for validation (no output to follow up on).
    """
    task = await accessible_resource(token, AITASK_STEVE_TYPE, NAMESPACE, name)
    phase = (task.get("status") or {}).get("phase")
    uid = task.get("metadata", {}).get("uid", "")
    item = await store.aget(AITASK_STORE_NAMESPACE, uid) if uid else None
    record = dict(item.value) if item else {}

    if phase in _FINISHED_PHASES:
        return AITaskChat(name=name, uid=uid, task=task, record=record, seed_messages=build_seed_messages(task, record))
    pending = _pending_validation(task, record)
    if pending:
        return AITaskChat(name=name, uid=uid, task=task, record=record, pending=pending)
    raise TaskNotFinished(f"AITask {name} has not finished yet (phase {phase or 'Pending'})")


async def submit_validation(token: str, name: str, validation_id: str, response: str, user_id: str | None) -> dict:
    """
    Record the answer to the pending validation of AITask *name* on the AITask
    (the validation-response annotation), as the holder of *token*. Rancher checks
    that the user may ``patch`` the AITask, so only those users can approve.

    Returns the answer. Raises AccessDenied / NotFound / RBACError.
    """
    answer = build_validation_answer(validation_id, response, user_id)
    body = {"metadata": {"annotations": {ANNOTATION_VALIDATION_RESPONSE: json.dumps(answer)}}}
    await patch_resource(token, AITASK_STEVE_TYPE, NAMESPACE, name, body)
    return answer
