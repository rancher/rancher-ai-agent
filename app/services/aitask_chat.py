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
"""

import hashlib
import os
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.store.base import BaseStore

from ..constants import AITASK_SEED_KWARG, AITASK_STORE_NAMESPACE
from .rbac import AITASK_STEVE_TYPE, accessible_resource

NAMESPACE = "cattle-ai-agent-system"

AITASK_CHAT_TAG = "aitask"

_FINISHED_PHASES = ("Succeeded", "Failed")


class TaskNotFinished(Exception):
    """Raised when a follow-up is requested for an AITask that has no outcome yet."""
    pass


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


async def resolve_aitask_chat(token: str, store: BaseStore, name: str) -> tuple[str, list[BaseMessage]]:
    """
    Read AITask *name* as the holder of *token* and return its uid and the seed
    messages of a follow-up conversation.

    Raises:
        AccessDenied / NotFound / RBACError: From the Rancher access check.
        TaskNotFinished: If the task has not finished yet (no output to follow up on).
    """
    task = await accessible_resource(token, AITASK_STEVE_TYPE, NAMESPACE, name)
    phase = (task.get("status") or {}).get("phase")
    if phase not in _FINISHED_PHASES:
        raise TaskNotFinished(f"AITask {name} has not finished yet (phase {phase or 'Pending'})")

    uid = task.get("metadata", {}).get("uid", "")
    item = await store.aget(AITASK_STORE_NAMESPACE, uid) if uid else None
    record = item.value if item else {}
    return uid, build_seed_messages(task, record)
