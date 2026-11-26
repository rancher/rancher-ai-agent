"""
Kubernetes operator controller for AITask custom resources.

An AITask is one execution of an AITaskDefinition (created by the scheduler).
For each new AITask this controller builds an agent from the agents listed in
the task (a supervisor when there are several, a single agent when there is one)
and invokes it with the task prompt, using the unattended AITask system prompts.
The agents call their MCP servers with the Rancher token of the Secret referenced
by ``spec.credentialsRef`` (``RANCHER_API_TOKEN`` wins, for testing).

The outcome (report or error) is stored in the LangGraph store under
``AITASK_STORE_NAMESPACE`` keyed by the AITask uid.

Human validation: a run that reaches a tool needing confirmation is paused
(``status.pendingValidation=true``, phase stays ``Running``). It is resumed with
the user's answer, given in the follow-up chat (resumed by the websocket) or with
the ``ai.cattle.io/validation-response`` annotation (resumed here). A run still
paused when the agent restarts is failed: its state was only in memory.
The running logic is shared with the websocket in ``services/aitask_runner.py``.
"""

import os

import aiosqlite
import kopf
from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore
from psycopg_pool import AsyncConnectionPool

from ..services.aitask_runner import (
    LOST_RUN_ERROR,
    build_task_graph,
    claim_validation,
    continue_task,
    fail_record,
    finish_run,
    load_record,
    new_record,
    now,
    parse_result,  # noqa: F401 (re-exported)
    parse_validation_answer,
    save_record,
    status_patch,
    task_token,
)
from ..services.llm import LLMManager
from ..services.memory import create_sqlite_store, create_store, postgres_enabled, sqlite_store_enabled, sqlite_store_path
from ..constants import AITASK_STORE_NAMESPACE, ANNOTATION_VALIDATION_RESPONSE, VALIDATION_VIA_CHAT  # noqa: F401

_FINISHED_PHASES = ("Succeeded", "Failed")

# Store and pool owned by the kopf event loop (see on_startup).
_store: BaseStore | None = None
_db_pool: AsyncConnectionPool | None = None
_sqlite_conn: aiosqlite.Connection | None = None


@kopf.on.startup()
async def on_startup(logger, **kwargs):
    """Create the LangGraph store on the kopf event loop (the FastAPI pool can't be shared across loops)."""
    global _store, _db_pool, _sqlite_conn
    if postgres_enabled():
        _db_pool = AsyncConnectionPool(
            conninfo=os.environ.get("DB_CONNECTION_STRING", ""),
            max_size=5,
            kwargs={"autocommit": True},
            open=False,
        )
        await _db_pool.open()
        _store = await create_store(_db_pool)
        backend = "postgres"
    elif sqlite_store_enabled():
        _store, _sqlite_conn = await create_sqlite_store(sqlite_store_path())
        backend = "sqlite"
    else:
        _store = await create_store(None)
        backend = "in-memory"
    logger.info(f"AITask store initialized ({backend})")


@kopf.on.cleanup()
async def on_cleanup(**kwargs):
    global _store, _db_pool, _sqlite_conn
    if _db_pool:
        await _db_pool.close()
    if _sqlite_conn:
        await _sqlite_conn.close()
    _db_pool = None
    _sqlite_conn = None
    _store = None


def _now() -> str:
    return now()


async def _save_result(uid: str, value: dict, logger) -> None:
    await save_record(_store, uid, value)


@kopf.on.resume('ai.cattle.io', 'v1alpha1', 'aitasks', timeout=3600.0)
@kopf.on.create('ai.cattle.io', 'v1alpha1', 'aitasks', timeout=3600.0)
async def run_ai_task(spec, status, name, namespace, uid, logger, patch, **kwargs):
    """
    Execute an AITask: build a supervisor (or single agent) from spec.agents,
    call it with spec.prompt and store the outcome in the LangGraph store.

    A run that needs human validation is paused (``status.pendingValidation``).
    """
    status = status or {}
    phase = status.get('phase')
    if phase == 'Running' and status.get('pendingValidation'):
        # The paused run lived in the in-memory checkpointer, lost on restart.
        logger.warning(f"AITask {namespace}/{name} was waiting for validation when the agent restarted, failing it")
        record = (await load_record(_store, uid) if _store else None) or new_record(name, spec)
        fail_record(record, LOST_RUN_ERROR)
        await _save_result(uid, record, logger)
        patch.status.update(status_patch(record))
        return
    if phase in _FINISHED_PHASES or phase == 'Running':
        # Already handled (e.g. operator restart); do not run twice.
        logger.info(f"AITask {namespace}/{name} already in phase {phase}, skipping")
        return

    record = new_record(name, spec)
    patch.status['phase'] = 'Running'
    patch.status['startTime'] = record["startTime"]

    try:
        token = await task_token(namespace, spec)
        graph, config, agents_metadata = await build_task_graph(LLMManager.get_instance(), record["agents"], uid, token)
        logger.info(f"AITask {namespace}/{name}: running with agents {[a['name'] for a in agents_metadata]}")
        result = await graph.ainvoke({"messages": [HumanMessage(content=record["prompt"])]}, config)
        await finish_run(graph, config, record, result)
        if record["pendingValidation"]:
            logger.info(f"AITask {namespace}/{name} is waiting for human validation")
        else:
            logger.info(f"AITask {namespace}/{name} succeeded with result {record['result']}")
    except Exception as e:
        logger.exception(f"AITask {namespace}/{name} failed")
        fail_record(record, str(e))

    await _save_result(uid, record, logger)
    patch.status.update(status_patch(record))


@kopf.on.field('ai.cattle.io', 'v1alpha1', 'aitasks',
               field=('metadata', 'annotations', ANNOTATION_VALIDATION_RESPONSE), timeout=3600.0)
async def answer_validation(new, spec, status, name, namespace, uid, logger, **kwargs):
    """
    Resume a run waiting for human validation when the validation-response
    annotation is set (e.g. with kubectl). Answers given in the follow-up chat
    (``via: chat``) are resumed by the websocket and ignored here.
    """
    answer = parse_validation_answer(new)
    if answer is None:
        if new:
            logger.warning(f"AITask {namespace}/{name}: invalid {ANNOTATION_VALIDATION_RESPONSE} annotation, ignored")
        return
    if answer.get("via") == VALIDATION_VIA_CHAT:
        logger.debug(f"AITask {namespace}/{name}: validation answered in the chat, resumed there")
        return
    if not (status or {}).get('pendingValidation'):
        logger.info(f"AITask {namespace}/{name} is not waiting for validation, answer ignored")
        return

    record = await load_record(_store, uid) if _store else None
    pending = (record or {}).get("pendingValidation")
    if not pending:
        logger.warning(f"AITask {namespace}/{name}: no pending validation in the store, answer ignored")
        return
    if answer.get("validationId") and answer["validationId"] != pending.get("id"):
        logger.warning(f"AITask {namespace}/{name}: answer to a previous validation ignored")
        return
    if not claim_validation(pending["id"]):
        logger.info(f"AITask {namespace}/{name}: validation already answered")
        return

    logger.info(f"AITask {namespace}/{name}: resuming with validation answer '{answer['response']}'")
    record = await continue_task(
        _store, LLMManager.get_instance(), namespace, name, uid,
        dict(spec), record, answer,
    )
    logger.info(f"AITask {namespace}/{name} resumed, phase {record['phase']}")
