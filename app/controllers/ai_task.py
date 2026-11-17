"""
Kubernetes operator controller for AITask custom resources.

An AITask is one execution of an AITaskDefinition (created by the scheduler).
For each new AITask this controller builds an agent from the agents listed in
the task (a supervisor when there are several, a single agent when there is one)
and invokes it with the task prompt, using the unattended AITask system prompts.

The outcome (report or error) is stored in the LangGraph store under
``AITASK_STORE_NAMESPACE`` keyed by the AITask uid.
"""

import logging
import os
import re
from datetime import datetime, timezone

import kopf
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from psycopg_pool import AsyncConnectionPool

from ..services.agent.loader import AgentConfig, AuthenticationType, load_agent_configs
from ..services.agent.factory import build_agent_from_configs
from ..services.agent.system_prompts import AITASK_RESULT_VALUES
from ..services.llm import LLMManager
from ..services.memory import create_store

AITASK_STORE_NAMESPACE = ("aitasks",)

_FINISHED_PHASES = ("Succeeded", "Failed")
_RESULT_PATTERN = re.compile(r"^\s*\**RESULT:\**\s*\**(" + "|".join(AITASK_RESULT_VALUES) + r")\b", re.IGNORECASE)

# TODO: use a persistent checkpointer once the executor owns its own event loop/DB pool.
_checkpointer = InMemorySaver()

# Store and pool owned by the kopf event loop (see on_startup).
_store: BaseStore | None = None
_db_pool: AsyncConnectionPool | None = None


@kopf.on.startup()
async def on_startup(logger, **kwargs):
    """Create the LangGraph store on the kopf event loop (the FastAPI pool can't be shared across loops)."""
    global _store, _db_pool
    if os.environ.get("DB_ENABLED", "false").lower() == "true":
        _db_pool = AsyncConnectionPool(
            conninfo=os.environ.get("DB_CONNECTION_STRING", ""),
            max_size=5,
            kwargs={"autocommit": True},
            open=False,
        )
        await _db_pool.open()
    _store = await create_store(_db_pool)
    logger.info(f"AITask store initialized ({'postgres' if _db_pool else 'in-memory'})")


@kopf.on.cleanup()
async def on_cleanup(**kwargs):
    global _store, _db_pool
    if _db_pool:
        await _db_pool.close()
    _db_pool = None
    _store = None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_agent_configs(agent_names: list[str]) -> list[AgentConfig]:
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


def parse_result(output: str) -> str | None:
    """Return the Healthy/Warning/Critical verdict from the last RESULT line of *output*, if any."""
    for line in reversed(output.strip().splitlines()):
        match = _RESULT_PATTERN.match(line)
        if match:
            return match.group(1).capitalize()
        if line.strip():
            return None
    return None


def _interrupt_message(interrupts) -> str:
    value = interrupts[0].value
    if isinstance(value, dict):
        return str(value.get("message", ""))
    return str(value)


async def _save_result(uid: str, value: dict, logger) -> None:
    if _store is None:
        logger.error("AITask store is not initialized, result not stored")
        return
    try:
        await _store.aput(AITASK_STORE_NAMESPACE, uid, value)
    except Exception:
        logger.exception(f"Failed to store result of AITask {value.get('name')}")


@kopf.on.resume('ai.cattle.io', 'v1alpha1', 'aitasks', timeout=3600.0)
@kopf.on.create('ai.cattle.io', 'v1alpha1', 'aitasks', timeout=3600.0)
async def run_ai_task(spec, status, name, namespace, uid, logger, patch, **kwargs):
    """
    Execute an AITask: build a supervisor (or single agent) from spec.agents,
    call it with spec.prompt and store the outcome in the LangGraph store.
    """
    phase = (status or {}).get('phase')
    if phase in _FINISHED_PHASES or phase == 'Running':
        # Already handled (e.g. operator restart); do not run twice.
        logger.info(f"AITask {namespace}/{name} already in phase {phase}, skipping")
        return

    agent_names = list(spec.get('agents') or [])
    record = {
        "name": name,
        "taskDefinitionRef": spec.get('taskDefinitionRef'),
        "prompt": spec['prompt'],
        "agents": agent_names,
        "phase": "Running",
        "result": None,
        "output": None,
        "error": None,
        "startTime": _now(),
        "completionTime": None,
    }
    patch.status['phase'] = 'Running'
    patch.status['startTime'] = record["startTime"]

    try:
        configs = _resolve_agent_configs(agent_names)
        if not configs:
            raise RuntimeError("No usable agents available for this task")

        # One agent -> single agent, several -> supervisor (handled by the factory).
        llm = LLMManager.get_instance()
        graph, agents_metadata = await build_agent_from_configs(llm, configs, _checkpointer, allow_planner=False)

        configurable = {"thread_id": f"aitask-{uid}", "user_id": "aitask", "aitask": True}
        if len(configs) == 1:
            # Direct child call: identity_preamble_middleware adds the Liz identity.
            configurable["agent"] = configs[0].name
        config = {"configurable": configurable}

        logger.info(f"AITask {namespace}/{name}: running with agents {[a['name'] for a in agents_metadata]}")
        result = await graph.ainvoke({"messages": [HumanMessage(content=record["prompt"])]}, config)

        state = await graph.aget_state(config)
        if state and state.interrupts:
            raise RuntimeError(
                "Tool requires human validation, which is not available for AITasks: "
                f"{_interrupt_message(state.interrupts)}"
            )

        messages = result.get("messages") or []
        output = str(messages[-1].text) if messages else ""
        record["output"] = output
        record["result"] = parse_result(output)
        record["phase"] = "Succeeded"
        logger.info(f"AITask {namespace}/{name} succeeded with result {record['result']}")
    except Exception as e:
        logger.exception(f"AITask {namespace}/{name} failed")
        record["phase"] = "Failed"
        record["error"] = str(e)

    record["completionTime"] = _now()
    await _save_result(uid, record, logger)

    patch.status['phase'] = record["phase"]
    patch.status['completionTime'] = record["completionTime"]
    if record["result"]:
        patch.status['result'] = record["result"]
