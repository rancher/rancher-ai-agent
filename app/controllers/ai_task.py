"""
Kubernetes operator controller for AITask custom resources.

An AITask is one execution of an AITaskDefinition (created by the scheduler).
For each new AITask this controller builds an agent from the agents listed in
the task (a supervisor when there are several, a single agent when there is one)
and invokes it with the task prompt.
"""

import logging
from datetime import datetime, timezone

import kopf
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ..services.agent.loader import AgentConfig, AuthenticationType, load_agent_configs
from ..services.agent.factory import build_agent_from_configs
from ..services.llm import LLMManager

_FINISHED_PHASES = ("Succeeded", "Failed")

# TODO: use a persistent checkpointer once the executor owns its own event loop/DB pool.
_checkpointer = InMemorySaver()


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


@kopf.on.resume('ai.cattle.io', 'v1alpha1', 'aitasks', timeout=3600.0)
@kopf.on.create('ai.cattle.io', 'v1alpha1', 'aitasks', timeout=3600.0)
async def run_ai_task(spec, status, name, namespace, uid, logger, patch, **kwargs):
    """
    Execute an AITask: build a supervisor (or single agent) from spec.agents and
    call it with spec.prompt.
    """
    phase = (status or {}).get('phase')
    if phase in _FINISHED_PHASES or phase == 'Running':
        # Already handled (e.g. operator restart); do not run twice.
        logger.info(f"AITask {namespace}/{name} already in phase {phase}, skipping")
        return

    prompt = spec['prompt']
    thread_id = f"aitask-{uid}"
    patch.status['phase'] = 'Running'
    patch.status['startTime'] = _now()

    try:
        configs = _resolve_agent_configs(spec.get('agents') or [])
        if not configs:
            raise RuntimeError("No usable agents available for this task")

        # One agent -> single agent, several -> supervisor (handled by the factory).
        llm = LLMManager.get_instance()
        graph, agents_metadata = await build_agent_from_configs(llm, configs, _checkpointer)

        logger.info(f"AITask {namespace}/{name}: running with agents {[a['name'] for a in agents_metadata]}")
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            {"configurable": {"thread_id": thread_id, "user_id": "aitask"}},
        )
        output = result["messages"][-1].content if result.get("messages") else result

        # TODO: persist findings/result, set phase Succeeded, enforce limits and handle interrupts.
        logger.info(f"AITask {namespace}/{name} outcome: {output}")
        patch.status['phase'] = 'Succeeded'
        patch.status['completionTime'] = _now()
    except Exception:
        logger.exception(f"AITask {namespace}/{name} failed")
        patch.status['phase'] = 'Failed'
        patch.status['completionTime'] = _now()
