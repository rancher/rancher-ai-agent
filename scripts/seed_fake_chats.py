"""Seed fake chats into the Postgres checkpointer without calling any LLM.

Runs the real child agent graph (``create_child_agent`` with all its middleware) against the
Postgres checkpointer, with a fake chat model and fake tools in place of the LLM and the MCP
server. LangGraph therefore writes the same ``checkpoints``, ``checkpoint_writes`` and
``checkpoint_blobs`` rows a real chat produces: one checkpoint per superstep, tool-call loops,
``messages_history``, ``token_usage`` and summarization once a chat grows long enough.

Chats are spread over the last ``--days`` by shifting the clock used for message and
token-usage timestamps. Checkpoint ids/``ts`` are set by LangGraph and keep the real time.
UI tools are not selected (they need the UI tools ConfigMap), so there are no ``ui-tools``
usage entries.

Usage:
    DB_CONNECTION_STRING=postgresql://... \\
        uv run python -m scripts.seed_fake_chats --user-id <rancher user id> [--count 50] [--days 14] \\
        [--min-turns 8] [--max-turns 20]
"""

import argparse
import asyncio
import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

TURNS = [
    ("List my clusters", "You have 3 clusters: `local` (Active), `prod-eu` (Active) and `dev-us` (Updating)."),
    ("Why is pod nginx-7d4b9 crashlooping?", "The container exits with code 1: the ConfigMap `nginx-conf` it mounts is missing the `default.conf` key."),
    ("How many nodes does prod-eu have?", "`prod-eu` has 5 nodes: 3 control-plane/etcd and 2 workers, all Ready."),
    ("Show deployments in the default namespace", "There are 4 deployments in `default`: `nginx`, `redis`, `api` and `worker`. All are available."),
    ("Scale the api deployment to 3 replicas", "Done, `api` in `default` is now scaled to 3 replicas."),
    ("What Fleet GitRepos are failing?", "1 GitRepo is not ready: `fleet-default/monitoring` fails with `helm chart not found`."),
    ("Upgrade dev-us to the latest Kubernetes version", "`dev-us` can be upgraded from v1.30.4 to v1.31.2. Should I proceed?"),
    ("Which namespaces use the most memory?", "Top namespaces by memory: `cattle-monitoring-system` (3.1Gi), `kube-system` (1.4Gi), `default` (0.9Gi)."),
    ("Are there any pending PVCs?", "Yes, `data-postgres-0` in `db` is Pending: no StorageClass is set as default."),
    ("Explain what a Rancher project is", "A project groups namespaces of a cluster so RBAC, quotas and policies can be managed together."),
    ("Check certificates expiring soon", "2 certificates expire in less than 30 days: `ingress-tls` in `web` and `webhook-cert` in `cattle-system`."),
    ("Restart the redis deployment", "`redis` in `default` has been restarted; the new pod is Running."),
    ("Show the events of the worker deployment", "Recent events for `worker`:\n\n- `Scheduled` pod worker-5f8c to node w-2\n- `Pulled` image `worker:1.4.2`\n- `BackOff` restarting failed container (x3)\n\nThe container is failing its readiness probe on `/healthz`."),
    ("What does the readiness probe check?", "It does an HTTP GET on port 8080 `/healthz` every 10s with a 1s timeout. The app takes ~3s to answer while warming its cache, so the probe times out."),
    ("Increase the probe timeout to 5 seconds", "Updated `worker`: `readinessProbe.timeoutSeconds` is now 5. The rollout finished and 2/2 pods are Ready."),
    ("Which users have cluster-owner on prod-eu?", "Cluster owners of `prod-eu`: `admin`, `jdoe` and the group `platform-team`."),
    ("Remove jdoe from cluster owners", "This will remove the `cluster-owner` role binding of `jdoe` on `prod-eu`. Do you want to continue?"),
    ("yes", "Done, `jdoe` no longer has the `cluster-owner` role on `prod-eu`."),
    ("Is monitoring installed on dev-us?", "No, the `rancher-monitoring` chart is not installed on `dev-us`. The latest available version is 105.1.0."),
    ("Install it with default values", "`rancher-monitoring` 105.1.0 is being installed in `cattle-monitoring-system`. Prometheus, Grafana and Alertmanager pods are starting."),
    ("Summarize the health of all clusters", "| Cluster | State | Nodes | Alerts |\n|---|---|---|---|\n| local | Active | 1/1 | 0 |\n| prod-eu | Active | 5/5 | 1 warning |\n| dev-us | Active | 3/3 | 0 |\n\nThe warning on `prod-eu` is high disk usage on node `w-1` (87%)."),
    ("What is using disk on w-1?", "Most of the disk on `w-1` is used by container images (41Gi) and the logs of `api` pods (12Gi). Pruning unused images would free ~25Gi."),
    ("Create a namespace called staging in dev-us", "Namespace `staging` created in `dev-us` and added to the `Default` project."),
    ("Add a resource quota of 4 CPUs and 8Gi to it", "ResourceQuota `staging-quota` created: `limits.cpu: 4`, `limits.memory: 8Gi`."),
]
ANSWERS = dict(TURNS)
AGENTS = ["rancher", "fleet", "provisioning"]
MODELS = {"openai": ["gpt-4o", "gpt-4o-mini"], "bedrock": ["anthropic.claude-sonnet"], "ollama": ["qwen3:14b"]}
# Number of tool-call rounds the fake model makes before answering a turn.
TOOL_ROUNDS = [0, 0, 1, 1, 1, 2]


# --- Fake tools (stand-ins for the MCP tools) ---

def _fake_tool_result(kind: str, cluster: str, namespace: str, name: str) -> str:
    return json.dumps({
        "kind": kind,
        "cluster": cluster,
        "namespace": namespace,
        "name": name,
        "status": random.choice(["Active", "Running", "Ready", "Pending"]),
        "items": random.randint(0, 12),
    })


@tool
def listKubernetesResources(kind: str = "Pod", cluster: str = "local", namespace: str = "default") -> str:
    """List Kubernetes resources of a kind in a cluster/namespace."""
    return _fake_tool_result(kind, cluster, namespace, "")


@tool
def getKubernetesResource(kind: str = "Deployment", cluster: str = "local", namespace: str = "default", name: str = "") -> str:
    """Get a Kubernetes resource."""
    return _fake_tool_result(kind, cluster, namespace, name)


@tool
def updateKubernetesResource(kind: str = "Deployment", cluster: str = "local", namespace: str = "default", name: str = "") -> str:
    """Patch a Kubernetes resource."""
    return _fake_tool_result(kind, cluster, namespace, name)


TOOLS = [listKubernetesResources, getKubernetesResource, updateKubernetesResource]


# --- Fake chat model ---

def _text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else json.dumps(message.content)


class FakeAgentModel(BaseChatModel):
    """Answers each turn from ``TURNS``, optionally after a few tool calls, and reports token usage."""

    model: str

    @property
    def _llm_type(self) -> str:
        return "fake-agent"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeAgentModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        human = messages[last_human]
        prompt = _text(human)

        if "Context Extraction Assistant" in prompt:  # SummarizationMiddleware's prompt
            message = AIMessage(content="The user has been inspecting and operating their Rancher clusters "
                                        "(`local`, `prod-eu`, `dev-us`): workloads, nodes, RBAC and charts.")
        else:
            # Decided from the request id so it stays the same across the turn's model calls.
            request_id = human.additional_kwargs.get("request_id", prompt)
            rounds = random.Random(request_id).choice(TOOL_ROUNDS)
            done = sum(1 for m in messages[last_human:] if isinstance(m, AIMessage) and m.tool_calls)
            if done < rounds:
                tool_call = {
                    "name": random.choice(TOOLS).name,
                    "args": {"cluster": random.choice(["local", "prod-eu", "dev-us"]), "namespace": "default"},
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "tool_call",
                }
                message = AIMessage(content="", tool_calls=[tool_call])
            else:
                message = AIMessage(content=ANSWERS.get(prompt, "Done."))

        # Rough token counts: the system prompt and tool schemas plus the whole context.
        input_tokens = 1500 + sum(len(_text(m)) for m in messages) // 4
        output_tokens = len(_text(message)) // 4 + random.randint(10, 80)
        cache_read = random.choice([0, 0, input_tokens // 2])
        message.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
        }
        message.response_metadata = {"model_name": self.model}
        return ChatResult(generations=[ChatGeneration(message=message)])


# --- Shifted clock for message and token-usage timestamps ---

class _Clock:
    offset = timedelta(0)


class _ShiftedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return datetime.now(tz) + _Clock.offset


def _set_clock(when: datetime) -> None:
    _Clock.offset = when - datetime.now(timezone.utc)


def _patch_clocks() -> None:
    """Make the middleware stamp messages and token usage with the shifted clock."""
    from app.services.agent.middleware import inject_kwargs, token_usage

    inject_kwargs.datetime = _ShiftedDatetime  # type: ignore[misc]
    token_usage.datetime = _ShiftedDatetime  # type: ignore[misc]


# --- Seeding ---

def _pick_turns(min_turns: int, max_turns: int) -> list[str]:
    """Pick the prompts of a chat, without repeating any until every turn has been used."""
    n = random.randint(min_turns, max_turns)
    prompts: list[str] = []
    while len(prompts) < n:
        prompts += random.sample([p for p, _ in TURNS], min(n - len(prompts), len(TURNS)))
    return prompts


async def _seed_chat(checkpointer: Any, user_id: str, days: int, min_turns: int, max_turns: int) -> str:
    from app.services.agent.child import create_child_agent
    from app.services.agent.loader import AgentConfig

    agent_name = random.choice(AGENTS)
    provider = random.choice(list(MODELS))
    os.environ["ACTIVE_LLM"] = provider  # read by token_usage._build_entry
    llm = FakeAgentModel(model=random.choice(MODELS[provider]))
    agent_config = AgentConfig(
        name=agent_name,
        displayName=agent_name.capitalize(),
        description=f"Fake {agent_name} agent",
        system_prompt=f"You are the {agent_name} agent.",
        mcp_url="http://fake-mcp",
    )
    graph = create_child_agent(llm, TOOLS, checkpointer, agent_config)

    now = datetime.now(timezone.utc)
    when = now - timedelta(days=random.randint(0, max(days - 1, 0)), minutes=random.randint(0, 24 * 60))
    when = min(when, now - timedelta(hours=2))

    chat_id = str(uuid.uuid4())
    prompts = _pick_turns(min_turns, max_turns)
    for prompt in prompts:
        _set_clock(when)
        request_id = str(uuid.uuid4())
        # Same config and input as routers/websocket.py (_build_config / _build_input_data).
        request_metadata = {
            "agent": agent_name,
            "user_input": prompt,
            "context": {},
            "labels": {},
            "tags": [],
            "ui_tools": {},
        }
        config = {
            "configurable": {
                "thread_id": chat_id,
                "user_id": user_id,
                "request_id": request_id,
                "request_metadata": request_metadata,
                "agent": agent_name,
            },
            "recursion_limit": 50,
        }
        human = HumanMessage(
            content=prompt,
            additional_kwargs={
                "request_id": request_id,
                "request_metadata": request_metadata,
                "created_at": _ShiftedDatetime.now().isoformat(),
            },
        )
        await graph.ainvoke({"messages": [human]}, config=config)
        when += timedelta(minutes=random.randint(1, 20))

    print(f"{chat_id}  {agent_name:<12} {len(prompts):>2} turns  {prompts[0]}")
    return chat_id


async def seed(user_id: str, count: int, days: int, min_turns: int, max_turns: int) -> None:
    os.environ["DB_ENABLED"] = "true"
    # Imported after DB_ENABLED is set so the manager picks Postgres.
    from app.services.memory import create_memory_manager

    _patch_clocks()
    manager = await create_memory_manager()
    try:
        checkpointer = manager.get_checkpointer()
        # Sequential: the shifted clock and ACTIVE_LLM are process-wide.
        for _ in range(count):
            await _seed_chat(checkpointer, user_id, days, min_turns, max_turns)
        print(f"Created {count} chats for user {user_id}")
    finally:
        _Clock.offset = timedelta(0)
        await manager.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed fake chats into the Postgres checkpointer.")
    parser.add_argument("--user-id", required=True, help="Rancher user id that owns the chats")
    parser.add_argument("--count", type=int, default=50, help="Number of chats to create (default: 50)")
    parser.add_argument("--days", type=int, default=14, help="Spread the chats over the last N days (default: 14)")
    parser.add_argument("--min-turns", type=int, default=8, help="Minimum question/answer turns per chat (default: 8)")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum question/answer turns per chat (default: 20)")
    args = parser.parse_args()

    if not os.environ.get("DB_CONNECTION_STRING"):
        parser.error("DB_CONNECTION_STRING must be set")
    if not 1 <= args.min_turns <= args.max_turns:
        parser.error("--min-turns must be >= 1 and <= --max-turns")

    asyncio.run(seed(args.user_id, args.count, args.days, args.min_turns, args.max_turns))


if __name__ == "__main__":
    main()
