"""Shared fixtures and helpers for all e2e tests.

Fixtures are session-scoped so the Kubernetes environment and MCP container
are set up once and reused across every test file in the e2e directory.
"""

import json
import logging
import os
import pathlib
import re
import yaml

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import AmazonBedrockModel
from kubernetes import client, dynamic
from fastapi.testclient import TestClient
from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from app.main import app
from app.services.llm import LLMManager

logger = logging.getLogger(__name__)

MCP_IMAGE_NAME = os.environ.get("MCP_IMAGE_NAME", "ghcr.io/rancher/rancher-ai-mcp:v1.0.0")


# ─── Test Case Data Classes ──────────────────────────────────────────────────


@dataclass
class E2ETestCase:
    """Defines a single-turn e2e test case for LLM-as-judge evaluation."""
    id: str
    prompt: str
    expected: str
    description: str = ""
    resources: List[str] = field(default_factory=list)
    expected_agent: Optional[str] = None


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation test."""
    prompt: str
    expected: Optional[str] = None
    expected_confirmation_message: Optional[str] = None
    expected_agent: Optional[str] = None


@dataclass
class MultiTurnTestCase:
    """Defines a multi-turn conversation e2e test case."""
    id: str
    turns: List[ConversationTurn]
    description: str = ""
    resources: List[str] = field(default_factory=list)


@dataclass
class PromptResult:
    """Result from a single prompt interaction, including metadata for assertions."""
    response: str
    thread_id: str
    agent_name: Optional[str] = None
    selection_mode: Optional[str] = None


# ─── Callback Handlers ───────────────────────────────────────────────────────


class MessageTrackingCallback(BaseCallbackHandler):
    """Custom callback handler to track LLM messages and responses."""

    def __init__(self):
        self.messages = []
        self.responses = []

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List], **kwargs: Any
    ) -> None:
        self.messages.extend(messages)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        self.responses.append(response)


# ─── WebSocket Helpers ────────────────────────────────────────────────────────


def ws_send_and_receive(websocket, prompt: str) -> str:
    """Send a prompt over the WebSocket and collect the full response."""
    websocket.send_text(prompt)
    msg = ""
    while not msg.endswith("</message>"):
        msg += websocket.receive_text()
    return msg


def ws_send_json_and_receive(
    websocket, prompt: str, context: dict = None, agent: str = ""
) -> str:
    """Send a JSON-formatted prompt with optional context and collect the response."""
    payload: dict = {"prompt": prompt}
    if context:
        payload["context"] = context
    if agent:
        payload["agent"] = agent
    return ws_send_and_receive(websocket, json.dumps(payload))


# ─── Metadata Parsers ────────────────────────────────────────────────────────


def parse_chat_metadata(text: str) -> dict:
    """Extract chat metadata from a <chat-metadata>...</chat-metadata> tag."""
    match = re.search(r"<chat-metadata>(.*?)</chat-metadata>", text)
    if not match:
        return {}
    return json.loads(match.group(1))


def parse_agent_metadata(text: str) -> dict:
    """Extract agent metadata from an <agent-metadata>...</agent-metadata> tag."""
    match = re.search(r"<agent-metadata>(.*?)</agent-metadata>", text)
    if not match:
        return {}
    return json.loads(match.group(1))


# ─── High-Level Test Helpers ─────────────────────────────────────────────────


def run_single_prompt(test_client, prompt: str) -> PromptResult:
    """
    Opens a WebSocket, sends one prompt, and returns a PromptResult.
    Each call creates a new conversation thread.
    """
    with test_client.websocket_connect("/v1/ws/messages") as websocket:
        chat_meta_text = websocket.receive_text()
        chat_meta = parse_chat_metadata(chat_meta_text)
        thread_id = chat_meta.get("chatId", "")

        response = ws_send_and_receive(websocket, prompt)
        agent_meta = parse_agent_metadata(response)

        return PromptResult(
            response=response,
            thread_id=thread_id,
            agent_name=agent_meta.get("agentName"),
            selection_mode=agent_meta.get("selectionMode"),
        )


def run_multi_turn(test_client, prompts: List[str]) -> List[PromptResult]:
    """
    Opens a single WebSocket session, sends multiple prompts in sequence,
    and returns a list of PromptResult objects sharing the same thread_id.
    """
    results = []
    with test_client.websocket_connect("/v1/ws/messages") as websocket:
        chat_meta_text = websocket.receive_text()
        chat_meta = parse_chat_metadata(chat_meta_text)
        thread_id = chat_meta.get("chatId", "")

        for prompt in prompts:
            response = ws_send_and_receive(websocket, prompt)
            agent_meta = parse_agent_metadata(response)

            results.append(PromptResult(
                response=response,
                thread_id=thread_id,
                agent_name=agent_meta.get("agentName"),
                selection_mode=agent_meta.get("selectionMode"),
            ))
    return results


def run_prompt_with_context(
    test_client, prompt: str, context: dict = None, agent: str = ""
) -> str:
    """
    Opens a WebSocket, sends a JSON prompt with context, and returns the response.
    """
    with test_client.websocket_connect("/v1/ws/messages") as websocket:
        websocket.receive_text()  # consume chat-metadata
        return ws_send_json_and_receive(websocket, prompt, context, agent)


# ─── Evaluation Helpers ──────────────────────────────────────────────────────


def evaluate_and_assert(test_cases: list, e2e_results: list):
    """Run deepeval GEval correctness evaluation and assert all cases pass."""
    metric = GEval(
        name="Correctness",
        criteria=(
            "Determine if the 'actual output' conveys the same fundamental meaning "
            "and key concepts as the 'expected output'. The wording, phrasing, "
            "structure, and level of detail may differ — focus only on whether the "
            "core factual content is semantically equivalent. Ignore stylistic "
            "differences, additional context, suggestions, and agent metadata like "
            "agent selection messages."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=AmazonBedrockModel(model="eu.anthropic.claude-opus-4-5-20251101-v1:0"),
    )
    results = evaluate(test_cases, metrics=[metric])
    e2e_results.append(results)
    failed = [r for r in results.test_results if not r.success]
    assert not failed, f"{len(failed)} test case(s) failed evaluation"


# ─── Kubernetes Resource Helpers ──────────────────────────────────────────────


def create_k8s_resources(resource_yamls: List[str]) -> List[dict]:
    """Create Kubernetes resources from a list of YAML strings."""
    dyn = dynamic.DynamicClient(client.ApiClient())
    created: List[dict] = []

    for resource_yaml in resource_yamls:
        body = yaml.safe_load(resource_yaml)
        api_version = body.get("apiVersion", "v1")
        kind = body["kind"]
        metadata = body.get("metadata", {})
        namespace = metadata.get("namespace")

        api = dyn.resources.get(api_version=api_version, kind=kind)
        try:
            if namespace:
                api.create(body=body, namespace=namespace)
            else:
                api.create(body=body)
            logger.info(f"Created {kind} '{metadata.get('name')}'"
                        f"{f' in namespace {namespace}' if namespace else ''}")
            created.append(body)
        except client.rest.ApiException as e:
            if e.status == 409:
                logger.info(f"{kind} '{metadata.get('name')}' already exists, skipping")
                created.append(body)
            else:
                raise

    return created


def delete_k8s_resources(resource_dicts: List[dict]) -> None:
    """Delete Kubernetes resources in reverse order (LIFO)."""
    dyn = dynamic.DynamicClient(client.ApiClient())

    for body in reversed(resource_dicts):
        api_version = body.get("apiVersion", "v1")
        kind = body["kind"]
        metadata = body.get("metadata", {})
        name = metadata.get("name")
        namespace = metadata.get("namespace")

        try:
            api = dyn.resources.get(api_version=api_version, kind=kind)
            if namespace:
                api.delete(name=name, namespace=namespace)
            else:
                api.delete(name=name)
            logger.info(f"Deleted {kind} '{name}'"
                        f"{f' from namespace {namespace}' if namespace else ''}")
        except client.rest.ApiException:
            logger.warning(f"Failed to delete {kind} '{name}', ignoring")


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up Kubernetes resources and mock memory manager for all e2e tests."""
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()

    v1 = client.CoreV1Api()
    ext_v1 = client.ApiextensionsV1Api()

    crd_path = (
        pathlib.Path(__file__).parents[2]
        / "chart" / "agent" / "templates" / "crds" / "ai.cattle.io_aiagentconfigs.yaml"
    )
    with open(crd_path) as f:
        crd_body = yaml.safe_load(f)
    try:
        ext_v1.create_custom_resource_definition(body=crd_body)
    except client.rest.ApiException as e:
        if e.status != 409:
            raise

    namespace_body = client.V1Namespace(
        metadata=client.V1ObjectMeta(name="cattle-ai-agent-system")
    )
    try:
        v1.create_namespace(body=namespace_body)
    except client.rest.ApiException as e:
        if e.status != 409:
            raise

    class MockStorageType:
        value = "memory"

    class MockMemoryManager:
        storage_type = MockStorageType()
        _checkpointer = MemorySaver()

        def get_checkpointer(self):
            return self._checkpointer

    app.memory_manager = MockMemoryManager()

    yield

    try:
        v1.delete_namespace(name="cattle-ai-agent-system")
    except client.rest.ApiException:
        pass

    try:
        ext_v1.delete_custom_resource_definition(name="aiagentconfigs.ai.cattle.io")
    except client.rest.ApiException:
        pass


@pytest.fixture(scope="session")
def test_client():
    """Provides a FastAPI TestClient for e2e tests."""
    return TestClient(app)


@pytest.fixture(scope="session")
def agent_test_session():
    """
    Starts the MCP container and configures the LLM with tracking callbacks.
    Session-scoped so the container and LLM instance are reused across all e2e tests.
    """
    with DockerContainer(MCP_IMAGE_NAME) \
        .with_command(["/usr/local/bin/mcp", "serve", "--insecure", "--log-level", "debug"]) \
        .with_exposed_ports("9092") \
        .waiting_for(LogMessageWaitStrategy("MCP Server started!")) as container:

        container.start()
        host_port = container.get_exposed_port("9092")
        host_ip = container.get_container_host_ip()
        os.environ["MCP_URL"] = f"{host_ip}:{host_port}"
        os.environ["INSECURE_SKIP_TLS"] = "true"

        message_tracking_callback = MessageTrackingCallback()
        usage_metadata_callback = UsageMetadataCallbackHandler()
        llm_instance = LLMManager.get_instance()
        llm_instance.callbacks = [message_tracking_callback, usage_metadata_callback]

        yield {
            "message_tracking_callback": message_tracking_callback,
            "usage_metadata_callback": usage_metadata_callback,
            "llm_instance": llm_instance,
        }

        llm_instance.callbacks = []
        _write_usage_summary(usage_metadata_callback)

        print("\n--- MCP Container Logs ---")
        stdout, stderr = container.get_logs()
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
        if stderr:
            print(stderr.decode("utf-8", errors="replace"))
        print("--- End MCP Container Logs ---")


@pytest.fixture()
def k8s_resources(request):
    """
    Per-test fixture that creates Kubernetes resources declared in the test case's
    ``resources`` field and tears them down after the test completes.
    """
    test_case = request.node.callspec.params.get("test_case")
    resources = getattr(test_case, "resources", []) if test_case else []

    if not resources:
        yield []
        return

    created = create_k8s_resources(resources)
    yield created
    delete_k8s_resources(created)


@pytest.fixture(scope="session")
def e2e_results():
    """Collects DeepEval results across all e2e tests, written once at session end."""
    results = []
    yield results
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file and results:
        with open(summary_file, "a") as f:
            f.write("## E2E tests summary \n\n")
            f.write("| Test Case | Metric | Score | Reason | Status |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for entry in results:
                for result in entry.test_results:
                    for metric in result.metrics_data:
                        score = metric.score if metric.score is not None else 0.0
                        reason = metric.reason or "N/A"
                        status = "✅ PASS" if score >= metric.threshold else "❌ FAIL"
                        f.write(f"| {result.name} - {result.input[:20]}... | {metric.name} | {score:.2f} | {reason} | {status} |\n")


def _write_usage_summary(callback: UsageMetadataCallbackHandler):
    """Write token usage summary to stdout and optionally to GitHub Actions step summary."""
    usage = callback.usage_metadata
    if not usage:
        print("No usage metadata collected.")
        return

    rows = []
    for model, data in usage.items():
        rows.append(
            f"| **{model}** | {data.get('input_tokens', 0)} | "
            f"{data.get('output_tokens', 0)} | {data.get('total_tokens', 0)} |"
        )

    summary_md = (
        "### 📊 LLM Token Consumption Summary (All E2E Tests)\n"
        "| Model | Input Tokens | Output Tokens | Total Tokens |\n"
        "| :--- | :--- | :--- | :--- |\n"
        + "\n".join(rows)
        + "\n"
    )

    print(summary_md)

    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary_md)
