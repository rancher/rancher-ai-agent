"""Shared fixtures for all e2e tests.

Fixtures are session-scoped so the Kubernetes environment and MCP container
are set up once and reused across every test file in the e2e directory.
"""

import os
import pathlib
import pytest
import yaml

from kubernetes import client, config
from fastapi.testclient import TestClient
from app.main import app
from app.services.llm import LLMManager
from langchain_core.callbacks import UsageMetadataCallbackHandler
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from tests.e2e.helpers import (
    MessageTrackingCallback,
    MCP_IMAGE_NAME,
    create_k8s_resources,
    delete_k8s_resources,
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up Kubernetes resources and mock memory manager for all e2e tests."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    v1 = client.CoreV1Api()
    ext_v1 = client.ApiextensionsV1Api()

    # Create CRD from chart
    crd_path = (
        pathlib.Path(__file__).parents[2]
        / "chart" / "agent" / "templates" / "crds" / "ai.cattle.io_aiagentconfigs.yaml"
    )
    with open(crd_path) as f:
        crd_body = yaml.safe_load(f)
    try:
        ext_v1.create_custom_resource_definition(body=crd_body)
    except client.rest.ApiException as e:
        if e.status != 409:  # Ignore if already exists
            raise

    # Create namespace
    namespace_body = client.V1Namespace(
        metadata=client.V1ObjectMeta(name="cattle-ai-agent-system")
    )
    try:
        v1.create_namespace(body=namespace_body)
    except client.rest.ApiException as e:
        if e.status != 409:  # Ignore if already exists
            raise

    # Set up MockMemoryManager
    class MockStorageType:
        value = "memory"

    class MockMemoryManager:
        storage_type = MockStorageType()

        def get_checkpointer(self):
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    app.memory_manager = MockMemoryManager()

    yield

    # Cleanup after all tests
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
        .with_command(["/usr/local/bin/mcp", "serve", "--insecure"]) \
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


@pytest.fixture()
def k8s_resources(request):
    """
    Per-test fixture that creates Kubernetes resources declared in the test case's
    ``resources`` field and tears them down after the test completes.

    The fixture reads from the ``test_case`` parameter (accessed via ``request.node``).
    If the test case has no resources, this is a no-op.

    Usage – just add ``k8s_resources`` to your test signature:

        def test_example(agent_test_session, test_client, test_case, k8s_resources):
            ...

    Resources are defined as YAML strings in the test case dataclass:

        E2ETestCase(
            id="list_pods",
            prompt="list pods in namespace e2e-test",
            expected="...",
            resources=[
                '''
                apiVersion: v1
                kind: Namespace
                metadata:
                  name: e2e-test
                ''',
                '''
                apiVersion: v1
                kind: Pod
                metadata:
                  name: nginx
                  namespace: e2e-test
                spec:
                  containers:
                    - name: nginx
                      image: nginx:latest
                ''',
            ],
        )
    """
    # Retrieve test_case from the parametrize marker
    test_case = None
    for marker in request.node.iter_markers("parametrize"):
        # Look for test_case in the current call's params
        if "test_case" in request.node.callspec.params:
            test_case = request.node.callspec.params["test_case"]
            break

    resources = getattr(test_case, "resources", []) if test_case else []

    if not resources:
        yield []
        return

    created = create_k8s_resources(resources)
    yield created
    delete_k8s_resources(created)


def _write_usage_summary(callback: UsageMetadataCallbackHandler):
    """Write token usage summary to stdout and optionally to GitHub Actions step summary."""
    usage = callback.usage_metadata
    if not usage:
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
