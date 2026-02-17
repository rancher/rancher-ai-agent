import os
import pathlib
import pytest
import requests
import re
import warnings
import yaml

from unittest.mock import patch
from typing import Any, Dict, List
from kubernetes import client, config
from fastapi.testclient import TestClient
from app.main import app
from app.services.llm import LLMManager
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

test_client = TestClient(app)

MCP_IMAGE_NAME = "ghcr.io/rancher/rancher-ai-mcp:v0.1.2-alpha.4" 
JUDGE_SYSTEM_PROMPT = """## System Prompt: Semantic Judge

You are a precise Quality Assurance Judge evaluating the output of an AI agent against a Ground Truth reference. 

### Your Task:
Compare the **Actual Response** to the **Expected Reference**. Determine if the Actual Response accurately conveys the same core information, facts, and intent as the Reference.

### Decision Criteria:
Score the similarity from 1 to 10, where:
* **1-3:** Poor match, major contradictions or missing critical facts.
* **4-5:** Partial match but important information is missing or incorrect.
* **6:** Adequate match on core meaning with minor issues.
* **7-8:** Strong match with mostly correct meaning and details.
* **9-10:** Excellent semantic match with equivalent meaning and facts.

* **Ignore:** Differences in tone, word choice, sentence structure, or the presence of polite filler (e.g., "Sure, I can help with that").
* Penalize contradictions, omissions of critical information, and hallucinated facts that change meaning.

### Output Format:
You must output exactly one integer from 1 to 10."""

class MessageTrackingCallback(BaseCallbackHandler):
    """Custom callback handler to track LLM messages."""
    
    def __init__(self):
        self.messages = []
        self.responses = []
    
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List], **kwargs: Any) -> None:
        """Called when chat model starts."""
        self.messages.extend(messages)
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        self.responses.append(response)


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Set up test environment once for all tests in this module."""
    # Initialize Kubernetes client
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    
    v1 = client.CoreV1Api()
    ext_v1 = client.ApiextensionsV1Api()

    # Create CRD from chart
    crd_path = pathlib.Path(__file__).parents[2] / "chart" / "agent" / "templates" / "crds" / "ai.cattle.io_aiagentconfigs.yaml"
    with open(crd_path) as f:
        crd_body = yaml.safe_load(f)
    try:
        ext_v1.create_custom_resource_definition(body=crd_body)
    except client.rest.ApiException as e:
        if e.status != 409:  # Ignore if already exists
            raise

    # Create namespace cattle-ai-agent-system
    namespace_body = client.V1Namespace(
        metadata=client.V1ObjectMeta(name="cattle-ai-agent-system")
    )
    try:
        v1.create_namespace(body=namespace_body)
    except client.rest.ApiException as e:
        if e.status != 409:  # Ignore if already exists
            raise

    # Set up MockMemoryManager
    class MockMemoryManager:
        def get_checkpointer(self):
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    app.memory_manager = MockMemoryManager()
    
    yield
    
    # Cleanup after all tests
    try:
        v1.delete_namespace(name="cattle-ai-agent-system")
    except client.rest.ApiException:
        pass  # Ignore errors during cleanup

    try:
        ext_v1.delete_custom_resource_definition(name="aiagentconfigs.ai.cattle.io")
    except client.rest.ApiException:
        pass  # Ignore errors during cleanup

expect_answer = """<message>A Pod is the smallest and simplest unit in the Kubernetes object model that you create or deploy. It represents a single instance of a running process in your cluster.
  
  Here's a breakdown:
  *   **Encapsulation:** A Pod encapsulates one or more containers (such as Docker containers), storage resources, a unique network IP, and options that govern how the containers should run.
  *   **Shared Context:** Containers within a Pod share the same network namespace, IP address, and storage. This allows them to communicate with each other using `localhost` and share data through mounted volumes.
  *   **Atomic Unit:** Pods are treated as atomic units. When you scale a Deployment, you scale the number of Pods, not individual containers. If a Pod needs to be restarted due to a failure or an update, the entire Pod is recreated.
  *   **Ephemeral:** Pods are designed to be relatively ephemeral. They can be started, stopped, and replaced without affecting the application's overall availability (when managed by higher-level controllers like Deployments).
  
  For more detailed information, you can refer to the official Kubernetes documentation on Pods: [https://kubernetes.io/docs/concepts/workloads/pods/](https://kubernetes.io/docs/concepts/workloads/pods/)
  <suggestion>List all pods in a cluster</suggestion><suggestion>Inspect a specific pod</suggestion><suggestion>What is a Deployment?</suggestion></message>"""

def test_e2e():
    with DockerContainer(MCP_IMAGE_NAME).with_command(["/mcp", "serve" ,"--insecure"]).with_exposed_ports("9092").waiting_for(LogMessageWaitStrategy("MCP Server started!")) as container:
        container.start()
        host_port = container.get_exposed_port("9092")
        host_ip = container.get_container_host_ip()
        service_url = f"{host_ip}:{host_port}"

        print(f"Custom Service is running at: {service_url}")
        os.environ["MCP_URL"] = service_url

        # Set up callback to track LLM messages
        message_tracking_callback = MessageTrackingCallback()
        usage_metadata_callback = UsageMetadataCallbackHandler()
        llm_instance = LLMManager.get_instance()
        
        # Configure LLM to use the callback
        llm_instance.callbacks = [message_tracking_callback, usage_metadata_callback]
        
        try:
            with test_client.websocket_connect("/v1/ws/messages") as websocket:
                # Consume any initial messages from the server (chat-metadata, etc.)
                websocket.receive_text()
                messages = []
                prompts = ["what is a pod?"]
                for prompt in prompts:
                    websocket.send_text(prompt)
                    msg = ""
                    while not msg.endswith("</message>"):
                        msg += websocket.receive_text()
                    messages.append(msg)
                
                #assert message_tracking_callback.messages == [[SystemMessage(content="You are a helpful assistant for Kubernetes-related questions. Answer the user's question based on your knowledge and provide suggestions for follow-up questions.")], [HumanMessage(content="what is a pod?")]]

                score = _assert_llm_as_judge(expected=expect_answer, actual=messages[0], prompt=prompts[0])

                print(f"Final Consumption: {usage_metadata_callback.usage_metadata}")
                
                usage = usage_metadata_callback.usage_metadata
                score_status = "FAIL" if score < 6 else "WARN" if score in (7, 8) else "PASS"
                summary_md = f"""
                ### 📊 LLM Token Consumption Summary
                | Metric | Count |
                | :--- | :--- |
                | **Semantic Judge Score (1-10)** | {score} |
                | **Semantic Judge Status** | {score_status} |
                | **Input Tokens** | {usage[ os.environ["MODEL"] ].get('input_tokens', 0)} |
                | **Output Tokens** | {usage[ os.environ["MODEL"] ].get('output_tokens', 0)} |
                | **Total Tokens** | {usage[ os.environ["MODEL"] ].get('total_tokens', 0)} |
                """

                # Write to GitHub Actions Summary
                if "GITHUB_STEP_SUMMARY" in os.environ:
                    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                        f.write(summary_md)
                                        
        finally:
            # Clear callbacks
            llm_instance.callbacks = []

def _assert_llm_as_judge(expected, actual, prompt):
    llm = LLMManager.get_instance()
    response = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=f"Prompt: {prompt}\n\nResponse 1: {expected}\n\nResponse 2: {actual}\n\nRate semantic equivalence from 1 to 10. Return only a single integer.")
    ]).text.strip()

    match = re.search(r"\b(10|[1-9])\b", response)
    assert match, f"Judge returned invalid score: '{response}'"

    score = int(match.group(1))

    if score in (7, 8):
        warnings.warn(f"Semantic judge returned warning score: {score}", UserWarning)

    assert score >= 6, f"Semantic score too low ({score}). Expected: {expected}, Actual: {actual}"

    return score
