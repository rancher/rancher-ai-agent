"""Reusable utilities for e2e tests.

This module provides helper classes, functions, and constants that are shared
across all e2e test files. To add a new test case, use the E2ETestCase or
MultiTurnTestCase dataclasses.
"""

import json
import logging
import re
import warnings
import yaml

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from kubernetes import client as k8s_client
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm import LLMManager

# TODO: get from chart!
MCP_IMAGE_NAME = "ghcr.io/rancher/rancher-ai-mcp:v1.0.0"

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


# ─── Test Case Data Classes ──────────────────────────────────────────────────


@dataclass
class E2ETestCase:
    """Defines a single-turn e2e test case for LLM-as-judge evaluation.

    Attributes:
        id: Unique identifier used as the pytest test ID.
        prompt: The user prompt to send via WebSocket.
        expected: Reference answer for semantic comparison.
        description: Human-readable description of what the test validates.
        min_score: Minimum acceptable semantic similarity score (1-10, default 6).
        resources: Optional list of Kubernetes resource YAML strings to create before
                   the test and delete after.
    """
    id: str
    prompt: str
    expected: str
    description: str = ""
    min_score: int = 6
    resources: List[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation test.

    Attributes:
        prompt: The user prompt for this turn.
        expected: Reference answer for semantic comparison.
        min_score: Minimum acceptable semantic similarity score (1-10, default 6).
    """
    prompt: str
    expected: str
    min_score: int = 6


@dataclass
class MultiTurnTestCase:
    """Defines a multi-turn conversation e2e test case.

    Attributes:
        id: Unique identifier used as the pytest test ID.
        turns: Ordered list of conversation turns to execute.
        description: Human-readable description of what the test validates.
        resources: Optional list of Kubernetes resource YAML strings to create before
                   the test and delete after.
    """
    id: str
    turns: List[ConversationTurn]
    description: str = ""
    resources: List[str] = field(default_factory=list)


# ─── Callback Handlers ───────────────────────────────────────────────────────


class MessageTrackingCallback(BaseCallbackHandler):
    """Custom callback handler to track LLM messages and responses."""

    def __init__(self):
        self.messages = []
        self.responses = []

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List], **kwargs: Any
    ) -> None:
        """Called when chat model starts."""
        self.messages.extend(messages)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        self.responses.append(response)


# ─── WebSocket Helpers ────────────────────────────────────────────────────────


def ws_send_and_receive(websocket, prompt: str) -> str:
    """
    Sends a prompt over the WebSocket and collects the full response,
    reading until the closing </message> tag.
    """
    websocket.send_text(prompt)
    msg = ""
    while not msg.endswith("</message>"):
        msg += websocket.receive_text()
    return msg


def ws_send_json_and_receive(
    websocket, prompt: str, context: dict = None, agent: str = ""
) -> str:
    """
    Sends a JSON-formatted prompt with optional context over the WebSocket
    and collects the full response.
    """
    payload: dict = {"prompt": prompt}
    if context:
        payload["context"] = context
    if agent:
        payload["agent"] = agent
    return ws_send_and_receive(websocket, json.dumps(payload))


# ─── High-Level Test Helpers ─────────────────────────────────────────────────


def run_single_prompt(test_client, prompt: str) -> str:
    """
    Opens a WebSocket, sends one prompt, and returns the full response.
    Each call creates a new conversation thread.
    """
    with test_client.websocket_connect("/v1/ws/messages") as websocket:
        websocket.receive_text()  # consume chat-metadata
        return ws_send_and_receive(websocket, prompt)


def run_conversation(test_client, prompts: List[str]) -> List[str]:
    """
    Opens a single WebSocket session, sends multiple prompts in sequence,
    and returns the list of responses. All prompts share the same thread.
    """
    responses = []
    with test_client.websocket_connect("/v1/ws/messages") as websocket:
        websocket.receive_text()  # consume chat-metadata
        for prompt in prompts:
            msg = ws_send_and_receive(websocket, prompt)
            responses.append(msg)
    return responses


def run_prompt_with_context(
    test_client, prompt: str, context: dict = None, agent: str = ""
) -> str:
    """
    Opens a WebSocket, sends a JSON prompt with context, and returns the response.
    Use this to test context-enriched prompts that guide tool call parameters.
    """
    with test_client.websocket_connect("/v1/ws/messages") as websocket:
        websocket.receive_text()  # consume chat-metadata
        return ws_send_json_and_receive(websocket, prompt, context, agent)


# ─── Assertion Helpers ────────────────────────────────────────────────────────


def assert_llm_as_judge(
    expected: str, actual: str, prompt: str, min_score: int = 6
) -> int:
    """
    Uses LLM-as-judge to semantically compare expected and actual responses.

    Args:
        expected: The reference/expected response.
        actual: The actual response from the agent.
        prompt: The original prompt that produced the responses.
        min_score: Minimum acceptable semantic similarity score (1-10).

    Returns:
        The semantic similarity score (1-10).

    Raises:
        AssertionError: If the score is below min_score or the judge returns
            an invalid response.
    """
    llm = LLMManager.get_instance()
    response = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Prompt: {prompt}\n\n"
            f"Expected Reference: {expected}\n\n"
            f"Actual Response: {actual}\n\n"
            "Rate semantic equivalence from 1 to 10. Return only a single integer."
        )),
    ]).text.strip()

    match = re.search(r"\b(10|[1-9])\b", response)
    assert match, f"Judge returned invalid score: '{response}'"

    score = int(match.group(1))

    if score in (7, 8):
        warnings.warn(f"Semantic judge returned warning score: {score}", UserWarning)

    assert score >= min_score, (
        f"Semantic score too low ({score}/{min_score}).\n"
        f"Prompt: {prompt}\n"
        f"Expected: {expected}\n"
        f"Actual: {actual}"
    )

    return score


# ─── Kubernetes Resource Helpers ──────────────────────────────────────────────

logger = logging.getLogger(__name__)


def create_k8s_resources(resource_yamls: List[str]) -> List[dict]:
    """
    Creates Kubernetes resources from a list of YAML strings.

    Each YAML string is a standard Kubernetes resource manifest (Namespace,
    Pod, Deployment, Service, ConfigMap, etc.).  Resources are created via
    the dynamic client so *any* resource kind works without hard-coding APIs.

    Args:
        resource_yamls: List of YAML-formatted Kubernetes resource manifests.

    Returns:
        List of parsed resource dicts (useful for teardown).
    """
    from kubernetes import dynamic
    from kubernetes import client as k8s_client

    dyn = dynamic.DynamicClient(k8s_client.ApiClient())
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
        except k8s_client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                logger.info(f"{kind} '{metadata.get('name')}' already exists, skipping")
                created.append(body)
            else:
                raise

    return created


def delete_k8s_resources(resource_dicts: List[dict]) -> None:
    """
    Deletes Kubernetes resources previously created by create_k8s_resources.

    Resources are deleted in reverse order (LIFO) so dependent resources
    (e.g. Pods inside a Namespace) are removed before the Namespace itself.

    Args:
        resource_dicts: List of parsed resource dicts as returned by
                        create_k8s_resources.
    """
    from kubernetes import dynamic
    from kubernetes import client as k8s_client

    dyn = dynamic.DynamicClient(k8s_client.ApiClient())

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
        except k8s_client.rest.ApiException:
            logger.warning(f"Failed to delete {kind} '{name}', ignoring")
