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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
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
        resources: Optional list of Kubernetes resource YAML strings to create before
                   the test and delete after.
    """
    id: str
    prompt: str
    expected: str
    description: str = ""
    min_score: int = 6
    resources: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    expect_summary: Optional[bool] = None
    expected_agent: Optional[str] = None


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation test.

    Attributes:
        prompt: The user prompt for this turn.
        expected: Reference answer for semantic comparison.
        min_score: Minimum acceptable semantic similarity score (1-10, default 6).
        expected_agent: Optional expected agent name for this turn.
        expected_confirmation_message: Optional expected confirmation message for this turn.
    """
    prompt: str
    expected:  Optional[str] = None
    expected_confirmation_message: Optional[str] = None
    min_score: int = 6
    expected_agent: Optional[str] = None


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
    expected_tools: List[str] = field(default_factory=list)
    expect_summary: Optional[bool] = None


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


@dataclass
class PromptResult:
    """Result from a single prompt interaction, including metadata for assertions."""
    response: str
    thread_id: str
    agent_name: Optional[str] = None
    selection_mode: Optional[str] = None


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


def get_langgraph_state(checkpointer, thread_id: str) -> dict:
    """Retrieve the LangGraph checkpoint state for a given thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = checkpointer.get_tuple(config=config)
    if not checkpoint_tuple:
        return {}
    return checkpoint_tuple.checkpoint.get("channel_values", {})


def get_executed_tools(messages) -> list:
    """Extract tool names from ToolMessage instances in the message history."""
    return [msg.name for msg in messages if isinstance(msg, ToolMessage)]


def run_single_prompt(test_client, prompt: str) -> PromptResult:
    """
    Opens a WebSocket, sends one prompt, and returns a PromptResult
    containing the response, thread_id, and agent metadata.
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


def run_multi_turn(test_client, prompts: List[str]) -> List[PromptResult]:
    """
    Opens a single WebSocket session, sends multiple prompts in sequence,
    and returns a list of PromptResult objects. All prompts share the same
    thread_id (conversation).
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
    Use this to test context-enriched prompts that guide tool call parameters.
    """
    with test_client.websocket_connect("/v1/ws/messages") as websocket:
        websocket.receive_text()  # consume chat-metadata
        return ws_send_json_and_receive(websocket, prompt, context, agent)


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
