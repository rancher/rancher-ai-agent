"""Reusable utilities for e2e tests.

This module provides helper classes, functions, and constants that are shared
across all e2e test files. To add a new test case, use the E2ETestCase or
MultiTurnTestCase dataclasses.
"""

import json
import re
import warnings

from dataclasses import dataclass, field
from typing import Any, Dict, List
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
    """
    id: str
    prompt: str
    expected: str
    description: str = ""
    min_score: int = 6


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
    """
    id: str
    turns: List[ConversationTurn]
    description: str = ""


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
