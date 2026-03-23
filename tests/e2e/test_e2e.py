"""E2E tests for single-turn knowledge questions.

Tests general Kubernetes knowledge without requiring MCP tool calls.
Add new test cases to KNOWLEDGE_TEST_CASES below — each only needs a prompt
and a concise reference answer for the LLM-as-judge evaluation.
"""

import pytest

from tests.e2e.helpers import (
    E2ETestCase,
    run_single_prompt,
    assert_llm_as_judge,
)


# ─── Test Case Definitions ───────────────────────────────────────────────────
# Add new test cases here. Each needs:
#   - id:       unique pytest test ID
#   - prompt:   the question sent to the agent
#   - expected: concise reference answer (the judge checks semantic match, not exact text)
#   - min_score (optional): minimum acceptable judge score, default 6

EXPECT_POD_ANSWER = """<message>A Pod is the smallest and simplest unit in the Kubernetes object model that you create or deploy. It represents a single instance of a running process in your cluster.
  
  Here's a breakdown:
  *   **Encapsulation:** A Pod encapsulates one or more containers (such as Docker containers), storage resources, a unique network IP, and options that govern how the containers should run.
  *   **Shared Context:** Containers within a Pod share the same network namespace, IP address, and storage. This allows them to communicate with each other using `localhost` and share data through mounted volumes.
  *   **Atomic Unit:** Pods are treated as atomic units. When you scale a Deployment, you scale the number of Pods, not individual containers. If a Pod needs to be restarted due to a failure or an update, the entire Pod is recreated.
  *   **Ephemeral:** Pods are designed to be relatively ephemeral. They can be started, stopped, and replaced without affecting the application's overall availability (when managed by higher-level controllers like Deployments).
  
  For more detailed information, you can refer to the official Kubernetes documentation on Pods: [https://kubernetes.io/docs/concepts/workloads/pods/](https://kubernetes.io/docs/concepts/workloads/pods/)
  <suggestion>List all pods in a cluster</suggestion><suggestion>Inspect a specific pod</suggestion><suggestion>What is a Deployment?</suggestion></message>"""

KNOWLEDGE_TEST_CASES = [
    E2ETestCase(
        id="what_is_a_pod",
        prompt="what is a pod?",
        expected=EXPECT_POD_ANSWER,
        description="Basic Kubernetes Pod definition",
    ),
]


# ─── Parameterized Tests ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    KNOWLEDGE_TEST_CASES,
    ids=[tc.id for tc in KNOWLEDGE_TEST_CASES],
)
def test_knowledge_question(agent_test_session, test_client, test_case):
    """
    Sends a knowledge question via WebSocket and evaluates the response
    using LLM-as-judge against the expected reference answer.
    """
    msg = run_single_prompt(test_client, test_case.prompt)
    assert_llm_as_judge(
        expected=test_case.expected,
        actual=msg,
        prompt=test_case.prompt,
        min_score=test_case.min_score,
    )
