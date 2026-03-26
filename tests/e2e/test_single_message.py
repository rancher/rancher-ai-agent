"""E2E tests for single-turn knowledge questions.

Tests general Kubernetes knowledge without requiring MCP tool calls.
Add new test cases to KNOWLEDGE_TEST_CASES below — each only needs a prompt
and a concise reference answer for the LLM-as-judge evaluation.
"""

import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import GeminiModel

from app.main import app
from tests.e2e.helpers import (
    E2ETestCase,
    run_single_prompt,
    get_langgraph_state,
    get_executed_tools,
)


TEST_CASES = [
    E2ETestCase(
        id="what_is_a_pod",
        prompt="what is a pod in Kubernetes?",
        expected=(
            "A pod is the smallest deployable unit in Kubernetes, representing a single instance of a running process in a cluster. "
            "It can contain one or more containers that share storage and network resources. Pods are used to host applications and are managed by Kubernetes to ensure they run reliably."
        ),
        description="Basic Kubernetes knowledge that doesn't require tool calls",
        expected_tools=[],
        expect_summary=False,
        expected_agent="rancher",
    ),
    E2ETestCase(
        id="configmap_does_not_exist",
        prompt="does the Configmap 'does-not-exist' exist in namespace 'empty' in the cluster local?",
        expected="""I couldn't find the Configmap `does-not-exist` in the `empty` namespace of the `local` cluster.
        It's possible the Configmap does not exist, or there was a typo in the name""",
        description="Configmap does not exist",
        expected_tools=["getKubernetesResource"],
        expect_summary=False,
        expected_agent="rancher",
         resources=[
            """
            apiVersion: v1
            kind: Namespace
            metadata:
              name: empty
            """,
        ],
    ),
    E2ETestCase(
        id="show_empty_fleet_workspace",
        prompt="show me the GitRepos in the 'fleet-default' workspace",
        expected="There are no GitRepos found.",
        description="Show empty fleet workspace",
        expected_tools=["listGitRepos"],
        expect_summary=False,
        expected_agent="fleet",
    ),
    E2ETestCase(
        id="list_configmaps_in_namespace",
        prompt="list configmaps in namespace 'e2e-test' in cluster local",
        expected="There is a configmap named 'e2e-test-config' in the 'e2e-test' namespace.",
        description="List configmaps in a namespace",
        expected_tools=["listKubernetesResources"],
        expect_summary=False,
        expected_agent="rancher",
        resources=[
            """
            apiVersion: v1
            kind: Namespace
            metadata:
              name: e2e-test
            """,
            """
            apiVersion: v1
            kind: ConfigMap
            metadata:
              name: e2e-test-config
              namespace: e2e-test
            data:
              key: value
            """,
        ],
    ),
]


# ─── Parameterized Tests ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[tc.id for tc in TEST_CASES],
)
def test_single_message(agent_test_session, test_client, test_case, k8s_resources, e2e_results):
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' conveys the same fundamental meaning and key concepts as the 'expected output'. The wording, phrasing, structure, and level of detail may differ — focus only on whether the core factual content is semantically equivalent. Ignore stylistic differences, additional context, suggestions, and agent metadata like agent selection messages.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=GeminiModel(model="gemini-2.5-flash")
    )

    result = run_single_prompt(test_client, test_case.prompt)

    # --- Agent selection assertion ---
    if test_case.expected_agent is not None:
        assert result.agent_name == test_case.expected_agent, (
            f"Expected agent '{test_case.expected_agent}', got '{result.agent_name}'"
        )

    # --- LangGraph state assertions ---
    checkpointer = app.memory_manager.get_checkpointer()
    state = get_langgraph_state(checkpointer, result.thread_id)
    messages = state.get("messages", [])

    if test_case.expected_tools:
        actual_tools = get_executed_tools(messages)
        assert sorted(actual_tools) == sorted(test_case.expected_tools), (
            f"Expected tools {sorted(test_case.expected_tools)}, got {sorted(actual_tools)}"
        )

    if test_case.expect_summary is not None:
        summary = state.get("summary", {})
        has_summary = bool(summary.get("text"))
        assert has_summary == test_case.expect_summary, (
            f"Expected summary={'present' if test_case.expect_summary else 'absent'}, "
            f"got summary={'present' if has_summary else 'absent'}"
        )

    # --- Semantic correctness assertion ---
    test = LLMTestCase(
        name=test_case.id,
        input=test_case.prompt,
        actual_output=result.response,
        expected_output=test_case.expected
    )

    results = evaluate([test], metrics=[correctness_metric])

    e2e_results.append(results)

    failed = [r for r in results.test_results if not r.success]
    assert not failed, f"{len(failed)} test case(s) failed evaluation"
