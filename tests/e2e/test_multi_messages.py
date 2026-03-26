"""E2E tests for multi-turn conversations.

Validates that the agent maintains context across multiple exchanges within
a single WebSocket session (same conversation thread).  Each turn is
independently scored by deepeval GEval, and tool calls / summary are
asserted against LangGraph state after all turns complete.
"""

import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import GeminiModel

from app.main import app
from tests.e2e.helpers import (
    MultiTurnTestCase,
    ConversationTurn,
    run_multi_turn,
    get_langgraph_state,
    get_executed_tools,
)


# ─── Test Case Definitions ───────────────────────────────────────────────────

MULTI_TURN_TEST_CASES = [
    MultiTurnTestCase(
        id="create_configmap",
        description="Create a ConfigMap in a specific namespace and verify its creation",
        turns=[
            ConversationTurn(
                prompt='create a Kubernetes resource in the local cluster using this exact JSON, do not modify it and include the kind and apiVersion: {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cm1", "namespace": "cm-create-ns"}, "data": {"foo": "bar"}}',
                expected_confirmation_message="""<confirmation-response>[{"type": "create", "payload": {"apiVersion": "v1", "data": {"foo": "bar"}, "kind": "ConfigMap", "metadata": {"name": "cm1", "namespace": "cm-create-ns"}}, "resource": {"name": "cm1", "kind": "ConfigMap", "cluster": "local", "namespace": "cm-create-ns"}}]</confirmation-response>""",
                expected_agent="rancher",
            ),
            ConversationTurn(
                prompt="yes",
                expected=(
                    "The ConfigMap `cm1` has been created successfully."
                ),
                expected_agent="rancher",
            ),
            ConversationTurn(
                prompt='create a Kubernetes resource in the local cluster using this exact JSON, do not modify it and include the kind and apiVersion: {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cm2", "namespace": "cm-create-ns"}, "data": {"foo": "bar"}}',
                expected_confirmation_message="""<confirmation-response>[{"type": "create", "payload": {"apiVersion": "v1", "data": {"foo": "bar"}, "kind": "ConfigMap", "metadata": {"name": "cm2", "namespace": "cm-create-ns"}}, "resource": {"name": "cm2", "kind": "ConfigMap", "cluster": "local", "namespace": "cm-create-ns"}}]</confirmation-response>""",
                expected_agent="rancher",
            ),
            ConversationTurn(
                prompt="no",
                expected=(
                    "<message></message>" # empty message indicating user cancelled the action
                ),
            ),
            ConversationTurn(
                prompt="list all the names of the ConfigMaps in the cm-create-ns namespace",
                expected=(
                    "There is 2 ConfigMapz in the `cm-create-ns` namespace: kube-root-ca.crt and cm1."
                ),
                expected_agent="rancher",
            )
        ],
        expected_tools=["createKubernetesResource", "listKubernetesResources"],
        expect_summary=True,
        resources=[
            """
            apiVersion: v1
            kind: Namespace
            metadata:
              name: cm-create-ns
            """]
    ),
    MultiTurnTestCase(
        id="multiple_agents_adaptive",
        description="Test that the agent can adaptively select different agents across turns based on the prompt",
        turns=[
            ConversationTurn(
                prompt="what is a pod in Kubernetes?",
                expected=(
                    "A pod is the smallest deployable unit in Kubernetes, representing a single instance of a running process in a cluster. "
                    "It can contain one or more containers that share storage and network resources. Pods are used to host applications and are managed by Kubernetes to ensure they run reliably."
                ),
                expected_agent="rancher",
            ),
            ConversationTurn(
                prompt="show me the GitRepos in the 'fleet-default' workspace",
                expected="There are no GitRepos found.",
                expected_agent="fleet",
            ),
            ConversationTurn(
                prompt="what K3k virtual clusters are currently provisioned?",
                expected="There are no K3k clusters found.",
                expected_agent="provisioning",
            ),
        ],
        expected_tools=["listGitRepos", "listK3kClusters"],
        expect_summary=True,
        resources=[
            """
            apiVersion: v1
            kind: Namespace
            metadata:
              name: cm-create-ns
            """]
    )
]


# ─── Parameterized Tests ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    MULTI_TURN_TEST_CASES,
    ids=[tc.id for tc in MULTI_TURN_TEST_CASES],
)
def test_multi_turn_conversation(agent_test_session, test_client, test_case, k8s_resources, e2e_results):
    """
    Sends multiple prompts in sequence within a single WebSocket session
    and evaluates each response using deepeval GEval.

    This validates that the agent maintains conversational context —
    e.g., resolving pronouns like "it" to a previously mentioned resource.
    After all turns, tool calls and summary are asserted against LangGraph state.
    """
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' conveys the same fundamental meaning and key concepts as the 'expected output'. The wording, phrasing, structure, and level of detail may differ — focus only on whether the core factual content is semantically equivalent. Ignore stylistic differences, additional context, suggestions, and agent metadata like agent selection messages.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=GeminiModel(model="gemini-2.5-flash"),
    )

    prompts = [turn.prompt for turn in test_case.turns]
    results = run_multi_turn(test_client, prompts)
    thread_id = results[0].thread_id

    eval_cases = []
    for turn, result in zip(test_case.turns, results):
        # --- Per-turn agent assertion ---
        if turn.expected_agent is not None:
            assert result.agent_name == turn.expected_agent, (
                f"Turn '{turn.prompt[:30]}...': expected agent '{turn.expected_agent}', got '{result.agent_name}'"
            )

        if turn.expected_confirmation_message:
            assert turn.expected_confirmation_message in result.response, (
                f"Turn '{turn.prompt[:30]}...': expected confirmation message not found in response"
            )
        else:
            eval_cases.append(
                LLMTestCase(
                    name=test_case.id,
                    input=turn.prompt,
                    actual_output=result.response,
                    expected_output=turn.expected,
                )
            )

    # --- LangGraph state assertions ---
    checkpointer = app.memory_manager.get_checkpointer()
    state = get_langgraph_state(checkpointer, thread_id)
    messages = state.get("messages", [])

    if test_case.expected_tools:
        actual_tools = get_executed_tools(messages)
        missing = [t for t in test_case.expected_tools if t not in actual_tools]
        assert not missing, (
            f"Expected tools {sorted(test_case.expected_tools)} to be included in actual tools {sorted(actual_tools)}. Missing: {missing}"
        )

    if test_case.expect_summary is not None:
        summary = state.get("summary", {})
        has_summary = bool(summary.get("text"))
        assert has_summary == test_case.expect_summary, (
            f"Expected summary={'present' if test_case.expect_summary else 'absent'}, "
            f"got summary={'present' if has_summary else 'absent'}"
        )

    # --- Semantic correctness assertion ---
    eval_results = evaluate(eval_cases, metrics=[correctness_metric])

    e2e_results.append(eval_results)

    failed = [r for r in eval_results.test_results if not r.success]
    assert not failed, f"{len(failed)} turn(s) failed evaluation"
