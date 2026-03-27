"""E2E tests for multi-turn conversations.

Validates that the agent maintains context across multiple exchanges within
a single WebSocket session (same conversation thread).  Each turn is
independently scored by deepeval GEval.
"""

import pytest
from deepeval.test_case import LLMTestCase

from tests.e2e.conftest import (
    MultiTurnTestCase,
    ConversationTurn,
    run_multi_turn,
    evaluate_and_assert,
)


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
                    "<message></message>"
                ),
            ),
            ConversationTurn(
                prompt="list all the names of the ConfigMaps in the cm-create-ns namespace in the cluster local",
                expected=(
                    "There is 2 ConfigMapz in the `cm-create-ns` namespace: kube-root-ca.crt and cm1." # Note: kube-root-ca.crt is automatically created by Kubernetes in every namespace
                ),
                expected_agent="rancher",
            )
        ],
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
            )
        ],
    )
]


@pytest.mark.parametrize(
    "test_case",
    MULTI_TURN_TEST_CASES,
    ids=[tc.id for tc in MULTI_TURN_TEST_CASES],
)
def test_multi_turn_conversation(agent_test_session, test_client, test_case, k8s_resources, e2e_results):
    """
    Sends multiple prompts in sequence within a single WebSocket session
    and evaluates each response using deepeval GEval.
    """
    prompts = [turn.prompt for turn in test_case.turns]
    results = run_multi_turn(test_client, prompts)

    eval_cases = []
    for turn, result in zip(test_case.turns, results):
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

    evaluate_and_assert(eval_cases, e2e_results)
