import pytest
from deepeval.test_case import LLMTestCase

from tests.e2e.conftest import (
    E2ETestCase,
    run_single_prompt,
    evaluate_and_assert,
)


TEST_CASES = [
    E2ETestCase(
        id="configmap_does_not_exist",
        prompt="does the Configmap 'does-not-exist' exist in namespace 'empty' in the cluster local?",
        expected="""I couldn't find the Configmap `does-not-exist` in the `empty` namespace of the `local` cluster.
        It's possible the Configmap does not exist, or there was a typo in the name""",
        description="Configmap does not exist",
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
        id="context_from_ui",
        prompt="""{"prompt": "show all running pods", "context": {"cluster": "local", "namespace": "default"}}""",
        expected="There are no running pods found.",
        description="Show all running pods with context from UI",
        expected_agent="rancher",
    ),
    E2ETestCase(
        id="list_configmaps_in_namespace",
        prompt="list configmaps in namespace 'e2e-test' in cluster local",
        expected="There is a configmap named 'e2e-test-config' in the 'e2e-test' namespace.",
        description="List configmaps in a namespace",
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


@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[tc.id for tc in TEST_CASES],
)
def test_single_message(agent_test_session, test_client, test_case, k8s_resources, e2e_results):
    result = run_single_prompt(test_client, test_case.prompt)

    if test_case.expected_agent is not None:
        assert result.agent_name == test_case.expected_agent, (
            f"Expected agent '{test_case.expected_agent}', got '{result.agent_name}'"
        )

    test = LLMTestCase(
        name=test_case.id,
        input=test_case.prompt,
        actual_output=result.response,
        expected_output=test_case.expected,
    )

    evaluate_and_assert([test], e2e_results)
