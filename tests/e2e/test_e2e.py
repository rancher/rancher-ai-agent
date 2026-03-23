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

TEST_CASESa = [
    E2ETestCase(
        id="namespace_does_not_exist",
        prompt="show namespace does-not-exist in cluster local",
        expected="""I couldn't find the namespace `does-not-exist` in the `local` cluster. 
        It's possible the namespace does not exist, or there was a typo in the name""",
        description="Namespace does not exist",
    ),
     E2ETestCase(
        id="namespace_does_not_exist",
        prompt="show namespace does-not-exist in cluster local",
        expected="""I couldn't find the namespace `does-not-exist` in the `local` cluster. 
        It's possible the namespace does not exist, or there was a typo in the name""",
        description="Namespace does not exist",
    ),
]

TEST_CASES = [
     E2ETestCase(
        id="namespace_does_not_exist",
        prompt="is there a namespace called e2e-test in cluster local",
        expected="""I couldn't find the namespace `e2e-test` in the `local` cluster. 
        It's possible the namespace does not exist, or there was a typo in the name""",
        description="Namespace does not exist",
        resources=[
        """
        apiVersion: v1
        kind: Namespace
        metadata:
          name: e2e-test
        """
        ],
    ),
]



# ─── Parameterized Tests ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[tc.id for tc in TEST_CASES],
)
def test_knowledge_question(agent_test_session, test_client, test_case, k8s_resources):
    """
    Sends a knowledge question via WebSocket and evaluates the response
    using LLM-as-judge against the expected reference answer.
    If the test case defines resources, they are created before the test
    and cleaned up after.
    """
    msg = run_single_prompt(test_client, test_case.prompt)
    assert_llm_as_judge(
        expected=test_case.expected,
        actual=msg,
        prompt=test_case.prompt,
        min_score=test_case.min_score,
    )
