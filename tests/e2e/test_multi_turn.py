"""E2E tests for multi-turn conversations.

Validates that the agent maintains context across multiple exchanges within
a single WebSocket session (same conversation thread).  Each turn is
independently scored by the LLM-as-judge.
"""

import pytest

from tests.e2e.helpers import (
    MultiTurnTestCase,
    ConversationTurn,
    ws_send_and_receive,
    assert_llm_as_judge,
)


from tests.e2e.test_e2e import EXPECT_POD_ANSWER


# ─── Test Case Definitions ───────────────────────────────────────────────────

MULTI_TURN_TEST_CASES = [
    MultiTurnTestCase(
        id="pod_followup",
        description="Ask about pods, then follow up with troubleshooting",
        turns=[
            ConversationTurn(
                prompt="what is a pod?",
                expected=EXPECT_POD_ANSWER,
            ),
            ConversationTurn(
                prompt="how do I check if it is running?",
                expected=(
                    "You can check a pod's status by viewing the Pods list in the "
                    "Rancher UI. The Status column shows whether the pod is Running, "
                    "Pending, CrashLoopBackOff, etc. You can click on a pod to see "
                    "detailed status information, events, and container logs."
                ),
            ),
        ],
    ),
]


# ─── Parameterized Tests ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    MULTI_TURN_TEST_CASES,
    ids=[tc.id for tc in MULTI_TURN_TEST_CASES],
)
def test_multi_turn_conversation(agent_test_session, test_client, test_case):
    """
    Sends multiple prompts in sequence within a single WebSocket session
    and evaluates each response using LLM-as-judge.

    This validates that the agent maintains conversational context —
    e.g., resolving pronouns like "it" to a previously mentioned resource.
    """
    with test_client.websocket_connect("/v1/ws/messages") as websocket:
        websocket.receive_text()  # consume chat-metadata

        for turn in test_case.turns:
            msg = ws_send_and_receive(websocket, turn.prompt)
            assert_llm_as_judge(
                expected=turn.expected,
                actual=msg,
                prompt=turn.prompt,
                min_score=turn.min_score,
            )
