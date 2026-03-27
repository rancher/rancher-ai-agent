# Adding New E2E Tests

This guide explains how to add new end-to-end tests. All e2e tests use an LLM-as-judge approach (via [deepeval](https://docs.confident-ai.com/) GEval) to evaluate whether the agent's response is semantically correct, so you only need to provide a prompt and a reference answer.

## Single-Message Tests

Single-message tests send one prompt and evaluate a single response. Add new cases to `test_single_message.py`.

### Steps

1. Open `test_single_message.py` and add a new `E2ETestCase` to the `TEST_CASES` list:

```python
E2ETestCase(
    id="my_new_test",                      # Unique identifier (used in test output)
    prompt="your prompt to the agent",     # The user message sent via WebSocket
    expected="the expected answer",        # Reference answer for LLM-as-judge scoring
    description="Short description",       # Optional, for documentation
    expected_agent="rancher",              # Optional, asserts which child agent handled it
    resources=[],                          # Optional, Kubernetes YAML resources (see below)
)
```

2. That's it. The parametrized `test_single_message` function picks up all entries in `TEST_CASES` automatically.

### E2ETestCase Fields

| Field | Required | Description |
|---|---|---|
| `id` | Yes | Unique test identifier, used as the pytest test ID |
| `prompt` | Yes | The message sent to the agent |
| `expected` | Yes | Reference answer for semantic comparison (doesn't need to match exactly) |
| `description` | No | Human-readable description |
| `expected_agent` | No | If set, asserts the agent name in the response metadata matches |
| `resources` | No | List of YAML strings for Kubernetes resources to create before the test and delete after |

### Example with Kubernetes Resources

If your test needs specific resources to exist in the cluster, provide them in the `resources` field as YAML strings. They are created before the test and cleaned up after:

```python
E2ETestCase(
    id="find_my_service",
    prompt="list services in namespace 'my-ns' in cluster local",
    expected="There is a service named 'my-svc' in the 'my-ns' namespace.",
    expected_agent="rancher",
    resources=[
        """
        apiVersion: v1
        kind: Namespace
        metadata:
          name: my-ns
        """,
        """
        apiVersion: v1
        kind: Service
        metadata:
          name: my-svc
          namespace: my-ns
        spec:
          ports:
            - port: 80
        """,
    ],
)
```

Resources are created in order and deleted in reverse order (LIFO) after the test.

### Sending Context (Simulating UI Context)

To simulate context that the UI sends (e.g. active cluster/namespace), format the prompt as a JSON string:

```python
E2ETestCase(
    id="with_ui_context",
    prompt='{"prompt": "show all pods", "context": {"cluster": "local", "namespace": "default"}}',
    expected="No pods found in the default namespace.",
)
```

## Multi-Message Tests

Multi-message tests send multiple prompts within a single WebSocket session (same conversation thread), validating that the agent maintains context across turns. Add new cases to `test_multi_messages.py`.

### Steps

1. Open `test_multi_messages.py` and add a new `MultiTurnTestCase` to the `MULTI_TURN_TEST_CASES` list:

```python
MultiTurnTestCase(
    id="my_conversation_test",
    description="Description of what this conversation tests",
    turns=[
        ConversationTurn(
            prompt="first user message",
            expected="expected response to first message",
            expected_agent="rancher",
        ),
        ConversationTurn(
            prompt="follow-up message",
            expected="expected response using context from first turn",
            expected_agent="rancher",
        ),
    ],
    resources=[],  # Optional, same as single-message tests
)
```

2. The parametrized `test_multi_turn_conversation` function picks up all entries automatically.

### ConversationTurn Fields

| Field | Required | Description |
|---|---|---|
| `prompt` | Yes | The user message for this turn |
| `expected` | No* | Reference answer for semantic comparison |
| `expected_confirmation_message` | No* | Expected confirmation XML in the response (for human-validation flows) |
| `expected_agent` | No | If set, asserts the agent name in the response metadata matches |

*Each turn should have either `expected` (evaluated by LLM-as-judge) or `expected_confirmation_message` (matched as a substring in the response).

### Testing Human-in-the-Loop (Confirmation) Flows

When a tool requires human validation, the agent sends a `<confirmation-response>` message and waits for the user to approve or reject. Use `expected_confirmation_message` to assert the confirmation payload, then follow up with "yes" or "no":

```python
MultiTurnTestCase(
    id="create_and_confirm",
    description="Create a resource with human validation",
    turns=[
        ConversationTurn(
            prompt='create a Kubernetes resource in the local cluster using this exact JSON: {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "test-cm", "namespace": "default"}, "data": {"key": "value"}}',
            expected_confirmation_message='<confirmation-response>[{"type": "create"',
            expected_agent="rancher",
        ),
        ConversationTurn(
            prompt="yes",  # Approve the creation
            expected="The ConfigMap `test-cm` has been created successfully.",
            expected_agent="rancher",
        ),
    ],
)
```

To test rejection, use `"no"` as the follow-up prompt and expect a cancellation response.

### Testing Multi-Agent Routing

To verify the parent agent routes to different child agents across turns, set `expected_agent` on each turn:

```python
MultiTurnTestCase(
    id="cross_agent_conversation",
    description="Conversation spanning multiple agents",
    turns=[
        ConversationTurn(
            prompt="list pods in namespace default in cluster local",
            expected="...",
            expected_agent="rancher",
        ),
        ConversationTurn(
            prompt="show me the GitRepos in fleet-default",
            expected="...",
            expected_agent="fleet",
        ),
    ],
)
```

## How Evaluation Works

- Each response (except confirmation turns) is evaluated using deepeval's **GEval** metric with a "Correctness" criterion.
- The judge LLM checks whether the actual response conveys the same fundamental meaning as the expected output — exact wording is **not** required.
- The passing threshold is `0.5` (configurable per metric). Scores below this fail the test.
- Results are aggregated across the session and written to `GITHUB_STEP_SUMMARY` when running in CI.

## Running E2E Tests Locally

1. Have a Rancher instance running and accessible.
2. Set the required environment variables:
   - `RANCHER_URL` — URL of your Rancher instance
   - `RANCHER_API_TOKEN` — a valid Rancher API token
   - LLM provider credentials (e.g. `GOOGLE_API_KEY`, `GEMINI_MODEL`, `ACTIVE_LLM`, or `OPENAI_API_KEY`)
3. Run the tests:

```bash
uv run deepeval test run tests/e2e/ -vvv
```
