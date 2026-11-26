# Spike: Agentic Tasks (POC)

This document describes the proof of concept on the `agentic-tasks` branch and lists the open
questions that still need to be answered before this becomes a real feature.

## Goal

Let Liz run **unattended tasks**: a prompt (for example "check the health of all downstream
clusters and report problems") that runs on a schedule with no user watching. The result is stored
so a user can read it later from the UI.

## Overview

The POC adds two CRDs and two kopf controllers to the existing agent process. Everything is behind a
feature flag.

```
AITaskDefinition ──(scheduler daemon, cron)──▶ AITask ──(executor)──▶ supervisor / agent ──▶ store
        ▲                                         ▲                                           │
        └──────────── GET /v1/api/aitaskdefinitions, /v1/api/aitasks[/{name}] ◀───────────────┘
                       (read as the calling user through Rancher Steve)
```

### Feature flag

- `AGENTIC_TASK_ENABLED=true` (Helm: `agenticTask.enabled`) registers the AITask/AITaskDefinition
  kopf handlers and the REST endpoints (`app/main.py`). When it is off, nothing changes.

### CRDs (`crd-generation/api/v1alpha1/`, `ai.cattle.io/v1alpha1`, namespaced)

**AITaskDefinition** (`aitd`): the template for a recurring or manual-only task.

| Field | Description |
| :--- | :--- |
| `spec.displayName` | Human readable name. |
| `spec.prompt` | What Liz has to do (max 8192 chars). |
| `spec.schedule` | Cron expression or macro (`@daily`, ...). Empty means manual only. `CRON_TZ`/`TZ` prefixes are rejected. |
| `spec.timeZone` | IANA time zone, UTC by default. |
| `spec.suspend` | Stops new scheduled runs. |
| `spec.agents` | `AIAgentConfig` names the task may use. Empty means every enabled agent. |
| `status.conditions` | `ScheduleValid` and `Ready`. |
| `status.lastScheduleTime` / `status.nextRunTime` | Owned by the scheduler. |

The name is limited to 52 characters so that `<definition>-<unix minutes>` fits in 63.

**AITask** (`ait`): one execution of a definition. The spec is immutable (CEL rule).

| Field | Description |
| :--- | :--- |
| `spec.taskDefinitionRef` | Owning definition. |
| `spec.prompt`, `spec.agents` | Copied from the definition when the task is created. |
| `status.phase` | `Pending`, `Running`, `Succeeded`, `Failed`. |
| `status.startTime` / `status.completionTime` | Run timestamps. |
| `status.result` | `Healthy`, `Warning`, `Critical`. |
| `status.pendingValidation` | `true` while the run is paused waiting for a human validation answer (phase stays `Running`). See [Human validation in tasks](#human-validation-in-tasks). |

AITasks have the labels `ai.cattle.io/task-definition` and `ai.cattle.io/trigger`, the
annotations `ai.cattle.io/scheduled-at` and `ai.cattle.io/validation-response` (the answer to a
pending validation), and an owner reference to the definition, so they are
garbage-collected together. `common_types.go` also defines a `CredentialsSecretRef` and the
`ai.cattle.io/task-credentials` label. These are not wired yet, see [RBAC](#rbac--permissions).

The CRDs are installed by the chart (`chart/agent/templates/crds/`). The agent service account
gets the needed permissions on `aitasks`, `aitaskdefinitions` and their `status` subresources.

### Scheduler (`app/controllers/ai_task_definition.py`)

A `kopf.daemon` runs for each AITaskDefinition and works like a small CronJob controller:

- It evaluates the cron schedule in the configured time zone (`croniter`). It sleeps until the next
  run, for 60s at most, so changes to the spec are picked up.
- **Suspended / manual only**: no tasks are created. When the definition is resumed, missed runs
  are not caught up.
- **Overlap**: if an AITask of the same definition is still `Pending`/`Running`, the run is skipped.
- **Missed runs** (for example while the pod was down): only the latest missed time runs, and only
  if it is at most 5 minutes late (`STARTING_DEADLINE`).
- **Idempotency**: task names are deterministic (`<definition>-<unix minutes>`), so a 409 on
  create is treated as "already created". Every handled time is recorded in
  `status.lastScheduleTime`.
- Status (conditions, `nextRunTime`) is only patched when it changes.

### Run now

`POST /v1/api/aitaskdefinitions/{name}/run` starts a run immediately, also for suspended or
manual-only definitions.

1. The definition is read through Steve as the user (403/404).
2. If an AITask of the definition is still `Pending`/`Running`, it returns 409.
3. It sets the annotation `ai.cattle.io/run-requested-at=<now>` on the definition through Steve
   as the user, so the user needs `patch` on that AITaskDefinition (`resourceNames` works).
4. It returns 202 with the name of the AITask that will be created.

A `kopf.on.field` handler on the annotation creates the AITask with the agent service account:
label `ai.cattle.io/trigger=manual`, name `<definition>-<hash of the annotation value>`, so it is
idempotent. The same works with
`kubectl annotate aitd <name> ai.cattle.io/run-requested-at=<value> --overwrite`.

### Executor (`app/controllers/ai_task.py`)

A `kopf.on.create`/`on.resume` handler (timeout 1h) runs each new AITask:

1. It skips the task if it is already `Running`/`Succeeded`/`Failed`, so a task is never run twice
   after an operator restart.
2. It sets `phase=Running`, resolves `spec.agents` against the enabled `AIAgentConfig`s and skips
   `OAUTH2` agents, which need an interactive login.
3. It builds the graph with the new `build_agent_from_configs()` (`factory.py`): a single agent for
   one config, otherwise a supervisor. The planner is always disabled (`allow_planner=False`). No
   websocket is needed.
4. It invokes the graph with `configurable.aitask=True`. This switches the prompts through
   `aitask_prompt_middleware`:
   - The supervisor prompt is replaced by `AITASK_SUPERVISOR_PROMPT`: unattended, never ask
     questions, prefer read-only actions, write a structured final report, and end with
     `RESULT: Healthy|Warning|Critical`.
   - Child agents get `AITASK_CHILD_INSTRUCTIONS` appended to their prompt.
   - `aitask` is forwarded from the supervisor to child agents (`_FORWARDED_CONFIG_KEYS`).
5. If the graph ends in a human-validation **interrupt**, the run is paused: the task gets
   `status.pendingValidation=true` (phase stays `Running`) and the confirmation prompt is stored.
   See [Human validation in tasks](#human-validation-in-tasks).
6. The verdict is parsed from the last line of the output (`parse_result`). The record
   (prompt, agents, phase, result, output, error, times) is written to the LangGraph **store** under
   namespace `("aitasks",)`, keyed by the AITask uid. Status is patched with the phase, the
   completion time and the result.

The checkpointer used for task runs is an in-memory `InMemorySaver` (see TODO in the code). It is
process-wide (`app/services/aitask_runner.py`), so a paused run can be resumed from the kopf loop or
the FastAPI loop.

### Human validation in tasks

A tool listed in an agent's `humanValidationTools` pauses the run instead of failing it:

1. The executor stores `pendingValidation: {id, interruptId, message, requestedAt}` in the task
   record (`message` is the usual `<confirmation-response>` payload) and sets
   `status.pendingValidation=true`. The phase stays `Running`, so no new run of the definition
   starts while it waits.
2. A user opens the follow-up chat (`/v1/ws/messages?aitask=<name>`), which is allowed while the
   task waits. The confirmation prompt is sent on connect, like a live interrupt, and the user's
   next message is the answer (`yes` approves, anything else cancels).
3. The answer is written as the user through Steve, as the annotation
   `ai.cattle.io/validation-response` on the AITask. Rancher only accepts it if the user may
   **`patch` that AITask** (`resourceNames` works), so being able to read the task is not enough to
   approve. A denied user gets an error and the task keeps waiting. The annotation is also the
   audit record of who answered (Kubernetes audit log).
4. The websocket then resumes the paused graph with `Command(resume=<answer>)`, with the **task's**
   agents and identity (not the user's), and streams it to the user. Once it finishes, the record
   and status are updated, and the chat continues as a normal follow-up whose seed contains the
   prompt, each validation request and answer, and the output. A new interrupt asks again.
5. Answers can also be given with kubectl:
   `kubectl annotate ait <name> ai.cattle.io/validation-response='{"response":"yes"}' --overwrite`.
   A kopf `on.field` handler resumes the run in the controller (no streaming). The value is JSON:
   `response`, plus optional `validationId` (must match the pending one), `userId`, `answeredAt` and
   `via`. Chat answers have `via: chat` and the controller ignores them, because the websocket is
   already resuming them.

Only one answer resumes a given validation: the pending `id` is checked against the store and
claimed in-process (`claim_validation`). A later answer gets "already answered". A stream to a
client that disconnects keeps running, so the task is never left waiting. If the agent restarts
while a task waits, the paused state is lost and the task is marked `Failed`.

Answered validations are listed in `GET /aitasks/{name}` (`validations`), with the pending prompt in
`validationMessage` and its id in `validationId`.

### How to answer a validation

#### Prerequisites

1. `AGENTIC_TASK_ENABLED=true` (Helm: `agenticTask.enabled`).
2. An agent used by the task lists the tool in `humanValidationTools`, for example:

   ```yaml
   apiVersion: ai.cattle.io/v1alpha1
   kind: AIAgentConfig
   metadata:
     name: rancher-core-agent
     namespace: cattle-ai-agent-system
   spec:
     # ...
     humanValidationTools:
       - name: "patch_resource"
         type: UPDATE
   ```

3. The user who answers needs `get` and `patch` on the AITask (or `aitasks` in general) in
   `cattle-ai-agent-system`. Grant them through Rancher (a role template or a global role with
   `namespacedRules`), since the chat checks the user's permissions through Steve. A plain
   Kubernetes Role is enough for kubectl:

   ```yaml
   apiVersion: rbac.authorization.k8s.io/v1
   kind: Role
   metadata:
     name: aitask-approver
     namespace: cattle-ai-agent-system
   rules:
     - apiGroups: ["ai.cattle.io"]
       resources: ["aitasks"]
       verbs: ["get", "list", "watch", "patch"]
   ```

#### Find tasks waiting for validation

```sh
kubectl get ait -n cattle-ai-agent-system
# NAME             DEFINITION   PHASE     RESULT   VALIDATION   AGE
# daily-29374512   daily        Running            true         3m
```

The prompt itself is not in the CR. Read it with the API, as the user (`R_SESS` cookie):

```sh
curl -s -b "R_SESS=$TOKEN" https://<agent>/v1/api/aitasks/daily-29374512 | jq '{phase, pendingValidation, validationId, validationMessage}'
```

`validationMessage` is a `<confirmation-response>` with the action, for example
`{"type": "update", "resource": {...}, "payload": {"original": ..., "patched": ...}}`.

#### Option 1: answer in the follow-up chat (websocket)

1. Open the follow-up chat of the task:
   - Test UI: `/ui?aitask=<aitask name>`, or type the name in the AITask field and press Open.
   - Any client: connect to `/v1/ws/messages?aitask=<aitask name>` with the user's `R_SESS` cookie.
2. After `<chat-metadata>`, the server sends the confirmation prompt as a message:
   `<message>`, `<confirmation-response>{...}</confirmation-response>`, `</message>`.
3. Send the answer as the next message, as plain text or as `{"prompt": "yes"}`:
   - `yes` approves and runs the action.
   - Anything else (for example `no`) cancels it. The task then goes on without it.
4. The resumed run is streamed in a new `<message>`. Then:
   - If the task asks for another validation, its prompt is streamed too. Answer it the same way.
   - If the task finished (`Succeeded` or `Failed`), the chat goes on as a normal follow-up. The
     next message is the first question about the result, and the conversation history starts
     with the prompt, the validation requests and answers, and the output.
5. Possible replies instead of the resumed run:
   - `<error>` "You are not allowed to validate AITask X." when the user lacks `patch`. The task
     keeps waiting.
   - "This validation was already answered." when someone else answered first (in another chat or
     with kubectl). The chat then shows the new prompt, or says that the task is running again
     and closes.

#### Option 2: answer with kubectl

1. Set the answer annotation on the AITask. `--overwrite` is needed for every new answer, since
   the controller only reacts when the value changes:

   ```sh
   # Approve
   kubectl annotate ait daily-29374512 -n cattle-ai-agent-system --overwrite \
     ai.cattle.io/validation-response='{"response":"yes"}'

   # Cancel
   kubectl annotate ait daily-29374512 -n cattle-ai-agent-system --overwrite \
     ai.cattle.io/validation-response='{"response":"no"}'
   ```

   The value is JSON. Only `response` is required. To make sure you answer the request you
   reviewed, and not a newer one, add its id (`validationId` in the API response above):
   `{"response":"yes","validationId":"<id>"}`. A mismatching id is ignored. Don't set
   `"via":"chat"`: those answers are left to the websocket and are not resumed by the controller.

2. The controller resumes the run in the background (no streaming). Follow it with:

   ```sh
   kubectl get ait daily-29374512 -n cattle-ai-agent-system -w
   ```

   `VALIDATION` goes back to `false` right away, and the phase ends `Succeeded`/`Failed`, or
   `VALIDATION` is `true` again if the task asks for another validation.
3. The output, and the answers with `"via": "kubectl"`, are in `GET /v1/api/aitasks/<name>`
   (`output`, `validations`). The follow-up chat on the task can be opened as usual.

An annotation set while the task is not waiting (or with invalid JSON) is ignored and logged by the
controller.

### Storage (`app/services/memory.py`)

A LangGraph `BaseStore` is added next to the checkpointer to hold task outputs. The outputs can be
large, so they are kept out of the CR status.

| Backend | When | Notes |
| :--- | :--- | :--- |
| Postgres (`AsyncPostgresStore`) | `DB_ENABLED=true` | Takes precedence. |
| SQLite (`AsyncSqliteStore`) | `SQLITE_ENABLED=true` | WAL mode plus `busy_timeout` so the FastAPI and kopf loops can share the file. `SQLITE_PATH` defaults to `/tmp/ai-agent-store.db`. |
| In-memory | Otherwise | One process-wide instance shared by both loops. Lost on restart. |

FastAPI and kopf run on **different event loops**, so each loop opens its own pool or connection.
The kopf side does this in `on_startup`/`on_cleanup`.

Helm: `storage.sqlite.{enabled,path}` and `storage.sqlite.pvc.{existingClaim,mountPath}`. When a PVC
is set, it is mounted, the DB goes to `<mountPath>/ai-agent-store.db`, and the deployment uses the
`Recreate` strategy (RWO volume, single writer).

SQLite was added as a short-term option for the POC because of the open Postgres issues (see
below).

### REST API (`app/routers/ai_task.py`)

| Endpoint | Description |
| :--- | :--- |
| `GET /v1/api/aitaskdefinitions` | Definitions the user can access, sorted by display name. |
| `POST /v1/api/aitaskdefinitions/{name}/run` | Run now, see [Run now](#run-now). |
| `GET /v1/api/aitasks?definition=<name>` | Tasks of a definition the user can access, newest first. The definition name is validated as a label value. |
| `GET /v1/api/aitasks/{name}` | Task summary plus `prompt`, `agents`, `output`/`error` from the store, and `chatId` of the user's follow-up conversation (see [Follow-up conversation](#follow-up-conversation)). |

Authorization is delegated to Rancher. Resources are read through Steve (`/v1/ai.cattle.io.aitask...`)
with the caller's `R_SESS` token, so Rancher applies the user's and group roles, including
`resourceNames`. `rbac.py` was refactored into generic `accessible_resources` /
`accessible_resource` helpers with `AccessDenied` (403) and `NotFound` (404) errors. These
checks are always on and do not depend on `RBAC_ENABLED`.

### Follow-up conversation

A user who can read an AITask can continue a conversation from its output, for example "why is
cluster X critical?". There is **one private conversation per AITask run and per user**.

- The client opens the normal chat websocket with the task:
  `/v1/ws/messages?aitask=<aitask name>`. The task is read through Steve as the user (same check as
  `GET /aitasks/{name}`). It must be `Succeeded` or `Failed`, or waiting for validation (see
  [Human validation in tasks](#human-validation-in-tasks)). Otherwise a `<chat-error>` is sent and
  the socket is closed. This also happens when `AGENTIC_TASK_ENABLED` is off.
- The thread id is deterministic, `aitask-<task uid>-<sha256(user id)[:16]>`, so opening the same
  task again continues the same conversation, and users never share one. A `thread_id` in the path
  is ignored in this mode. The id is returned as `chatId` by `GET /aitasks/{name}`.
- On the first turn of an empty thread, the user's message is preceded by two seed messages: the
  task prompt (`HumanMessage`) and the task output, or `The task failed: <error>` (`AIMessage`).
  The output is read from the store. The seed goes into `messages` (LLM context) and
  `messages_history` (chat history), flagged with `additional_kwargs.aitask_seed`.
- From then on it is a normal chat. The follow-up runs **as the user asking**: their agents, their
  token and RBAC, the interactive prompts and human validation. It is listed in `GET /chats` (named
  `Task <definition>: <task>` from the seed's `labels.summary`) and can be reopened with
  `/v1/ws/messages/{chatId}`. It only shows up in the chat list after the first question.
- Storage: the task output stays in the store. The conversation lives in the chat checkpointer
  (`AsyncPostgresSaver` with `DB_ENABLED=true`, otherwise in memory). With SQLite only the store is
  persistent, so follow-up chats are lost on restart, but the task output is not.

Code: `app/services/aitask_chat.py`, websocket handling in `app/routers/websocket.py`.

### Known limitations of the POC

- **No real identity for task runs.** Tasks run with `user_id="aitask"`. For `RANCHER` auth agents
  the MCP client falls back to the `RANCHER_API_TOKEN` env var (dev only), because there is no user
  session. `OAUTH2` agents are skipped.
- Paused runs waiting for validation are lost on restart (in-memory checkpointer) and the task is
  failed. Approvals don't expire, and nothing notifies users that a task is waiting.
- There is no create/update API for definitions (only `kubectl`).
- There is no retention/cleanup of old AITasks or stored outputs.
- The in-memory checkpointer means a task interrupted by a pod restart can't be resumed. It is just
  never re-run (`Running` is skipped on resume).
- It only works with a single replica (the scheduler runs in every replica, SQLite is a single
  writer, and paused runs and validation claims are kept in process memory).
- The same LLM as interactive chat is used.
- Follow-up conversations use the user's own agents, so they may see different data than the task
  run did. Reopening one with `/v1/ws/messages/{chatId}` does not check access to the AITask again
  (its content is already in the user's chat). Deleting the AITask does not delete them. Supervisor
  child threads (`::child::`) don't get the seed; the supervisor passes the context it needs when it
  delegates.

## Gaps against the v1 scope

| v1 item | POC status |
| :--- | :--- |
| Scheduled runs | Done. |
| Run now | Done. See [Run now](#run-now). |
| One service account per task | Missing. See [RBAC](#rbac-how-to-use-permissions). |
| Non-interactive MCP agents only | Partly. `OAUTH2` agents are skipped, but the reason is not shown. |
| Action level (Report / Propose / Fix) | Partly. Tools that need validation pause the run until a user with `patch` on the AITask answers, see [Human validation in tasks](#human-validation-in-tasks). No per-definition action level. |
| Mode (adaptive or single agent) | Partly. No `spec.mode`, planner always disabled. |
| Conversation per task | Partly. One follow-up conversation per run and per user, see [Follow-up conversation](#follow-up-conversation). No shared conversation per definition, see [Task conversation](#task-conversation). |
| Run summary (✅/⚠️/🔴, findings, actions, skipped checks) | Partly. Only `RESULT:` is parsed. |
| Notifications | Missing. |
| Actions recorded per task and service account | Missing. |
| Slack, email, tickets | Missing. See [Delivery](#delivery-slack-email-tickets). |
| Manage tasks (pause, edit, run now, delete) | Partly. Run now and read API, the rest via `kubectl`. |
| Admin-only task creation | Missing. |

## Open questions / next steps

### Delivery (Slack, email, tickets)

Slack and email don't need an MCP agent: the executor can send the report directly after the run,
using a Slack webhook/bot token or SMTP credentials stored in a Secret. Tickets (Jira, ServiceNow,
...) can use a third party agent. Questions:

- Add a `spec.delivery` field (target, channel/recipients, Secret ref) instead of relying on the
  prompt, so delivery is guaranteed.
- When to send: always, only on `Warning`/`Critical`, or only when the result changes.
- A failed delivery should not fail the task, but must be recorded.
- What is sent: full report, or a short summary with a link to Rancher (avoids leaking data and
  message size limits).
- Who may reference the delivery Secrets.

### Task conversation

A private follow-up conversation per run and per user is implemented (see
[Follow-up conversation](#follow-up-conversation)). Still open:

- A shared conversation per definition: created when the definition is saved, with its id in
  `status`. Each run appends its summary, and admins of the task can read it. This needs a thread
  owned by the definition instead of a user, which the chat endpoints and the checkpointer
  `user_id` filter don't support today.
- Approvals are given from the per-user follow-up conversation (see
  [Human validation in tasks](#human-validation-in-tasks)). They don't survive a restart until a
  persistent checkpointer is used.

### RBAC: how to use permissions?

A task must run with a well-defined identity, and that identity's permissions must limit what the
agents can see and do in Rancher. We **can't use the agent service account**: it is too privileged,
it is not a Rancher user, and it would let anyone who can create an AITaskDefinition act with the
agent's permissions (privilege escalation).

Options:

1. **Dedicated user and token in a Secret.** Admins create a Rancher user, possibly one per task
   or team, with the needed roles, generate an API token, and store it in a Secret in
   `cattle-ai-agent-system`. The definition references it (`credentialsRef`, Secrets labelled
   `ai.cattle.io/task-credentials`; the types already exist in `common_types.go`).
   - Pros: works today, simple, the permissions are explicit.
   - Cons: long-lived tokens to rotate and protect, manual setup, and the task does not run "as"
     the user who created it. We would need to check that the user who creates a definition is
     allowed to use that Secret, otherwise this is privilege escalation again.
2. **Implement CIBA (Client-Initiated Backchannel Authentication) in the Rancher OIDC provider.**
   The agent asks for a token on behalf of the user who owns the task. The user approves the
   request out of band (notification or UI), and the agent gets a short-lived token with the user's
   permissions.
   - Pros: no stored long-lived credentials, the task runs as a real user, and there is a clear
     audit trail.
   - Cons: needs changes in Rancher's OIDC provider. Approval needs a user, which conflicts with
     fully unattended schedules unless approval can be granted once per definition or a refresh
     token is allowed.

Still to decide: who "owns" a definition (creator vs. explicit identity), how to stop users from
referencing credentials they shouldn't use, and how this works for non-Rancher MCP auth types
(`BASIC`, `HEADER`, `OAUTH2`).

### Postgres storage

The store already supports `AsyncPostgresStore` when `DB_ENABLED=true`, but this path is not
validated. The **existing Postgres issues must be fixed first**. After that:

- Validate the Postgres store from both event loops (FastAPI and kopf), each with its own pool.
- Use a persistent checkpointer for task runs, so interrupted runs (or runs waiting for human
  validation) can be resumed.
- Decide whether SQLite stays as an option or is removed once Postgres works.
- Define retention for stored outputs and old AITask CRs (TTL, keep last N per definition).

### Human validation

A tool that needs human validation pauses the run until a user answers (see
[Human validation in tasks](#human-validation-in-tasks)). Still open:

- Should tasks be **read-only** by default, with write tools disabled instead of interrupted?
- Notifying users that a task waits for approval. **CIBA could help** here: the backchannel authentication request could also be the approval step
  (binding message: "Liz wants to scale deployment X, approve?").
- **Stale approvals**: what happens if the cluster state changed between the request and the
  approval, so the action no longer makes sense? Options:
  - Re-validate before executing: re-read the target resources (resourceVersion / preconditions)
    and abort, or re-plan, if they changed.
  - Make the approval expire after a timeout, and fail or skip the task.
  - Re-run the analysis after approval and only execute if it still reaches the same conclusion.
- A persistent checkpointer (see Postgres) is needed to keep the interrupted run across restarts.

### Choose a different LLM for agentic tasks

Unattended tasks may need a different model than interactive chat (cheaper for routine checks,
or stronger for complex analysis, with no latency constraint). Options:

- A global setting for agentic tasks (env or `llm-config`) that overrides the chat LLM.
- A per-definition `spec.model` (or provider + model) field, validated against the configured
  providers.

`LLMManager` is a singleton today, so the executor would need a way to build a separate LLM
instance.

### Limits

Unattended runs need guardrails against runaway cost and loops:

- **Max execution time** per task (today only kopf's 1h handler timeout applies, and it does not
  set a clean `Failed` status).
- **Max tool calls** / max agent steps (LangGraph recursion limit, or a middleware counter).
- **Max concurrent tasks** globally and per definition (overlap is already prevented per
  definition).
- Minimum schedule interval, to avoid `* * * * *` definitions.
- Output size limit for the stored report.

Still to decide: global defaults vs. per-definition overrides (`spec.limits`), and what the task
status shows when a limit is hit (for example `Failed` with reason `LimitExceeded` and partial
output).
