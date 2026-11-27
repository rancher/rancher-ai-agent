# Spike: Agentic Tasks (POC)

This document summarizes the proof of concept on the `agentic-tasks` branch and the open questions
to answer before it becomes a real feature.

## Goal

Let Liz run **unattended tasks**: a prompt (for example "check the health of all downstream
clusters and report problems") that runs on a schedule or on demand, with nobody watching. A user
reads the result later in the UI and can ask follow-up questions about it.

## Summary of the POC

Everything is behind the feature flag `AGENTIC_TASK_ENABLED` (Helm: `agenticTask.enabled`).

```
AITaskDefinition ──(schedule or "run now")──▶ AITask ──(executor)──▶ Liz agents ──▶ stored report
                                                                                        │
                       UI / API (read as the calling user, Rancher RBAC) ◀───────────────┘
```

| Concept | What it is |
| :--- | :--- |
| **AITaskDefinition** | The template: prompt, schedule (cron, optional), time zone, allowed agents, suspend flag, and the credentials the task runs with. |
| **AITask** | One run of a definition. Has a phase (`Pending`, `Running`, `Succeeded`, `Failed`) and a result (`Healthy`, `Warning`, `Critical`). Deleted together with its definition. |
| **Report** | The full output of the run, kept in the agent's store (not in the CR, because it can be large). |

### What works

- **Scheduled runs.** Works like a small CronJob: no overlapping runs of the same definition, missed
  runs are not caught up (only a run that is at most 5 minutes late is started), and a run is never
  executed twice.
- **Run now.** `POST /v1/api/aitaskdefinitions/{name}/run` starts a run, also for suspended or
  manual-only definitions. The user needs `patch` on the definition.
- **Unattended execution.** The task uses a single agent or the supervisor, without the planner.
  Liz gets a special prompt: never ask questions, prefer read-only actions, write a structured
  report, and end with a `Healthy`/`Warning`/`Critical` verdict.
- **Task credentials.** Each definition references a Secret with the Rancher token of a dedicated
  user. The task only sees and does what that user can. See [Task identity](#task-identity-token-secret-vs-ciba).
- **Human validation.** If a tool needs approval, the run pauses instead of failing. Any user with
  `patch` on the AITask can approve or reject it from the follow-up chat or with `kubectl`. Only the
  first answer counts.
- **Follow-up conversation.** A user who can read an AITask can open a private chat about its
  result ("why is cluster X critical?"). There is one conversation per run and per user. It runs
  with the user's own agents and permissions, and shows up in the normal chat list.
- **Read API.** List definitions, list the tasks of a definition, and read one task with its
  report. Access is always checked by Rancher as the calling user.
- **Storage.** Postgres (not validated yet), SQLite (added for the POC, can use a PVC), or in-memory.

### Known limitations

- A run paused for validation, or interrupted by a restart, is lost on restart and the task is
  marked `Failed`. Approvals don't expire and nobody is notified that a task is waiting.
- Anyone who can create or edit a definition can reference any task credentials Secret.
- No create/update API for definitions (only `kubectl`), no cleanup of old tasks and reports.
- The task uses the same LLM as the chat.

## Try it

```sh
# 1. Token of a dedicated Rancher user (e.g. read-only on the clusters), in a labelled Secret
kubectl create secret generic liz-task-health-token -n cattle-ai-agent-system \
  --from-literal=token='token-xxxxx:yyyy'
kubectl label secret liz-task-health-token -n cattle-ai-agent-system ai.cattle.io/task-credentials=true

# 2. Definition
kubectl apply -f - <<EOF
apiVersion: ai.cattle.io/v1alpha1
kind: AITaskDefinition
metadata:
  name: daily-health
  namespace: cattle-ai-agent-system
spec:
  displayName: Daily cluster health
  prompt: Check the health of all downstream clusters and report problems.
  schedule: "@daily"
  credentialsRef:
    name: liz-task-health-token
EOF

# 3. Run it now and follow it
kubectl annotate aitd daily-health -n cattle-ai-agent-system --overwrite \
  ai.cattle.io/run-requested-at="$(date -u +%FT%TZ)"
kubectl get ait -n cattle-ai-agent-system -w

# 4. If it waits for validation (VALIDATION=true), answer it
kubectl annotate ait <task> -n cattle-ai-agent-system --overwrite \
  ai.cattle.io/validation-response='{"response":"yes"}'
```

The report is in `GET /v1/api/aitasks/<task>`. The follow-up chat (and validation from the chat) is
at `/ui?aitask=<task>` in the test UI.

## Gaps against the v1 scope

| v1 item | Status |
| :--- | :--- |
| Scheduled runs | Done |
| Run now | Done |
| One identity per task | Partly: a user token per definition, from a Secret |
| Non-interactive MCP agents only | Partly: `OAUTH2` agents skipped silently |
| Action level (Report / Propose / Fix) | Partly: write actions pause for approval, no per-definition level |
| Mode (adaptive or single agent) | Done |
| Conversation per task | Partly: per run and per user, no shared conversation per definition |
| Run summary (✅/⚠️/🔴, findings, actions, skipped checks) | Partly: only the verdict is parsed |
| Manage tasks (pause, edit, run now, delete) | Partly: run now and read API, the rest via `kubectl` |
| Notifications | Missing |
| Actions recorded per task and identity | Missing |
| Slack, email, tickets | Missing |
| Admin-only task creation | Missing |

## Open questions

### Delivery (Slack, email, tickets)

Slack and email can be sent by the executor after the run, with credentials in a Secret. Tickets
(Jira, ServiceNow, ...) can use a third-party agent.

- A `spec.delivery` field, so delivery doesn't depend on the prompt?
- When to send: always, only on `Warning`/`Critical`, or only when the result changes?
- What to send: the full report, or a summary with a link to Rancher (less data leaked, no size
  limits)?
- A failed delivery should be recorded, but not fail the task.

### Shared conversation per definition

Today each user has a private conversation per run. A shared conversation per definition (each run
appends its summary, task admins can read it) needs a thread owned by the definition instead of a
user, which the chat doesn't support today.

### Human validation

- Should tasks be **read-only** by default, with write tools disabled?
- How are users notified that a task waits for approval? (CIBA could help, see below.)
- **Stale approvals**: the cluster may change between the request and the approval. Options:
  re-check the target resources before acting, expire approvals after a timeout, or re-run the
  analysis after approval.
- A persistent checkpointer is needed so paused runs survive restarts.

### Postgres storage

The existing Postgres issues must be fixed first. Then: validate the Postgres store, use it to keep
paused runs across restarts, decide whether SQLite stays, and define retention for old tasks and
reports (TTL or last N per definition).

### LLM for tasks

Tasks may need a different model than the chat (cheaper for routine checks, stronger for complex
analysis). Options: a global setting for tasks, or a per-definition model field.

### Limits

Guardrails against runaway cost and loops: max execution time, max tool calls / agent steps, max
concurrent tasks, minimum schedule interval, and max report size. To decide: global defaults vs.
per-definition overrides, and how a hit limit is shown (for example `Failed` with reason
`LimitExceeded` and the partial report).

## Task identity: token Secret vs CIBA

A task must run with a well-defined identity whose Rancher permissions limit what it can see and
do. The **agent's service account can't be used**: it is too privileged, it is not a Rancher user,
and anyone who can create a definition would act with the agent's permissions.

### Option 1: user token in a Secret (implemented)

An admin creates a dedicated Rancher user with only the roles the task needs, creates an API token
for it, and stores it in a labelled Secret. The definition references the Secret, and every run
uses that token.

**Pros**

- Works today, with no changes in Rancher.
- Simple to understand: the permissions of the task are the roles of that user.
- Fully unattended: nobody has to approve anything for a run to start.

**Cons**

- Long-lived token, stored in the cluster. If the Secret leaks, the token has all the rights of
  that user until it is revoked.
- Manual setup and manual rotation.
- Anyone who can create or edit a definition can reference any labelled Secret, which is privilege
  escalation unless we add a check (for example a webhook that verifies the user may read that
  Secret).
- Approvals for write actions are given by anyone with `patch` on the AITask, but the action is
  still executed with the task's token.

### Option 2: CIBA (needs to be implemented in Rancher)

CIBA (OpenID Connect Client-Initiated Backchannel Authentication) lets the agent ask a named user
for a token; the user approves the request on their own device. **Rancher's OIDC provider doesn't
support CIBA today, so it has to be implemented in Rancher first.** 

The preferred way to use it: a user **approves once** when the definition is set up. The agent
keeps a revocable refresh token (in a Secret it owns) and gets a short-lived token for each run, so
scheduled runs stay unattended. Write actions ask the approver for a fresh confirmation on their
device, which also solves the notification problem of human validation.

**Pros**

- No long-lived token copied by hand: the stored refresh token is bound to a real user, revocable,
  and possibly read-only.
- Runs use short-lived tokens, and rotation can be automatic.
- The task runs as a real person, with a clear audit trail of who authorized it and who approved
  each action.
- Approvals go to a named approver, who is notified on their device, instead of waiting for someone
  to open the chat.
- No privilege escalation through Secret references: the user who grants access is the one whose
  permissions are used.

**Cons**

- CIBA needs to be implemented in Rancher's OIDC provider
- More moving parts: OIDC client registration, refresh tokens, revocation, polling the token endpoint.
- The task stops working if the refresh token expires (for example after a long idle time), the
  user leaves, or the grant is revoked, and someone has to approve again.
- Write approvals need the approver to be available, so fixes can wait for a long time.

### Comparison

| | Token in a Secret | CIBA |
| :--- | :--- | :--- |
| Available | Yes, implemented in the POC | No, needs to be implemented in Rancher |
| Setup | Admin creates a user, a token and a Secret by hand | A user approves once on their device |
| Credential stored | Long-lived Rancher token | Revocable refresh token, short-lived tokens per run |
| Rotation | Manual | Automatic (if the provider rotates refresh tokens) |
| If the Secret leaks | Full rights of the user until revoked | Limited to the grant (read-only if scopes are supported), revocable |
| Task runs as | A dedicated technical user | The user who approved |
| Approving write actions | Anyone with `patch` on the AITask, via chat or `kubectl` | A named approver, notified on their device |
| Unattended scheduled runs | Yes | Yes, while the grant is valid |

