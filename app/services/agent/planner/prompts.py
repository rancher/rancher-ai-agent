"""Prompts and user-facing replies used by the planner agent."""

PLANNER_PROMPT = """\
You are a planning agent. Break the user's request into an ordered list of subtasks.
Assign each subtask to exactly one of the available agents (use the agent name).

Available agents:
{agents}

Return a plan where each subtask has a clear, self-contained task description and the
name of the agent best suited to perform it.

Rules you MUST follow:
- Always return at least one subtask. If the request is simple, return exactly one
  subtask that covers the whole request.
- If the request is complex, break it down into as many subtasks as needed so each one
  covers a single, self-contained piece of work. Do not bundle unrelated or multi-step
  work into a single subtask when it can reasonably be split,
- The "agent" field of every subtask MUST be one of the agent names listed above,
  copied exactly (case-sensitive). Do not invent new agent names.
- Respond with a single, valid JSON object only. Do not add explanations, comments,
  markdown code fences, or any text before or after the JSON.
"""

REDUCER_PROMPT = """\
You are summarizing the outcome of a multi-step plan for the user.

Original request:
{request}

Results of each subtask:
{results}

Combine these into a single, coherent final answer for the user.
"""

REDUCER_SYSTEM_PROMPT = """\
You are the reducer of a planner agent. You are given the original user request and the
results of each subtask that was executed. Combine them into a single, coherent final
answer for the user.
"""

SUBTASK_EVALUATION_SYSTEM_PROMPT = """\
You are the evaluator of a planner agent. You are given a subtask that was assigned to a
child agent and the response that agent produced. Judge whether the agent actually
completed the task. Answer with a single word: "yes" if it completed the task, "input" if
it is asking the user for information it needs to continue, or "no" if it did not complete
the task. Do not add any other text.
"""

SUBTASK_EVALUATION_PROMPT = """\
Did the agent complete the assigned subtask?

Subtask:
{task}

Agent's response:
{response}

If the agent is asking the user a question or requesting information or a decision it
needs in order to continue (e.g. a missing name or value), it is waiting for user input.
The task was NOT completed if the agent hit an error, lacks the capability or permissions,
refused, or otherwise did not accomplish what was asked. If the agent accomplished the task
and produced a useful result, it was completed.

Answer with a single word: "yes" if it completed the task, "input" if it is waiting for
user input, or "no" if it did not complete the task.
"""

PLANNER_FEEDBACK_SUFFIX = """\

The user reviewed previous versions of this plan and requested changes. Produce a new
plan that takes ALL of the following feedback into account, in order (later feedback
refines or overrides earlier feedback):
{feedback}
"""

PLANNER_RETRY_SUFFIX = """\

---
The previous plan attempt failed and execution was stopped early:
{details}

Treat the new user message above as the user's chosen way to recover, and produce a plan
accordingly:
- If the user wants to retry the step that failed, produce a plan that reattempts that
  failed step (incorporating any new or corrected information the user provided), followed
  by the remaining steps needed to satisfy the original request.
- If the user wants to start from the beginning, produce a plan that re-runs the ENTIRE
  original request from the first step, without assuming any step of the previous attempt
  is still valid.
- If the user wants to create a new plan, treat their message as new or corrected
  information and produce a brand-new, complete plan that covers the entire original
  request.
"""

PLANNER_CANCEL_SUFFIX = """\

---
The previous plan attempt was cancelled by the user and execution was stopped early:
{details}

Treat the new user message above as the user's chosen way to recover, and produce a plan
accordingly:
- If the user wants to retry the step that was cancelled, produce a plan that reattempts
  that cancelled step (incorporating any new or corrected information the user provided),
  followed by the remaining steps needed to satisfy the original request.
- If the user wants to start from the beginning, produce a plan that re-runs the ENTIRE
  original request from the first step, without assuming any step of the previous attempt
  is still valid.
- If the user wants to create a new plan, treat their message as new or corrected
  information and produce a brand-new, complete plan that covers the entire original
  request.
"""

PLAN_FAILED_PREFIX = "PLAN FAILED:"

# These four options are surfaced to the client as the failed subtask's "actions" and,
# if the user picks one, sent straight back as the next request: _create_plan and
# _handle_failure_actions match on their exact text to trigger the corresponding
# recovery path (retry, restart, request details, cancel) without asking the LLM to
# generate a brand-new plan.
RETRY_SUBTASK_REQUEST = "Retry executing the failed subtask"
RESTART_PLAN_REQUEST = "Restart the execution of the entire plan"
REQUEST_FAILURE_DETAILS = "Request more details about the failure"
CANCEL_PLAN_REQUEST = "Cancel the plan"

PLAN_CANCELLED_REPLY = (
    "The current plan has been cancelled as per your request."
)
PLAN_FAILED_REPLY = (
    "There was a problem generating a plan for your request. Please try again with a different prompt."
)

PLAN_SUBTASK_FAILED_REPLY = (
    PLAN_FAILED_PREFIX + ' I couldn\'t complete the step "{task}": {error}\n\n'
    "Here is the plan that was being executed:\n{plan}\n\n"
    "I've stopped the plan here instead of continuing with the remaining steps, so the "
    "results stay consistent. Please provide any missing information or adjust your "
    "request, and I'll create a new plan."
)

PLAN_CANCELLED_PREFIX = "PLAN CANCELLED:"

PLAN_SUBTASK_CANCELLED_REPLY = (
    PLAN_CANCELLED_PREFIX + ' You cancelled the step "{task}".\n\n'
    "Here is the plan that was being executed:\n{plan}\n\n"
    "I've stopped the plan here instead of continuing with the remaining steps, so the "
    "results stay consistent. Let me know if you'd like to retry it, restart the plan, or "
    "adjust your request, and I'll create a new plan."
)
