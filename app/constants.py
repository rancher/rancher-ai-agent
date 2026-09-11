CONTEXT_PARAMETERS_SUFFIX = (
    "\n\n  Use the following parameters to populate tool calls when appropriate. \n"
)

INTERRUPT_CANCEL_REPLY = "Previous tool canceled by the user."

# Reply shown when a human-validation is canceled while a plan is being executed.
# Kept in the chat history so a follow-up "yes" lets the LLM revise the plan.
PLAN_CANCEL_REPLY = (
    "The current plan has been canceled. Would you like me to revise and propose a "
    "new plan that avoids the task you've recently canceled?"
)

# LLM-facing version of the plan-cancellation reply. Stored as the AIMessage content so
# the model knows exactly which step was rejected and must avoid it when the user asks to
# continue. The user still only sees PLAN_CANCEL_REPLY (dispatched live and surfaced via
# the ``display_message`` kwarg on reload). ``{task}`` is the canceled step description.
PLAN_CANCEL_LLM_REPLY = (
    "The current plan has been canceled. Would you like me to revise and propose a "
    "new plan that avoids the task you've recently canceled?\n\n"
    "(Internal guidance — do not repeat to the user: the canceled step was \"{task}\". "
    "If the user asks to continue, propose a NEW plan that reaches their goal WITHOUT "
    "this step or any equivalent action, and request approval before executing it. "
    "Do not simply re-propose the previous plan.)"
)

# Control message the client sends to stop the currently running agent execution.
STOP_MESSAGE = "<stop>"

# Message injected as the tool result when a running execution is stopped by the user.
STOP_CANCEL_REPLY = "Execution stopped by the user."
