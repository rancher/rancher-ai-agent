CONTEXT_PARAMETERS_SUFFIX = (
    "\n\n  Use the following parameters to populate tool calls when appropriate. \n"
)

INTERRUPT_CANCEL_REPLY = "Previous tool canceled by the user."

# Control message the client sends to stop the currently running agent execution.
STOP_MESSAGE = "<stop>"

# Message injected as the tool result when a running execution is stopped by the user.
STOP_CANCEL_REPLY = "Execution stopped by the user."
