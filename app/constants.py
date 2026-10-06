CONTEXT_PARAMETERS_SUFFIX = (
    "\n\n  Use the following parameters to populate tool calls when appropriate. \n"
)

INTERRUPT_CANCEL_REPLY = "Previous tool canceled by the user."

# Tag the client includes in a request's ``tags`` to stop the currently running agent execution.
STOP_TAG = "stop"

# Message sent to the client when a running execution has been stopped.
STOP_REPLY = "<stop>message stopped</stop>"

# Message injected as the tool result when a running execution is stopped by the user.
STOP_CANCEL_REPLY = "Execution stopped by the user."
