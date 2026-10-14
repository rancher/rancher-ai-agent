import json

CONTEXT_PARAMETERS_SUFFIX = (
    "\n\n  Use the following parameters to populate tool calls when appropriate. \n"
)

INTERRUPT_CANCEL_REPLY = "Previous tool canceled by the user."

# Tag the client includes in a request's ``tags`` to stop the currently running agent execution.
STOP_TAG = "stop"

# Reasons a running execution can be stopped.
STOP_REASON_USER = "user-stop"
STOP_REASON_NEW_REQUEST = "new-request"

# Message injected as the tool result when a running execution is stopped by the user.
STOP_CANCEL_REPLY = "Execution stopped by the user."
