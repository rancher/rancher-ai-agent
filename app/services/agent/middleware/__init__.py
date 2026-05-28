"""
Middleware package for agent middleware factories and classes.

See README.md for an overview of the middleware system and how to add new middleware.
"""

from ._constants import INTERRUPT_CANCEL_MESSAGE
from .messages_history import MessagesHistoryMiddleware
from .inject_kwargs import inject_additional_kwargs_middleware
from .cancel_human_validation import cancel_human_validation_middleware
from .ui_tools import (
    ui_tools_middleware,
    _dispatch_ui_tools,
    _dispatch_ui_tools_event,
    _collect_context_until_human,
    _extract_tool_text,
)
from .identity_preamble import identity_preamble_middleware
from .human_validation import (
    human_validation_middleware,
    _should_interrupt,
    _build_interrupt_ui_tools,
    _process_tool_result,
    convert_to_string_if_needed,
)
