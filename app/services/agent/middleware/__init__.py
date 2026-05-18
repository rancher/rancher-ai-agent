"""
Middleware package for agent middleware factories and classes.

See README.md for an overview of the middleware system and how to add new middleware.
"""

from ._constants import INTERRUPT_CANCEL_MESSAGE, ChildAgentCancelled
from .messages_history import MessagesHistoryMiddleware
from .inject_kwargs import inject_additional_kwargs_middleware
from .cancel_check import create_cancel_check_middleware
from .ui_tools import (
    create_ui_tools_middleware,
    _dispatch_ui_tools,
    _dispatch_ui_tools_event,
    _collect_context_until_human,
    _extract_tool_text,
)
from .child_agent_tool import _create_child_agent_middleware
from .identity_preamble import _create_identity_preamble_middleware
from .tool_execution import (
    _create_tool_execution_middleware,
    _should_interrupt,
    _build_interrupt_ui_tools,
    _build_agent_metadata,
    _process_tool_result,
    convert_to_string_if_needed,
)

__all__ = [
    "INTERRUPT_CANCEL_MESSAGE",
    "ChildAgentCancelled",
    "MessagesHistoryMiddleware",
    "inject_additional_kwargs_middleware",
    "create_cancel_check_middleware",
    "create_ui_tools_middleware",
    "_dispatch_ui_tools",
    "_dispatch_ui_tools_event",
    "_collect_context_until_human",
    "_extract_tool_text",
    "_create_child_agent_middleware",
    "_create_identity_preamble_middleware",
    "_create_tool_execution_middleware",
    "_should_interrupt",
    "_build_interrupt_ui_tools",
    "_build_agent_metadata",
    "_process_tool_result",
    "convert_to_string_if_needed",
]
