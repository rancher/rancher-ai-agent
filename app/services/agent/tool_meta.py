"""Helpers for reading the metadata an MCP server attaches to its tools."""

from langchain_core.tools import BaseTool

# Set to true in the ``_meta`` of every MCP tool that changes resources, so that the user is asked to confirm
# the change before the tool is called. The change is previewed by the ``<tool_name>Plan`` tool.
REQUIRES_HUMAN_VALIDATION_META_KEY = "requiresHumanValidation"


def get_tool_meta(tool: BaseTool) -> dict:
    """Return the MCP ``_meta`` of a tool loaded through ``langchain.mcp`` (stored under ``metadata["mcp"]["tool"]``)."""
    return ((((tool.metadata or {}).get("mcp") or {}).get("tool") or {}).get("_meta")) or {}


def requires_human_validation(tool: BaseTool) -> bool:
    """Return True if the MCP server marked the tool as one that changes resources."""
    return get_tool_meta(tool).get(REQUIRES_HUMAN_VALIDATION_META_KEY) is True
