"""
Unit tests for the tool metadata helpers (app.services.agent.tool_meta).
"""
from langchain_core.tools import tool as langchain_tool

from app.services.agent.tool_meta import get_tool_meta, requires_human_validation


def _tool_with_metadata(metadata):
    @langchain_tool("someTool")
    def _tool(x: str = "") -> str:
        """A test tool."""
        return "result"

    _tool.metadata = metadata
    return _tool


def test_get_tool_meta_returns_mcp_meta():
    tool = _tool_with_metadata({"mcp": {"tool": {"_meta": {"toolset": "rancher"}}}})

    assert get_tool_meta(tool) == {"toolset": "rancher"}


def test_get_tool_meta_without_metadata_is_empty():
    assert get_tool_meta(_tool_with_metadata(None)) == {}
    assert get_tool_meta(_tool_with_metadata({})) == {}
    assert get_tool_meta(_tool_with_metadata({"mcp": {"tool": {}}})) == {}
    assert get_tool_meta(_tool_with_metadata({"mcp": {"tool": {"_meta": None}}})) == {}


def test_requires_human_validation():
    marked = _tool_with_metadata({"mcp": {"tool": {"_meta": {"requiresHumanValidation": True}}}})
    not_marked = _tool_with_metadata({"mcp": {"tool": {"_meta": {"requiresHumanValidation": False}}}})
    other_meta = _tool_with_metadata({"mcp": {"tool": {"_meta": {"toolset": "rancher"}}}})

    assert requires_human_validation(marked)
    assert not requires_human_validation(not_marked)
    assert not requires_human_validation(other_meta)
    assert not requires_human_validation(_tool_with_metadata(None))


def test_requires_human_validation_ignores_truthy_values_that_are_not_true():
    tool = _tool_with_metadata({"mcp": {"tool": {"_meta": {"requiresHumanValidation": "yes"}}}})

    assert not requires_human_validation(tool)
