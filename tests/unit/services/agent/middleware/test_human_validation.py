"""Unit tests for human_validation middleware."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent.middleware.human_validation import (
    _build_interrupt_ui_tools,
    _process_tool_result,
    _should_interrupt,
)


# ---------------------------------------------------------------------------
# _should_interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_should_interrupt_returns_empty_when_tool_not_in_list():
    result = await _should_interrupt(
        human_validation_tools=["applyResource"],
        tool_call={"name": "listPods", "args": {}},
        planning_tools_by_name={},
    )
    assert result == ""


@pytest.mark.asyncio
async def test_should_interrupt_returns_empty_for_empty_validation_list():
    result = await _should_interrupt(
        human_validation_tools=[],
        tool_call={"name": "applyResource", "args": {}},
        planning_tools_by_name={},
    )
    assert result == ""


@pytest.mark.asyncio
async def test_should_interrupt_raises_when_plan_tool_missing():
    with pytest.raises(ValueError, match="planning tool 'applyResourcePlan' not found"):
        await _should_interrupt(
            human_validation_tools=["applyResource"],
            tool_call={"name": "applyResource", "args": {}},
            planning_tools_by_name={},
        )


@pytest.mark.asyncio
async def test_should_interrupt_returns_confirmation_response():
    plan_tool = AsyncMock()
    plan_tool.ainvoke.return_value = '{"type": "update", "resource": {"kind": "Pod"}}'

    result = await _should_interrupt(
        human_validation_tools=["applyResource"],
        tool_call={"name": "applyResource", "args": {"manifest": "..."}},
        planning_tools_by_name={"applyResourcePlan": plan_tool},
    )

    assert result.startswith("<confirmation-response>")
    assert result.endswith("</confirmation-response>")
    inner = result.removeprefix("<confirmation-response>").removesuffix("</confirmation-response>")
    parsed = json.loads(inner)
    assert parsed["type"] == "update"
    plan_tool.ainvoke.assert_awaited_once_with({"manifest": "..."})


@pytest.mark.asyncio
async def test_should_interrupt_handles_list_text_response():
    plan_tool = AsyncMock()
    plan_tool.ainvoke.return_value = [{"text": '{"type": "create"}'}]

    result = await _should_interrupt(
        human_validation_tools=["applyResource"],
        tool_call={"name": "applyResource", "args": {}},
        planning_tools_by_name={"applyResourcePlan": plan_tool},
    )

    inner = result.removeprefix("<confirmation-response>").removesuffix("</confirmation-response>")
    assert json.loads(inner)["type"] == "create"


@pytest.mark.asyncio
async def test_should_interrupt_handles_non_json_response():
    plan_tool = AsyncMock()
    plan_tool.ainvoke.return_value = "plain text plan"

    result = await _should_interrupt(
        human_validation_tools=["applyResource"],
        tool_call={"name": "applyResource", "args": {}},
        planning_tools_by_name={"applyResourcePlan": plan_tool},
    )

    inner = result.removeprefix("<confirmation-response>").removesuffix("</confirmation-response>")
    assert json.loads(inner) == "plain text plan"


# ---------------------------------------------------------------------------
# _build_interrupt_ui_tools
# ---------------------------------------------------------------------------


def _make_config(ui_tools_name="show-yaml"):
    return {
        "configurable": {
            "request_metadata": {
                "ui_tools": {"name": ui_tools_name},
            },
        },
    }


def _make_interrupt_message(data):
    return f"<confirmation-response>{json.dumps(data)}</confirmation-response>"


@patch("app.services.agent.middleware.human_validation._dispatch_ui_tools")
def test_build_ui_tools_returns_empty_when_no_ui_tools_config(mock_dispatch):
    result = _build_interrupt_ui_tools(
        interrupt_message="<confirmation-response>{}</confirmation-response>",
        state={},
        config={"configurable": {}},
    )
    assert result == []
    mock_dispatch.assert_not_called()


@patch("app.services.agent.middleware.human_validation._dispatch_ui_tools")
def test_build_ui_tools_returns_empty_when_name_is_empty(mock_dispatch):
    config = {"configurable": {"request_metadata": {"ui_tools": {"name": ""}}}}
    result = _build_interrupt_ui_tools(
        interrupt_message="<confirmation-response>{}</confirmation-response>",
        state={},
        config=config,
    )
    assert result == []
    mock_dispatch.assert_not_called()


@patch("app.services.agent.middleware.human_validation._dispatch_ui_tools")
def test_build_ui_tools_creates_show_yaml_for_create_type(mock_dispatch):
    data = {
        "type": "create",
        "resource": {"kind": "Deployment", "name": "nginx", "namespace": "default"},
        "payload": {"apiVersion": "apps/v1", "kind": "Deployment"},
    }
    result = _build_interrupt_ui_tools(
        interrupt_message=_make_interrupt_message(data),
        state={},
        config=_make_config(),
    )

    assert len(result) == 1
    assert result[0]["toolName"] == "show-yaml"
    assert result[0]["input"]["resourceKind"] == "Deployment"
    assert result[0]["input"]["resourceName"] == "nginx"
    assert result[0]["input"]["resourceNamespace"] == "default"
    assert result[0]["input"]["yaml"] == {"apiVersion": "apps/v1", "kind": "Deployment"}
    mock_dispatch.assert_called_once_with(result)


@patch("app.services.agent.middleware.human_validation._dispatch_ui_tools")
def test_build_ui_tools_creates_show_yaml_diff_for_update_type(mock_dispatch):
    data = {
        "type": "update",
        "resource": {"kind": "Service", "name": "my-svc"},
        "payload": {
            "original": {"spec": {"replicas": 1}},
            "patched": {"spec": {"replicas": 3}},
        },
    }
    result = _build_interrupt_ui_tools(
        interrupt_message=_make_interrupt_message(data),
        state={},
        config=_make_config(),
    )

    assert len(result) == 1
    assert result[0]["toolName"] == "show-yaml-diff"
    assert result[0]["input"]["original"] == {"spec": {"replicas": 1}}
    assert result[0]["input"]["patched"] == {"spec": {"replicas": 3}}
    mock_dispatch.assert_called_once_with(result)


@patch("app.services.agent.middleware.human_validation._dispatch_ui_tools")
def test_build_ui_tools_handles_list_data(mock_dispatch):
    data = [
        {
            "type": "create",
            "resource": {"kind": "Pod", "name": "test-pod"},
            "payload": {"kind": "Pod"},
        }
    ]
    result = _build_interrupt_ui_tools(
        interrupt_message=_make_interrupt_message(data),
        state={},
        config=_make_config(),
    )

    assert len(result) == 1
    assert result[0]["toolName"] == "show-yaml"
    assert result[0]["input"]["resourceKind"] == "Pod"


@patch("app.services.agent.middleware.human_validation._dispatch_ui_tools")
def test_build_ui_tools_filters_none_values(mock_dispatch):
    data = {
        "type": "create",
        "resource": {"kind": "Pod"},
        "payload": {},
    }
    result = _build_interrupt_ui_tools(
        interrupt_message=_make_interrupt_message(data),
        state={},
        config=_make_config(),
    )

    assert "resourceName" not in result[0]["input"]
    assert "resourceNamespace" not in result[0]["input"]


# ---------------------------------------------------------------------------
# _process_tool_result
# ---------------------------------------------------------------------------


@patch("app.services.agent.middleware.human_validation.dispatch_custom_event")
def test_process_tool_result_plain_string(mock_dispatch):
    content, mcp_response, mcp_data = _process_tool_result("plain text", {})

    assert content == "plain text"
    assert mcp_response is None
    assert mcp_data is None
    mock_dispatch.assert_not_called()


@patch("app.services.agent.middleware.human_validation.dispatch_custom_event")
def test_process_tool_result_json_without_llm_or_ui(mock_dispatch):
    data = {"pods": ["a", "b"]}
    content, mcp_response, mcp_data = _process_tool_result(json.dumps(data), {})

    assert content == json.dumps(data)
    assert mcp_response is None
    assert mcp_data == json.dumps(data)
    mock_dispatch.assert_not_called()
