"""Unit tests for aitask_prompt_middleware."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.agent.middleware.aitask_prompt import aitask_prompt_middleware


def _run(middleware, base_prompt: str) -> str:
    request = MagicMock()
    request.system_prompt = base_prompt
    middleware.wrap_model_call(request, lambda r: r)
    return request.override.call_args.kwargs["system_message"].content


@pytest.mark.parametrize("configurable", [{}, {"aitask": False}, {"request_id": "r"}])
@patch("app.services.agent.middleware.aitask_prompt.get_config")
def test_prompt_unchanged_outside_aitask(mock_get_config, configurable):
    mock_get_config.return_value = {"configurable": configurable}
    assert _run(aitask_prompt_middleware(replace_with="TASK"), "BASE") == "BASE"
    assert _run(aitask_prompt_middleware(append=" EXTRA"), "BASE") == "BASE"


@patch("app.services.agent.middleware.aitask_prompt.get_config")
def test_prompt_replaced_in_aitask(mock_get_config):
    mock_get_config.return_value = {"configurable": {"aitask": True}}
    assert _run(aitask_prompt_middleware(replace_with="TASK"), "BASE") == "TASK"


@patch("app.services.agent.middleware.aitask_prompt.get_config")
def test_prompt_appended_in_aitask(mock_get_config):
    mock_get_config.return_value = {"configurable": {"aitask": True}}
    assert _run(aitask_prompt_middleware(append=" EXTRA"), "BASE") == "BASE EXTRA"
