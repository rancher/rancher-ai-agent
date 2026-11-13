"""Unit tests for token_usage helpers."""

from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.services.agent.middleware.token_usage import (
    collect_usage,
    record_usage,
    usage_entry,
    usage_update_from_response,
)


_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_usage_entry_records_provider_model_and_agent(monkeypatch):
    monkeypatch.setenv("ACTIVE_LLM", "bedrock")
    msg = AIMessage(
        content="x", id="m1", usage_metadata=_USAGE,
        response_metadata={"model_name": "eu.anthropic.claude-opus-4-8"},
    )
    with patch(
        "app.services.agent.middleware.token_usage.get_config",
        return_value={"configurable": {"agent": "rancher"}},
    ):
        entry = usage_entry(msg, "agent")

    assert entry["provider"] == "bedrock"
    assert entry["model"] == "eu.anthropic.claude-opus-4-8"
    assert entry["agent"] == "rancher"


def test_usage_entry_without_active_llm_or_graph_context(monkeypatch):
    monkeypatch.delenv("ACTIVE_LLM", raising=False)
    entry = usage_entry(AIMessage(content="x", id="m1", usage_metadata=_USAGE), "agent")

    assert entry["provider"] is None
    assert entry["agent"] is None  # get_config raises outside a graph run


def test_usage_update_from_response_keys_by_message_id():
    update = usage_update_from_response(AIMessage(content="x", id="m1", usage_metadata=_USAGE), "planner")

    assert set(update) == {"m1"}
    assert update["m1"]["source"] == "planner"


def test_usage_update_from_response_without_usage():
    assert usage_update_from_response(None, "planner") is None
    assert usage_update_from_response(AIMessage(content="x", id="m1"), "planner") is None


def test_record_usage_collects_only_inside_collect_usage():
    msg = AIMessage(content="x", id="m1", usage_metadata=_USAGE)

    record_usage(msg, "planner")  # no active block: ignored
    with collect_usage() as collected:
        record_usage(msg, "planner")
        record_usage(msg, "planner")  # same id: recorded once
        record_usage(AIMessage(content="y", id="m2"), "planner")  # no usage: skipped

    assert set(collected) == {"m1"}
    assert collected["m1"]["total_tokens"] == 15


def test_fallback_reads_anthropic_cache_counters():
    msg = AIMessage(content="x", id="m1", response_metadata={"usage": {
        "input_tokens": 100, "output_tokens": 10,
        "cache_read_input_tokens": 60, "cache_creation_input_tokens": 20,
    }})

    entry = usage_update_from_response(msg, "agent")["m1"]

    assert entry["cache_read"] == 60
    assert entry["cache_creation"] == 20


def test_fallback_reads_openai_cached_tokens():
    msg = AIMessage(content="x", id="m1", response_metadata={"token_usage": {
        "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
        "prompt_tokens_details": {"cached_tokens": 40},
    }})

    entry = usage_update_from_response(msg, "agent")["m1"]

    assert entry["cache_read"] == 40
    assert entry["cache_creation"] == 0
