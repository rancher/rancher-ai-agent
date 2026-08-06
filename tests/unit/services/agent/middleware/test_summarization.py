"""Unit tests for UsageTrackingSummarizationMiddleware."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from app.services.agent.middleware.summarization import UsageTrackingSummarizationMiddleware


_USAGE = {
    "input_tokens": 40,
    "output_tokens": 10,
    "total_tokens": 50,
    "input_token_details": {"cache_read": 5, "cache_creation": 0},
}


def _middleware(summary_response):
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=summary_response)
    model.invoke = MagicMock(return_value=summary_response)
    return UsageTrackingSummarizationMiddleware(
        model=model,
        trigger=[("messages", 30), ("tokens", 30000)],
        keep=("messages", 15),
        trim_tokens_to_summarize=None,
    )


@pytest.mark.asyncio
async def test_acreate_summary_with_usage_returns_text_and_usage():
    response = AIMessage(content="the summary", id="sum-1", usage_metadata=_USAGE)
    mw = _middleware(response)

    text, usage_update = await mw._acreate_summary_with_usage(
        [HumanMessage(content="a"), AIMessage(content="b")]
    )

    assert text == "the summary"
    assert usage_update == {
        "sum-1": {
            "source": "summarization",
            "input_tokens": 40,
            "output_tokens": 10,
            "total_tokens": 50,
            "cache_read": 5,
            "cache_creation": 0,
        }
    }


@pytest.mark.asyncio
async def test_acreate_summary_with_usage_handles_missing_metadata():
    response = AIMessage(content="summary", id="sum-2")  # no usage anywhere
    mw = _middleware(response)

    text, usage_update = await mw._acreate_summary_with_usage([HumanMessage(content="a")])

    assert text == "summary"
    assert usage_update is None


@pytest.mark.asyncio
async def test_acreate_summary_captures_usage_from_response_metadata_fallback():
    """Providers that only report counts on non-streamed calls via response_metadata."""
    response = AIMessage(
        content="summary", id="sum-3",
        response_metadata={"token_usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}},
    )
    mw = _middleware(response)

    text, usage_update = await mw._acreate_summary_with_usage([HumanMessage(content="a")])

    assert text == "summary"
    assert usage_update["sum-3"]["source"] == "summarization"
    assert usage_update["sum-3"]["total_tokens"] == 60


@pytest.mark.asyncio
async def test_abefore_model_records_summarization_usage_when_triggered():
    response = AIMessage(content="the summary", id="sum-1", usage_metadata=_USAGE)
    mw = _middleware(response)
    # Force summarization to run deterministically.
    mw._should_summarize = MagicMock(return_value=True)
    mw._determine_cutoff_index = MagicMock(return_value=1)

    state = {"messages": [HumanMessage(content="a", id="h1"), AIMessage(content="b", id="a1")]}
    result = await mw.abefore_model(state, MagicMock())

    assert result["token_usage"]["sum-1"]["source"] == "summarization"
    assert result["token_usage"]["sum-1"]["total_tokens"] == 50
    # Parent behaviour preserved: messages are replaced with a summary.
    assert isinstance(result["messages"][0], RemoveMessage)


@pytest.mark.asyncio
async def test_abefore_model_returns_none_when_not_triggered():
    response = AIMessage(content="unused", id="sum-1", usage_metadata=_USAGE)
    mw = _middleware(response)
    mw._should_summarize = MagicMock(return_value=False)

    state = {"messages": [HumanMessage(content="a", id="h1")]}
    result = await mw.abefore_model(state, MagicMock())

    assert result is None


@pytest.mark.asyncio
async def test_token_usage_accumulates_across_turns_in_real_graph():
    """Regression: create_agent registers middleware channels as LastValue, so
    token_usage must be merged with prior state each write. This asserts the full
    stack accumulates across turns (agent + summarization), never shrinking."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain.agents import create_agent
    from app.services.agent.middleware import MessagesHistoryMiddleware

    class UsageModel(GenericFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            r = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            r.generations[0].message.usage_metadata = {
                "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                "input_token_details": {"cache_read": 0, "cache_creation": 0},
            }
            return r

    model = UsageModel(messages=iter([AIMessage(content=f"r{i}", id=f"a{i}") for i in range(50)]))
    middleware = [
        MessagesHistoryMiddleware(),
        UsageTrackingSummarizationMiddleware(model=model, trigger=[("messages", 2)], keep=("messages", 1)),
    ]
    agent = create_agent(model, tools=[], middleware=middleware, checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "t1"}}
    totals = []
    sources_seen = set()
    for i in range(4):
        await agent.ainvoke({"messages": [HumanMessage(content=f"q{i}", id=f"h{i}")]}, config=cfg)
        state = await agent.aget_state(config=cfg)
        tu = state.values.get("token_usage", {})
        totals.append(sum(e["total_tokens"] for e in tu.values()))
        sources_seen.update(e["source"] for e in tu.values())

    # Monotonically non-decreasing — never shrinks after summarization.
    assert totals == sorted(totals)
    assert totals[-1] > totals[0]
    # Summarization was triggered and counted.
    assert "agent" in sources_seen
    assert "summarization" in sources_seen
