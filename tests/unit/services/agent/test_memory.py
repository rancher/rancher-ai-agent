import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.memory import MemoryManager

@pytest.mark.asyncio
async def test_fetch_chats_returns_non_empty_chats():
    mock_checkpointer = MagicMock()
    # Simulate two checkpoint tuples, one empty, one valid
    valid_tuple = MagicMock()
    valid_tuple.config = {"configurable": {"thread_id": "chat1"}}
    valid_tuple.metadata = {"user_id": "user1"}
    valid_tuple.__getitem__.side_effect = lambda k: valid_tuple.__dict__[k]
    # _is_empty_chat returns False for valid_tuple
    empty_tuple = MagicMock()
    empty_tuple.config = {"configurable": {"thread_id": "chat2"}}
    empty_tuple.metadata = {"user_id": "user1"}
    # _is_empty_chat returns True for empty_tuple
    async def alist_mock(*args, **kwargs):
        for item in [valid_tuple, empty_tuple]:
            yield item
    mock_checkpointer.alist = alist_mock
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    manager._is_empty_chat = MagicMock(side_effect=lambda tup: tup is empty_tuple)
    manager._get_chat_metadata = MagicMock(return_value={"name": "Test Chat", "created_at": "2024-01-01T00:00:00Z"})
    chats = await manager.fetch_chats("user1", {})
    assert len(chats) == 1
    assert chats[0]["id"] == "chat1"
    assert chats[0]["name"] == "Test Chat"
    assert chats[0]["createdAt"] == "2024-01-01T00:00:00Z"

@pytest.mark.asyncio
async def test_delete_chats_deletes_threads():
    mock_checkpointer = MagicMock()
    tuple1 = MagicMock()
    tuple1.config = {"configurable": {"thread_id": "chat1"}}
    tuple1.metadata = {"user_id": "user1"}
    tuple2 = MagicMock()
    tuple2.config = {"configurable": {"thread_id": "chat2"}}
    tuple2.metadata = {"user_id": "user1"}
    async def alist_mock(*args, **kwargs):
        for item in [tuple1, tuple2]:
            yield item
    mock_checkpointer.alist = alist_mock
    mock_checkpointer.adelete_thread = AsyncMock()
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    await manager.delete_chats("user1")
    mock_checkpointer.adelete_thread.assert_any_call("chat1")
    mock_checkpointer.adelete_thread.assert_any_call("chat2")
    assert mock_checkpointer.adelete_thread.await_count == 2

@pytest.mark.asyncio
async def test_fetch_chat_returns_chat_if_exists():
    mock_checkpointer = MagicMock()
    tuple1 = MagicMock()
    tuple1.config = {"configurable": {"thread_id": "chat1"}}
    tuple1.metadata = {"user_id": "user1"}
    mock_checkpointer.aget_tuple = AsyncMock(return_value=tuple1)
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    manager._is_empty_chat = MagicMock(return_value=False)
    manager._get_chat_metadata = MagicMock(return_value={"name": "Test Chat", "created_at": "2024-01-01T00:00:00Z"})
    chat = await manager.fetch_chat("chat1", "user1")
    assert chat["id"] == "chat1"
    assert chat["userId"] == "user1"
    assert chat["name"] == "Test Chat"
    assert chat["createdAt"] == "2024-01-01T00:00:00Z"

@pytest.mark.asyncio
async def test_fetch_chat_returns_none_for_another_users_chat():
    mock_checkpointer = MagicMock()
    tuple1 = MagicMock()
    tuple1.metadata = {"user_id": "owner"}
    mock_checkpointer.aget_tuple = AsyncMock(return_value=tuple1)
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    manager._is_empty_chat = MagicMock(return_value=False)
    assert await manager.fetch_chat("chat1", "intruder") is None


@pytest.mark.asyncio
async def test_fetch_chat_returns_none_if_not_found():
    mock_checkpointer = MagicMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    chat = await manager.fetch_chat("chat1", "user1")
    assert chat is None

@pytest.mark.asyncio
async def test_delete_chat_deletes_thread_if_exists():
    mock_checkpointer = MagicMock()
    tuple1 = MagicMock()
    tuple1.metadata = {"user_id": "user1"}
    mock_checkpointer.aget_tuple = AsyncMock(return_value=tuple1)
    mock_checkpointer.adelete_thread = AsyncMock()

    async def alist_mock(*args, **kwargs):
        return
        yield

    mock_checkpointer.alist = alist_mock
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    await manager.delete_chat("chat1", "user1")
    mock_checkpointer.adelete_thread.assert_awaited_once_with("chat1")

@pytest.mark.asyncio
async def test_delete_chat_deletes_child_and_planner_threads():
    """Child/planner threads hold part of the chat's usage, so they go with the chat."""
    owner_tuple = MagicMock()
    owner_tuple.metadata = {"user_id": "user1"}

    def _thread(thread_id):
        tup = MagicMock()
        tup.config = {"configurable": {"thread_id": thread_id}}
        return tup

    seen_filters = []

    async def alist_mock(*args, **kwargs):
        seen_filters.append(kwargs.get("filter"))
        for thread_id in ["chat1", "chat1::child::kubernetes", "chat1::planner::fleet", "chat2", "chat2::child::k8s"]:
            yield _thread(thread_id)

    mock_checkpointer = MagicMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=owner_tuple)
    mock_checkpointer.adelete_thread = AsyncMock()
    mock_checkpointer.alist = alist_mock
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer

    await manager.delete_chat("chat1", "user1")

    assert seen_filters == [{"user_id": "user1"}]
    deleted = {c.args[0] for c in mock_checkpointer.adelete_thread.await_args_list}
    assert deleted == {"chat1", "chat1::child::kubernetes", "chat1::planner::fleet"}

@pytest.mark.asyncio
async def test_delete_chat_ignores_another_users_chat():
    owner_tuple = MagicMock()
    owner_tuple.metadata = {"user_id": "owner"}
    mock_checkpointer = MagicMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=owner_tuple)
    mock_checkpointer.adelete_thread = AsyncMock()
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    await manager.delete_chat("chat1", "intruder")
    mock_checkpointer.adelete_thread.assert_not_awaited()

@pytest.mark.asyncio
async def test_delete_chat_does_nothing_if_not_found():
    mock_checkpointer = MagicMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.adelete_thread = AsyncMock()
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    await manager.delete_chat("chat1", "user1")
    mock_checkpointer.adelete_thread.assert_not_awaited()

def _usage_checkpoint(thread_id, token_usage, checkpoint_ns=""):
    tup = MagicMock()
    tup.config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
    tup.metadata = {"user_id": "user1"}
    tup.checkpoint = {"channel_values": {"token_usage": token_usage}}
    return tup


def _entry(total, source="agent", provider="bedrock", model="claude", **extra):
    return {
        "source": source, "provider": provider, "model": model,
        "input_tokens": total, "output_tokens": 0, "total_tokens": total,
        "cache_read": 0, "cache_creation": 0, **extra,
    }


def _usage_manager(*checkpoints):
    async def alist_mock(*args, **kwargs):
        for item in checkpoints:
            yield item

    mock_checkpointer = MagicMock()
    mock_checkpointer.alist = alist_mock
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    return manager


@pytest.mark.asyncio
async def test_fetch_chat_usage_sums_main_and_child_threads():
    """Usage is summed across the chat's own thread and its child/planner threads."""
    main = _usage_checkpoint("chat1", {
        "m1": {**_entry(120), "input_tokens": 100, "output_tokens": 20, "cache_read": 10, "cache_creation": 5},
        "ui1": _entry(10, source="ui-tools"),
        "sum1": _entry(50, source="summarization"),
        # Child agent running on the chat's own thread (direct agent request).
        "d1": _entry(7, agent="rancher"),
    })
    child = _usage_checkpoint("chat1::child::kubernetes", {"c1": {**_entry(230), "cache_read": 15}})
    planner = _usage_checkpoint("chat1::planner::fleet", {"p1": _entry(40, provider="openai", model="gpt-4o")})
    other = _usage_checkpoint("chat2", {"x1": _entry(999)})

    usage = await _usage_manager(main, child, planner, other).fetch_chat_usage("chat1", "user1")

    assert usage["chatId"] == "chat1"
    assert usage["totalTokens"] == 120 + 10 + 50 + 7 + 230 + 40
    assert usage["inputTokens"] == 100 + 10 + 50 + 7 + 230 + 40
    assert usage["outputTokens"] == 20
    assert usage["cacheReadTokens"] == 10 + 15
    assert usage["cacheCreationTokens"] == 5
    assert usage["llmCalls"] == 6
    assert "cost" not in usage

    by_agent = {k: v["totalTokens"] for k, v in usage["byAgent"].items()}
    assert by_agent == {"supervisor": 180, "rancher": 7, "kubernetes": 230, "fleet": 40}

    by_source = {k: v["totalTokens"] for k, v in usage["bySource"].items()}
    assert by_source == {"agent": 397, "ui-tools": 10, "summarization": 50}

    assert set(usage["byModel"]) == {"bedrock", "openai"}
    assert usage["byModel"]["bedrock"]["claude"]["totalTokens"] == 417
    assert usage["byModel"]["openai"]["gpt-4o"]["totalTokens"] == 40


@pytest.mark.asyncio
async def test_fetch_chat_usage_groups_legacy_entries_under_unknown():
    """Entries recorded before provider/model were stored fall into 'unknown'."""
    legacy = {"source": "agent", "input_tokens": 3, "output_tokens": 0, "total_tokens": 3}
    usage = await _usage_manager(_usage_checkpoint("chat1", {"m1": legacy})).fetch_chat_usage("chat1", "user1")

    assert usage["byModel"] == {"unknown": {"unknown": usage["byAgent"]["supervisor"]}}


@pytest.mark.asyncio
async def test_fetch_chat_usage_reads_latest_checkpoint_per_thread():
    """alist yields multiple checkpoints per thread; only the first (latest) counts."""
    latest = _usage_checkpoint("chat1", {"m1": _entry(10), "m0": _entry(5)})
    older = _usage_checkpoint("chat1", {"m0": _entry(5)})

    usage = await _usage_manager(latest, older).fetch_chat_usage("chat1", "user1")

    assert usage["totalTokens"] == 15
    assert usage["llmCalls"] == 2


@pytest.mark.asyncio
async def test_fetch_chat_usage_empty():
    usage = await _usage_manager().fetch_chat_usage("chat1", "user1")

    assert usage["totalTokens"] == 0
    assert usage["llmCalls"] == 0
    assert usage["byAgent"] == {} and usage["bySource"] == {} and usage["byModel"] == {}


@pytest.mark.asyncio
async def test_record_chat_usage_is_aggregated_with_the_chat():
    """Usage of out-of-graph calls is stored in usage threads that belong to the chat."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"DB_ENABLED": "false"}):
        manager = MemoryManager()
        await manager.initialize()

    await manager.record_chat_usage("chat1", "user1", {"s1": _entry(12, source="ui-tools")})
    await manager.record_chat_usage("chat1", "user1", {"s2": _entry(8, source="ui-tools")})
    await manager.record_chat_usage("chat1", "user1", {"s1": _entry(12, source="ui-tools")})  # same key
    await manager.record_chat_usage("chat2", "other", {"s3": _entry(99, source="ui-tools")})

    usage = await manager.fetch_chat_usage("chat1", "user1")
    assert usage["totalTokens"] == 20
    assert usage["llmCalls"] == 2
    assert set(usage["bySource"]) == {"ui-tools"}
    assert set(usage["byAgent"]) == {"supervisor"}

    # Usage threads are not chats, and are deleted with their chat.
    assert await manager.fetch_chats("user1") == []
    manager._is_empty_chat = MagicMock(return_value=False)
    manager.get_checkpointer().aget_tuple = AsyncMock(return_value=MagicMock(metadata={"user_id": "user1"}))
    await manager.delete_chat("chat1", "user1")
    assert (await manager.fetch_chat_usage("chat1", "user1"))["llmCalls"] == 0


@pytest.mark.asyncio
async def test_fetch_messages_payload():
    """Verify that fetch_messages includes UI tools in the agent response payload."""
    from langchain_core.messages import HumanMessage, AIMessage
    
    mock_checkpointer = MagicMock()
    checkpoint_tuple = MagicMock()
    
    # Create mock messages
    human_msg = MagicMock(spec=HumanMessage)
    human_msg.type = "human"
    human_msg.content = "Show me the dashboard"
    human_msg.additional_kwargs = {
        "request_id": "req1",
        "request_metadata": {
            "user_input": "Show me the dashboard",
            "agent": {"name": "test-agent"},
            "tags": ["dashboard"],
            "labels": {},
            "context": None,
        },
        "created_at": "2024-01-01T00:00:00Z",
    }
    
    ai_msg = MagicMock(spec=AIMessage)
    ai_msg.type = "ai"
    ai_msg.content = "Here is the dashboard"
    ai_msg.text = "Here is the dashboard"
    ai_msg.additional_kwargs = {
        "request_id": "req1",
        "selected_agent": {"name": "test-agent", "mode": "auto"},
        "ui_tools": [
            {
                "toolName": "chart_tool",
                "description": "Renders data charts",
                "prompt": "Show chart",
            },
            {
                "toolName": "table_tool",
                "description": "Renders data in table format",
                "prompt": "Show table",
            },
        ],
        "created_at": "2024-01-01T00:00:01Z",
    }
    
    # Setup checkpoint tuple
    checkpoint_tuple.checkpoint = {
        "channel_values": {
            "messages_history": [human_msg, ai_msg],
        }
    }
    mock_checkpointer.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    
    messages = await manager.fetch_messages("chat1", "user1", {})
    
    # Verify messages structure
    assert len(messages) == 2
    
    # Verify user message
    user_msg = messages[0]
    assert user_msg["role"] == "user"
    assert user_msg["message"] == "Show me the dashboard"
    assert user_msg["tags"] == ["dashboard"]
    
    # Verify agent message includes tools
    agent_msg = messages[1]
    assert agent_msg["role"] == "agent"
    assert agent_msg["message"] == "Here is the dashboard"
    assert "tools" in agent_msg
    assert len(agent_msg["tools"]) == 2
    assert agent_msg["tools"][0]["toolName"] == "chart_tool"
    assert agent_msg["tools"][1]["toolName"] == "table_tool"

@pytest.mark.asyncio
async def test_message_types_have_correct_fields():
    """Verify that different message types have the correct fields in fetch_messages.
    
    This test ensures:
    - ALL messages include chatId, role, agent, message, createdAt
    - USER messages: include also context, labels, tags from request metadata
    - AI messages: include also tools, tags, but NOT context or labels
    - TOOL messages: include also tools and confirmation, but NOT context or labels or tags
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    mock_checkpointer = MagicMock()
    checkpoint_tuple = MagicMock()
    chat_id = "chat-123"
    
    # Create user message with labels, context, and tags
    human_msg = MagicMock(spec=HumanMessage)
    human_msg.type = "human"
    human_msg.content = "Show pod logs from namespace-1"
    human_msg.additional_kwargs = {
        "request_id": "req1",
        "request_metadata": {
            "user_input": "Show pod logs from namespace-1",
            "agent": "kubernetes-agent",
            "tags": ["kubernetes", "logs"],
            "labels": {"summary": "Log viewer"},
            "context": {"namespace": "namespace-1"},
        },
        "created_at": "2024-01-01T00:00:00Z",
    }
    
    # Create AI message - should have tools and tags, but NOT context/labels
    ai_msg = MagicMock(spec=AIMessage)
    ai_msg.type = "ai"
    ai_msg.content = "Here are the pod logs"
    ai_msg.text = "Here are the pod logs"
    ai_msg.additional_kwargs = {
        "request_id": "req1",
        "created_at": "2024-01-01T00:00:01Z",
        "agent": "kubernetes-agent",
        "ui_tools": [
            {"toolName": "log_viewer", "description": "View logs"}
        ],
        "request_metadata": {
            "agent": "kubernetes-agent",
            "tags": ["kubernetes", "logs"],
            # Note: labels and context are NOT included to prevent inheriting user input metadata
        },
    }
    
    # Create tool message with interrupt - should have tools and confirmation, but NOT context/labels/tags
    tool_msg = MagicMock(spec=ToolMessage)
    tool_msg.type = "tool"
    tool_msg.name = "confirm_action"
    tool_msg.content = "User confirmed"
    tool_msg.additional_kwargs = {
        "interrupt_message": "Confirm deletion of pod?",
        "selected_agent": "kubernetes-agent",
        "confirmation": True,
        "ui_tools": [{"toolName": "action_tool"}],
    }
    
    # Setup checkpoint tuple
    checkpoint_tuple.checkpoint = {
        "channel_values": {
            "messages_history": [human_msg, ai_msg, tool_msg],
        }
    }
    mock_checkpointer.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    
    messages = await manager.fetch_messages(chat_id, "user1", {})
    
    assert len(messages) == 3
    
    # ===== VERIFY USER MESSAGE =====
    user_msg = messages[0]
    # Common fields (ALL messages)
    assert user_msg["chatId"] == chat_id, "User message should have chatId"
    assert user_msg["role"] == "user", "User message role should be 'user'"
    assert user_msg["agent"] == "kubernetes-agent", "User message should have agent"
    assert user_msg["message"] == "Show pod logs from namespace-1", "User message should contain content"
    assert "createdAt" in user_msg, "User message should have createdAt"
    # USER-specific fields
    assert user_msg["labels"] == {"summary": "Log viewer"}, "User message should have labels"
    assert user_msg["context"] == {"namespace": "namespace-1"}, "User message should have context"
    assert user_msg["tags"] == ["kubernetes", "logs"], "User message should have tags"
    
    # ===== VERIFY AI MESSAGE =====
    agent_msg = messages[1]
    # Common fields (ALL messages)
    assert agent_msg["chatId"] == chat_id, "AI message should have chatId"
    assert agent_msg["role"] == "agent", "AI message role should be 'agent'"
    assert agent_msg["message"] == "Here are the pod logs", "AI message should contain content"
    assert "createdAt" in agent_msg, "AI message should have createdAt"
    # AI-specific fields (from request_metadata, but WITHOUT labels/context)
    assert agent_msg["agent"] == "kubernetes-agent", "AI message should have agent"
    assert agent_msg["tags"] == ["kubernetes", "logs"], "AI message should have tags"
    assert "tools" in agent_msg, "AI message should have tools"
    assert len(agent_msg["tools"]) == 1, "AI message should have 1 tool"
    assert agent_msg["tools"][0]["toolName"] == "log_viewer", "AI message should preserve tool info"
    assert agent_msg.get("stopped", "") == "", "AI message should not be flagged as stopped"
    # Fields AI should NOT have
    assert "context" not in agent_msg, "AI message should NOT have context from user request"
    assert "labels" not in agent_msg, "AI message should NOT have labels from user request"
    
    # ===== VERIFY TOOL MESSAGE =====
    interrupt_msg = messages[2]
    # Common fields (ALL messages)
    assert interrupt_msg["chatId"] == chat_id, "Tool message should have chatId"
    assert interrupt_msg["role"] == "agent", "Tool message role should be 'agent'"
    assert interrupt_msg["agent"] == "kubernetes-agent", "Tool message should have agent"
    assert interrupt_msg["message"] == "Confirm deletion of pod?", "Tool message should contain interrupt_message"
    assert "createdAt" in interrupt_msg, "Tool message should have createdAt"
    # TOOL-specific fields
    assert "tools" in interrupt_msg, "Tool message should have tools"
    assert len(interrupt_msg["tools"]) == 1, "Tool message should have 1 tool"
    assert interrupt_msg["tools"][0]["toolName"] == "action_tool", "Tool message should preserve tool info"
    assert interrupt_msg["confirmation"] == True, "Tool message should have confirmation flag"
    # Fields TOOL should NOT have
    assert "context" not in interrupt_msg, "Tool message should NOT have context"
    assert "labels" not in interrupt_msg, "Tool message should NOT have labels"
    assert "tags" not in interrupt_msg, "Tool message should NOT have tags"

@pytest.mark.asyncio
async def test_fetch_messages_includes_stopped_ai_message():
    """A stopped AIMessage is returned flagged as stopped, even with empty content."""
    from langchain_core.messages import AIMessage

    mock_checkpointer = MagicMock()
    checkpoint_tuple = MagicMock()

    stopped_msg = MagicMock(spec=AIMessage)
    stopped_msg.type = "ai"
    stopped_msg.content = ""
    stopped_msg.text = ""
    stopped_msg.additional_kwargs = {
        "stopped": "message-manual-stop",
        "created_at": "2024-01-01T00:00:01Z",
        "request_metadata": {"agent": "kubernetes-agent", "tags": []},
    }

    checkpoint_tuple.checkpoint = {
        "channel_values": {
            "messages_history": [stopped_msg],
        }
    }
    mock_checkpointer.aget_tuple = AsyncMock(return_value=checkpoint_tuple)

    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer

    messages = await manager.fetch_messages("chat1", "user1", {})

    assert len(messages) == 1
    assert messages[0]["role"] == "agent"
    assert messages[0]["message"] == ""
    assert messages[0]["stopped"] == "message-manual-stop"


@pytest.mark.asyncio
async def test_fetch_messages_returns_partial_and_stopped_marker():
    """When an AI message is stopped with partial content, return both the partial message and an empty stopped marker.
    
    This proves that stopped responses always create two message entries:
    1. The partial/truncated content with the stop reason
    2. An empty message marking the response as complete/stopped
    """
    from langchain_core.messages import AIMessage, HumanMessage

    mock_checkpointer = MagicMock()
    checkpoint_tuple = MagicMock()

    # User message
    user_msg = MagicMock(spec=HumanMessage)
    user_msg.type = "human"
    user_msg.content = "Stop this response"
    user_msg.additional_kwargs = {
        "request_id": "req1",
        "request_metadata": {"agent": "test-agent", "tags": []},
        "created_at": "2024-01-01T00:00:00Z",
    }

    # AI message with partial content AND stopped flag
    stopped_msg = MagicMock(spec=AIMessage)
    stopped_msg.type = "ai"
    stopped_msg.content = "This is a partial response that was "  # Incomplete!
    stopped_msg.text = "This is a partial response that was "
    stopped_msg.additional_kwargs = {
        "stopped": "message-manual-stop",
        "created_at": "2024-01-01T00:00:01Z",
        "request_metadata": {"agent": "test-agent", "tags": []},
    }

    checkpoint_tuple.checkpoint = {
        "channel_values": {
            "messages_history": [user_msg, stopped_msg],
        }
    }
    mock_checkpointer.aget_tuple = AsyncMock(return_value=checkpoint_tuple)

    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer

    messages = await manager.fetch_messages("chat1", "user1", {})

    # Should return 3 messages: user + partial response + empty stopped marker
    assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}: {messages}"
    
    # Message 1: User message
    assert messages[0]["role"] == "user"
    assert messages[0]["message"] == "Stop this response"
    
    # Message 2: Partial AI response (with content, no stopped flag)
    assert messages[1]["role"] == "agent"
    assert messages[1]["message"] == "This is a partial response that was "
    assert messages[1].get("stopped", "") == "", "Partial message should NOT have stopped flag"
    
    # Message 3: Empty stopped marker (proves response was stopped)
    assert messages[2]["role"] == "agent"
    assert messages[2]["message"] == "", "Stopped marker should be empty"
    assert messages[2]["stopped"] == "message-manual-stop", "Stopped marker must have stop reason"


@pytest.mark.asyncio
async def test_fetch_messages_skips_ai_message_without_text():
    """An AIMessage with only non-text content blocks (e.g. a tool_use-only supervisor call) yields no row."""
    from langchain_core.messages import AIMessage

    mock_checkpointer = MagicMock()
    checkpoint_tuple = MagicMock()

    tool_call_msg = AIMessage(
        content=[{"type": "tool_use", "id": "call_1", "name": "fleet", "input": {"query": "create pod"}}],
        tool_calls=[{"id": "call_1", "name": "fleet", "args": {"query": "create pod"}}],
        additional_kwargs={"created_at": "2024-01-01T00:00:01Z", "request_metadata": {"agent": "", "tags": []}},
    )

    checkpoint_tuple.checkpoint = {
        "channel_values": {
            "messages_history": [tool_call_msg],
        }
    }
    mock_checkpointer.aget_tuple = AsyncMock(return_value=checkpoint_tuple)

    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer

    messages = await manager.fetch_messages("chat1", "user1", {})

    assert messages == []
