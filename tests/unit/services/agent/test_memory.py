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
    manager._aggregate_token_usage = AsyncMock(return_value={"totalTokens": 42, "byAgent": {}})
    chat = await manager.fetch_chat("chat1", "user1")
    assert chat["id"] == "chat1"
    assert chat["userId"] == "user1"
    assert chat["name"] == "Test Chat"
    assert chat["createdAt"] == "2024-01-01T00:00:00Z"
    assert chat["usage"] == {"totalTokens": 42, "byAgent": {}}

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
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    await manager.delete_chat("chat1", "user1")
    mock_checkpointer.adelete_thread.assert_awaited_once_with("chat1")

@pytest.mark.asyncio
async def test_delete_chat_does_nothing_if_not_found():
    mock_checkpointer = MagicMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.adelete_thread = AsyncMock()
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer
    await manager.delete_chat("chat1", "user1")
    mock_checkpointer.adelete_thread.assert_not_awaited()

def _usage_checkpoint(thread_id, token_usage):
    tup = MagicMock()
    tup.config = {"configurable": {"thread_id": thread_id}}
    tup.checkpoint = {"channel_values": {"token_usage": token_usage}}
    return tup


@pytest.mark.asyncio
async def test_aggregate_token_usage_buckets_per_source():
    """Usage is summed across main + child threads into per-source buckets."""
    main = _usage_checkpoint("chat1", {
        "m1": {"source": "agent", "input_tokens": 100, "output_tokens": 20,
               "total_tokens": 120, "cache_read": 10, "cache_creation": 5},
        "ui1": {"source": "ui-tools", "input_tokens": 8, "output_tokens": 2,
                "total_tokens": 10, "cache_read": 0, "cache_creation": 0},
        "sum1": {"source": "summarization", "input_tokens": 40, "output_tokens": 10,
                 "total_tokens": 50, "cache_read": 0, "cache_creation": 0},
    })
    child = _usage_checkpoint("chat1::child::kubernetes", {
        "c1": {"source": "agent", "input_tokens": 200, "output_tokens": 30,
               "total_tokens": 230, "cache_read": 15, "cache_creation": 0},
    })
    # A different conversation's thread that must be ignored.
    other = _usage_checkpoint("chat2", {
        "x1": {"source": "agent", "input_tokens": 999, "output_tokens": 999,
               "total_tokens": 999, "cache_read": 999, "cache_creation": 999},
    })

    async def alist_mock(*args, **kwargs):
        for item in [main, child, other]:
            yield item

    mock_checkpointer = MagicMock()
    mock_checkpointer.alist = alist_mock
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer

    usage = await manager._aggregate_token_usage("chat1", "user1")

    assert usage["totalTokens"] == 120 + 10 + 50 + 230
    assert usage["inputTokens"] == 100 + 8 + 40 + 200
    assert usage["outputTokens"] == 20 + 2 + 10 + 30
    assert usage["cacheReadTokens"] == 10 + 0 + 0 + 15
    assert usage["cacheCreationTokens"] == 5

    by_agent = usage["byAgent"]
    assert set(by_agent) == {"supervisor", "ui-tools", "summarization", "kubernetes"}
    assert by_agent["supervisor"]["totalTokens"] == 120
    assert by_agent["ui-tools"]["totalTokens"] == 10
    assert by_agent["summarization"]["totalTokens"] == 50
    assert by_agent["kubernetes"]["totalTokens"] == 230
    assert by_agent["kubernetes"]["cacheReadTokens"] == 15


@pytest.mark.asyncio
async def test_aggregate_token_usage_dedups_to_latest_checkpoint_per_thread():
    """alist yields multiple checkpoints per thread; only the first (latest) counts."""
    latest = _usage_checkpoint("chat1", {
        "m1": {"source": "agent", "input_tokens": 10, "output_tokens": 0,
               "total_tokens": 10, "cache_read": 0, "cache_creation": 0},
    })
    older = _usage_checkpoint("chat1", {
        "m0": {"source": "agent", "input_tokens": 5, "output_tokens": 0,
               "total_tokens": 5, "cache_read": 0, "cache_creation": 0},
    })

    async def alist_mock(*args, **kwargs):
        for item in [latest, older]:  # newest first
            yield item

    mock_checkpointer = MagicMock()
    mock_checkpointer.alist = alist_mock
    manager = MemoryManager()
    manager.checkpointer = mock_checkpointer

    usage = await manager._aggregate_token_usage("chat1", "user1")

    assert usage["totalTokens"] == 10
    assert usage["byAgent"]["supervisor"]["totalTokens"] == 10


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
