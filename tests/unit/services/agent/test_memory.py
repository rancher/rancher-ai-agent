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

@pytest.mark.asyncio
async def test_fetch_messages_payload():
    """Verify that fetch_messages includes UI tools in the agent response payload."""
    from langchain_core.messages import HumanMessage, AIMessage
    
    mock_checkpointer = MagicMock()
    checkpoint_tuple = MagicMock()
    
    # Create mock messages
    human_msg = MagicMock(spec=HumanMessage)
    human_msg.type = "human"
    human_msg.text = ""
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
            "messages": [human_msg, ai_msg],
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
