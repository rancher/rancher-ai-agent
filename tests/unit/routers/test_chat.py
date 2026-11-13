import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import status, HTTPException
from app.routers import chat as chat_router

@pytest.fixture
def mock_request():
    req = MagicMock()
    req.app.memory_manager = AsyncMock()
    req.cookies = {"R_SESS": "token"}
    req.headers = {"Host": "localhost"}
    return req

@pytest.mark.asyncio
async def test_get_chats_success(mock_request):
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {"id": "1", "userId": "u", "createdAt": 2},
        {"id": "2", "userId": "u", "createdAt": 1},
    ]
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.get_chats(mock_request)
        assert resp.status_code == status.HTTP_200_OK
        assert resp.body

@pytest.mark.asyncio
async def test_get_chats_sort_asc(mock_request):
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {"id": "1", "userId": "u", "createdAt": 2},
        {"id": "2", "userId": "u", "createdAt": 1},
    ]
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.get_chats(mock_request, sort="createdAt:asc")
        result = json.loads(resp.body)
        assert result[0]["createdAt"] == 1
        assert result[1]["createdAt"] == 2

@pytest.mark.asyncio
async def test_get_chat_not_found(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.get_chat(mock_request, chat_id="notfound")
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND

@pytest.mark.asyncio
async def test_update_chat_success(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.update_chat.return_value = {"id": "1", "userId": "u", "name": "updated"}
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.update_chat(mock_request, chat_id="1", chat_data={"name": "updated"})
        assert resp.status_code == status.HTTP_200_OK
        assert b"updated" in resp.body


@pytest.mark.asyncio
async def test_update_chat_not_found(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.update_chat(mock_request, chat_id="notfound", chat_data={"name": "updated"})
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_update_chat_update_failure(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.update_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.update_chat(mock_request, chat_id="1", chat_data={"name": "fail"})
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_update_chat_invalid_data(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(Exception):
            await chat_router.update_chat(mock_request, chat_id="1", chat_data=None)

@pytest.mark.asyncio
async def test_delete_chat_success(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.delete_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.delete_chat(mock_request, chat_id="1")
        assert resp.status_code == status.HTTP_204_NO_CONTENT

@pytest.mark.asyncio
async def test_get_chat_messages_not_found(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.get_chat_messages(mock_request, chat_id="notfound")
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_get_chat_usage_success(mock_request):
    usage = {"chatId": "1", "totalTokens": 42, "byAgent": {}, "bySource": {}, "byModel": {}}
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.fetch_chat_usage.return_value = usage
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.get_chat_usage(mock_request, chat_id="1")
    assert resp.status_code == status.HTTP_200_OK
    assert json.loads(resp.body) == usage
    mock_request.app.memory_manager.fetch_chat_usage.assert_awaited_once_with(chat_id="1", user_id="u")


@pytest.mark.asyncio
async def test_get_chat_usage_unauthorized(mock_request):
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await chat_router.get_chat_usage(mock_request, chat_id="1")
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_get_chat_usage_not_found(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = None
    mock_request.app.memory_manager.fetch_chat_usage.return_value = {"chatId": "notfound", "llmCalls": 0}
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.get_chat_usage(mock_request, chat_id="notfound")
    assert exc.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_get_chat_usage_of_chat_without_messages_but_with_usage(mock_request):
    """e.g. /complete/ui-tools called before the first message of the chat."""
    usage = {"chatId": "new", "llmCalls": 1, "totalTokens": 42}
    mock_request.app.memory_manager.fetch_chat.return_value = None
    mock_request.app.memory_manager.fetch_chat_usage.return_value = usage
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.get_chat_usage(mock_request, chat_id="new")
    assert resp.status_code == status.HTTP_200_OK
    assert json.loads(resp.body) == usage


@pytest.mark.asyncio
async def test_get_chat_usage_internal_error(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.fetch_chat_usage.side_effect = RuntimeError("boom")
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.get_chat_usage(mock_request, chat_id="1")
    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
