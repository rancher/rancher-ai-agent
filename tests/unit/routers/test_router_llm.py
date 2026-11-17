import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, status, HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from app.routers import llm as llm_router

@pytest.fixture
def mock_request():
    req = MagicMock()
    req.app.memory_manager = AsyncMock()
    req.cookies = {"R_SESS": "token"}
    req.headers = {"Host": "localhost"}
    return req


def _request(name="tools", chat_id="chat-1"):
    return llm_router.UIToolsRequest(
        prompt="test",
        tools=llm_router.UIToolsConfig(name=name, tools=["show"]),
        context=[],
        chatId=chat_id,
    )


@pytest.mark.asyncio
async def test_complete_ui_tools_success(mock_request):
    mock_selector = MagicMock()
    mock_selector.select_tools = AsyncMock(return_value=[{"toolName": "show"}])
    mock_selector.last_response = None
    
    with patch("app.routers.llm.get_user_id_from_token", AsyncMock(return_value="u")):
        with patch("app.routers.llm.create_ui_tools_selector", return_value=mock_selector):
            resp = await llm_router.complete_ui_tools(mock_request, _request(), MagicMock())
            assert resp.status_code == status.HTTP_200_OK
    mock_request.app.memory_manager.record_chat_usage.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_ui_tools_records_usage_on_chat(mock_request):
    mock_selector = MagicMock()
    mock_selector.select_tools = AsyncMock(return_value=[])
    mock_selector.last_response = AIMessage(
        content="", id="sel-1", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    )

    with patch("app.routers.llm.get_user_id_from_token", AsyncMock(return_value="u")), \
         patch("app.routers.llm.create_ui_tools_selector", return_value=mock_selector):
        resp = await llm_router.complete_ui_tools(mock_request, _request(), MagicMock())

    assert resp.status_code == status.HTTP_200_OK
    kwargs = mock_request.app.memory_manager.record_chat_usage.await_args.kwargs
    assert kwargs["chat_id"] == "chat-1"
    assert kwargs["user_id"] == "u"
    assert kwargs["usage"]["sel-1"]["source"] == "ui-tools"
    assert kwargs["usage"]["sel-1"]["total_tokens"] == 12


@pytest.mark.parametrize("body_chat", [{}, {"chatId": None}, {"chatId": ""}])
def test_complete_ui_tools_requires_chat_id(body_chat):
    app = FastAPI()
    app.include_router(llm_router.router)
    app.dependency_overrides[llm_router.get_llm] = lambda: MagicMock()
    body = {"prompt": "p", "tools": {"name": "tools"}, "context": [], **body_chat}
    with patch("app.routers.llm.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = TestClient(app).post("/v1/api/complete/ui-tools", json=body)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_complete_ui_tools_missing_name(mock_request):    
    with patch("app.routers.llm.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await llm_router.complete_ui_tools(mock_request, _request(name=""), MagicMock())
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST

