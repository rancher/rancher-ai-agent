import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import status, HTTPException
from app.routers import llm as llm_router

@pytest.fixture
def mock_request():
    req = MagicMock()
    req.app.memory_manager = AsyncMock()
    req.cookies = {"R_SESS": "token"}
    req.headers = {"Host": "localhost"}
    return req


@pytest.mark.asyncio
async def test_complete_ui_tools_success(mock_request):
    mock_selector = MagicMock()
    mock_selector.select_tools = AsyncMock(return_value=[{"toolName": "show"}])
    
    with patch("app.routers.llm.get_user_id_from_request", AsyncMock(return_value="u")):
        ui_tools_config = llm_router.UIToolsConfig(name="tools", tools=["show"])
        ui_tools_request = llm_router.UIToolsRequest(
            prompt="test",
            tools=ui_tools_config,
            context=[]
        )
        with patch("app.routers.llm.create_ui_tools_selector", return_value=mock_selector):
            resp = await llm_router.complete_ui_tools(mock_request, ui_tools_request, MagicMock())
            assert resp.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_complete_ui_tools_missing_name(mock_request):    
    with patch("app.routers.llm.get_user_id_from_request", AsyncMock(return_value="u")):
        ui_tools_config = llm_router.UIToolsConfig(name="", tools=None)
        ui_tools_request = llm_router.UIToolsRequest(
            prompt="test",
            tools=ui_tools_config,
            context=[]
        )
        with pytest.raises(HTTPException) as exc:
            await llm_router.complete_ui_tools(mock_request, ui_tools_request, MagicMock())
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST

