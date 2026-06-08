import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.language_models.llms import BaseLanguageModel

from ..services.auth import get_user_id_from_request
from ..dependencies import get_llm
from ..services.ui_tools.selector import create_ui_tools_selector

router = APIRouter(prefix="/v1/api/complete", tags=["llm"])

class LLMRequest(BaseModel):
    """Request model for simple LLM requests."""
    prompt: str


class UIToolsConfig(BaseModel):
    """Configuration for UI tools selection."""
    name: str
    tools: list | None = None


class UIToolsRequest(LLMRequest):
    """Request model for UI tools selection."""
    tools: UIToolsConfig
    context: list


@router.post("/ui-tools")
async def complete_ui_tools(
    request: Request,
    ui_tools_request: UIToolsRequest,
    llm: BaseLanguageModel = Depends(get_llm)
) -> JSONResponse:
    """
    Select appropriate UI tools based on the provided context using the LLM.

    Args:
        ui_tools_request: The request containing the prompt and tools configuration.
        llm: The language model instance (injected via dependency).

    Returns:
        A list of selected UI tools and an HTTP status code.
    """

    try:
        user_id = await get_user_id_from_request(request)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not ui_tools_request.tools or not ui_tools_request.tools.name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="tools.name is required")

        ui_tools_list = []

        # Try to select UI tools for the provided prompt and context
        ui_tools_config = ui_tools_request.tools.model_dump()

        selector = create_ui_tools_selector(
            llm=llm,
            ui_tools_config=ui_tools_config,
            system_prompt=ui_tools_request.prompt
        )

        if selector:
            ui_tools_list = await selector.select_tools(
                context=ui_tools_request.context,
            )
            logging.debug(f"Selected {len(ui_tools_list)} UI tool(s) for context: {ui_tools_request.context}")
        else:
            logging.debug(f"No UI tools available for tools config: {ui_tools_config}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ui_tools_list
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error selecting UI tools: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

