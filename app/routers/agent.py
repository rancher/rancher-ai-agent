import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/v1/api", tags=["agent"])

@router.get("/health")
async def health():
    """
    Liveness probe endpoint to verify the HTTP service is running.
    Returns 200 OK if the service is responding.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "healthy"}
    )

@router.get("/readiness")
async def readiness(request: Request):
    """
    Readiness probe endpoint to verify the agent is ready.
    Checks:
    - Memory manager is initialized
    - FastAPI startup is complete
    """
    try:        
        # Check memory manager is initialized
        if not hasattr(request.app, 'memory_manager'):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Memory manager not initialized"}
            )

        # Check if startup is complete
        if not getattr(request.app.state, 'ready', False):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Application startup not complete"}
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"detail": "Agent is ready"}
        )

    except Exception as e:
        logging.error(f"Readiness check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e)}
        )
