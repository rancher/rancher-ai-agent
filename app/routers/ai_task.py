import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from kubernetes import client
from kubernetes.client.rest import ApiException

from ..controllers.ai_task import AITASK_STORE_NAMESPACE
from ..services.auth import get_user_id_from_token
from ..services.agent.loader import _load_k8s_config

router = APIRouter(prefix="/v1/api", tags=["aitasks"])

NAMESPACE = "cattle-ai-agent-system"
LABEL_TASK_DEFINITION = "ai.cattle.io/task-definition"
_CRD = {"group": "ai.cattle.io", "version": "v1alpha1", "plural": "aitasks"}


def _custom_objects_api() -> client.CustomObjectsApi:
    _load_k8s_config()
    return client.CustomObjectsApi()


def _summary(task: dict) -> dict:
    """Summary of an AITask CR, as returned by the list endpoint."""
    metadata = task.get("metadata", {})
    spec = task.get("spec", {})
    task_status = task.get("status") or {}
    return {
        "name": metadata.get("name"),
        "taskDefinitionRef": spec.get("taskDefinitionRef"),
        "phase": task_status.get("phase") or "Pending",
        "startTime": task_status.get("startTime"),
        "completionTime": task_status.get("completionTime"),
        "createdAt": metadata.get("creationTimestamp"),
    }


# TODO check if user has access to the requested AITask!!
async def _require_user(request: Request) -> str:
    user_id = await get_user_id_from_token(request.cookies)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user_id


@router.get("/aitasks")
async def get_ai_tasks(request: Request, definition: str | None = None) -> JSONResponse:
    """
    List AITasks, newest first. ``definition`` filters by AITaskDefinition name.
    """
    await _require_user(request)

    try:
        api = _custom_objects_api()
        kwargs = {"label_selector": f"{LABEL_TASK_DEFINITION}={definition}"} if definition else {}
        tasks = await asyncio.to_thread(api.list_namespaced_custom_object, namespace=NAMESPACE, **_CRD, **kwargs)

        items = [_summary(t) for t in tasks.get("items", [])]
        items.sort(key=lambda t: t.get("createdAt") or "", reverse=True)
        return JSONResponse(status_code=status.HTTP_200_OK, content=items)
    except Exception as e:
        logging.error(f"Error listing AITasks: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )


@router.get("/aitasks/{name}")
async def get_ai_task(request: Request, name: str) -> JSONResponse:
    """
    Get an AITask with the outcome stored by the executor (``output``/``error``).
    """
    await _require_user(request)

    try:
        api = _custom_objects_api()
        try:
            task = await asyncio.to_thread(api.get_namespaced_custom_object, namespace=NAMESPACE, name=name, **_CRD)
        except ApiException as e:
            if e.status == status.HTTP_404_NOT_FOUND:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AITask not found")
            raise

        spec = task.get("spec", {})
        uid = task.get("metadata", {}).get("uid", "")
        item = await request.app.memory_manager.get_store().aget(AITASK_STORE_NAMESPACE, uid) if uid else None
        stored = item.value if item else {}

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                **_summary(task),
                "prompt": spec.get("prompt"),
                "agents": spec.get("agents") or [],
                "output": stored.get("output"),
                "error": stored.get("error"),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching AITask {name}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )
