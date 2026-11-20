import logging
import os

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from ..constants import AITASK_STORE_NAMESPACE, LABEL_TASK_DEFINITION
from ..services.rbac import (
    AITASK_DEFINITION_STEVE_TYPE,
    AITASK_STEVE_TYPE,
    AccessDenied,
    NotFound,
    RBACError,
    accessible_resource,
    accessible_resources,
)

router = APIRouter(prefix="/v1/api", tags=["aitasks"])

NAMESPACE = "cattle-ai-agent-system"

# Label values are at most 63 chars of [a-z0-9.-], starting/ending alphanumeric.
# AITaskDefinition names are used as label values, so this also keeps the label
# selector built from ``definition`` a single equality term.
_LABEL_VALUE_PATTERN = r"^[a-z0-9]([-a-z0-9.]{0,61}[a-z0-9])?$"


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


def _definition_summary(definition: dict) -> dict:
    """Summary of an AITaskDefinition CR, as returned by the list endpoint."""
    metadata = definition.get("metadata", {})
    spec = definition.get("spec", {})
    definition_status = definition.get("status") or {}
    return {
        "name": metadata.get("name"),
        "displayName": spec.get("displayName"),
        "prompt": spec.get("prompt"),
        "agents": spec.get("agents") or [],
        "schedule": spec.get("schedule") or "",
        "timeZone": spec.get("timeZone") or "UTC",
        "suspend": bool(spec.get("suspend")),
        "lastScheduleTime": definition_status.get("lastScheduleTime"),
        "nextRunTime": definition_status.get("nextRunTime"),
        "conditions": definition_status.get("conditions") or [],
        "createdAt": metadata.get("creationTimestamp"),
    }


def _session_token(request: Request) -> str:
    """
    Rancher session token of the caller. AITasks are read through Rancher as this
    user, so Rancher authenticates the token and enforces the user's (and their
    groups') permissions on ``aitasks.ai.cattle.io``.
    """
    token = os.environ.get("RANCHER_API_TOKEN", request.cookies.get("R_SESS", ""))
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return token


def _rbac_http_error(e: RBACError, what: str) -> HTTPException:
    if isinstance(e, AccessDenied):
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    if isinstance(e, NotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{what} not found")
    logging.error(f"Error resolving {what}: {e}")
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Unable to verify permissions with Rancher")


@router.get("/aitaskdefinitions")
async def get_ai_task_definitions(request: Request) -> JSONResponse:
    """
    List the AITaskDefinitions the user may access, sorted by display name.
    """
    token = _session_token(request)

    try:
        definitions = await accessible_resources(token, AITASK_DEFINITION_STEVE_TYPE, NAMESPACE)
    except RBACError as e:
        raise _rbac_http_error(e, "AITaskDefinitions")

    try:
        items = [_definition_summary(d) for d in definitions]
        items.sort(key=lambda d: ((d.get("displayName") or d.get("name") or "").lower(), d.get("name") or ""))
        return JSONResponse(status_code=status.HTTP_200_OK, content=items)
    except Exception as e:
        logging.error(f"Error listing AITaskDefinitions: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )


@router.get("/aitasks")
async def get_ai_tasks(
    request: Request,
    definition: str = Query(..., max_length=63, pattern=_LABEL_VALUE_PATTERN),
) -> JSONResponse:
    """
    List the AITasks of AITaskDefinition ``definition`` the user may access,
    newest first.
    """
    token = _session_token(request)
    label_selector = f"{LABEL_TASK_DEFINITION}={definition}"

    try:
        tasks = await accessible_resources(token, AITASK_STEVE_TYPE, NAMESPACE, label_selector)
    except RBACError as e:
        raise _rbac_http_error(e, "AITasks")

    try:
        items = [_summary(t) for t in tasks]
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
    Get an AITask (if the user may access it) with the outcome stored by the
    executor (``output``/``error``).
    """
    token = _session_token(request)

    try:
        task = await accessible_resource(token, AITASK_STEVE_TYPE, NAMESPACE, name)
    except RBACError as e:
        raise _rbac_http_error(e, "AITask")

    try:
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
    except Exception as e:
        logging.error(f"Error fetching AITask {name}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )
