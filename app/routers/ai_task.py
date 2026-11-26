import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from fastapi.responses import JSONResponse

from ..constants import (
    AITASK_STORE_NAMESPACE,
    ANNOTATION_RUN_REQUESTED_AT,
    LABEL_TASK_DEFINITION,
    LABEL_TRIGGER,
    LABEL_VALUE_PATTERN,
)
from ..controllers.ai_task_definition import _api as _k8s_api
from ..controllers.ai_task_definition import has_active_task, manual_task_name
from ..services.aitask_chat import aitask_chat_id, session_token
from ..services.auth import get_user_id_from_token
from ..services.rbac import (
    AITASK_DEFINITION_STEVE_TYPE,
    AITASK_STEVE_TYPE,
    AccessDenied,
    NotFound,
    RBACError,
    accessible_resource,
    accessible_resources,
    patch_resource,
)

router = APIRouter(prefix="/v1/api", tags=["aitasks"])

NAMESPACE = "cattle-ai-agent-system"

# AITaskDefinition names are used as label values, so validating them as such also
# keeps the label selector built from ``definition`` a single equality term.
_LABEL_VALUE_PATTERN = LABEL_VALUE_PATTERN


def _summary(task: dict) -> dict:
    """Summary of an AITask CR, as returned by the list endpoint."""
    metadata = task.get("metadata", {})
    spec = task.get("spec", {})
    task_status = task.get("status") or {}
    return {
        "name": metadata.get("name"),
        "taskDefinitionRef": spec.get("taskDefinitionRef"),
        "trigger": (metadata.get("labels") or {}).get(LABEL_TRIGGER),
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
    token = session_token(request.cookies)
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


@router.post("/aitaskdefinitions/{name}/run", status_code=status.HTTP_202_ACCEPTED)
async def run_ai_task_definition(
    request: Request,
    name: str = Path(..., max_length=63, pattern=_LABEL_VALUE_PATTERN),
) -> JSONResponse:
    """
    Run an AITaskDefinition now.

    Sets the run-requested-at annotation on the definition as the calling user, so
    Rancher checks that the user may ``patch`` it. The scheduler controller then
    creates the AITask, whose name is returned. Returns 409 if an AITask of the
    definition is still active.
    """
    token = _session_token(request)

    try:
        await accessible_resource(token, AITASK_DEFINITION_STEVE_TYPE, NAMESPACE, name)
    except RBACError as e:
        raise _rbac_http_error(e, "AITaskDefinition")

    try:
        active = await has_active_task(_k8s_api(), NAMESPACE, name)
    except Exception as e:
        logging.error(f"Error checking active AITasks of {name}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    if active:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An AITask of this definition is still active")

    requested_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    patch = {"metadata": {"annotations": {ANNOTATION_RUN_REQUESTED_AT: requested_at}}}
    try:
        await patch_resource(token, AITASK_DEFINITION_STEVE_TYPE, NAMESPACE, name, patch)
    except RBACError as e:
        raise _rbac_http_error(e, "AITaskDefinition")

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"name": manual_task_name(name, requested_at), "requestedAt": requested_at},
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
    executor (``output``/``error``) and the id of the user's follow-up chat on it
    (``chatId``, opened with ``/v1/ws/messages?aitask=<name>``; the chat exists once
    the user has sent a first message).
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
        user_id = await get_user_id_from_token(request.cookies) if uid else None

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                **_summary(task),
                "prompt": spec.get("prompt"),
                "agents": spec.get("agents") or [],
                "output": stored.get("output"),
                "error": stored.get("error"),
                "chatId": aitask_chat_id(uid, user_id) if user_id else None,
            },
        )
    except Exception as e:
        logging.error(f"Error fetching AITask {name}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )
