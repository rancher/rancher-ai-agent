import logging
import base64
import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from botocore.exceptions import InvalidRegionError
from kubernetes import client, config as k8s_config
from kubernetes.client.rest import ApiException

from ..services.llm import LLMManager
from ..services.auth import get_user_id_from_request

router = APIRouter(prefix="/v1/api", tags=["configuration"])

AGENT_NAMESPACE = "cattle-ai-agent-system"
SETTINGS_SECRET_NAME = "llm-secret"
SETTINGS_CONFIGMAP_NAME = "llm-config"
# Fields stored in the ConfigMap (plain text) rather than the Secret
CONFIGMAP_FIELDS = {"OLLAMA_MODEL", "BEDROCK_MODEL", "OPENAI_MODEL", "GEMINI_MODEL", "ACTIVE_LLM"}

AVAILABLE_LLM_PROVIDERS = {"ollama", "openai", "gemini", "bedrock"}

# Hardcoded models
AVAILABLE_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
        "o3",
        "gpt-4.1",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "gemini": [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
    "bedrock": [],
    "ollama": []
}

class SettingsUpdate(BaseModel):
    """Model for updating agent settings."""
    OPENAI_API_KEY: str = None
    OPENAI_URL: str = None
    OLLAMA_MODEL: str = None
    GEMINI_MODEL: str = None
    OPENAI_MODEL: str = None
    BEDROCK_MODEL: str = None
    OLLAMA_URL: str = None
    GOOGLE_API_KEY: str = None
    ACTIVE_LLM: str = None
    ENABLE_RAG: str = None
    EMBEDDINGS_MODEL: str = None
    LANGFUSE_HOST: str = None
    LANGFUSE_PUBLIC_KEY: str = None
    LANGFUSE_SECRET_KEY: str = None
    AWS_REGION: str = None
    AWS_BEARER_TOKEN_BEDROCK: str = None

async def check_k8s_permission(
    user_id: str,
    verb: str,
    resource: str,
    namespace: str
) -> bool:
    """
    Check if a user has permission to perform an action on a Kubernetes resource using SubjectAccessReview.
    
    Args:
        user_id: The user ID to check permissions for
        verb: The action to perform (e.g., 'patch', 'update', 'get')
        resource: The resource type (e.g., 'secrets', 'configmaps')
        namespace: The namespace to check permissions in (defaults to the AI agent namespace)
    """    
    try:
        try:
            # Load Kubernetes config
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            # Fall back to kubeconfig (for local development)
            k8s_config.load_kube_config()

        auth_api = client.AuthorizationV1Api()
        
        # Create SubjectAccessReview with resource attributes
        sar = client.V1SubjectAccessReview(
            metadata=client.V1ObjectMeta(),
            spec=client.V1SubjectAccessReviewSpec(
                user=user_id,
                resource_attributes=client.V1ResourceAttributes(
                    verb=verb,
                    resource=resource,
                    namespace=namespace,
                )
            )
        )
        
        # Send SubjectAccessReview to Kubernetes API
        response = auth_api.create_subject_access_review(sar)
        
        logging.debug(f"Permission check for user: {verb} {resource} in {namespace} - Allowed: {response.status.allowed}")
        return response.status.allowed
    
    except ApiException as e:
        logging.error(f"Kubernetes API error during permission check: {e}")
        return False
    except Exception as e:
        logging.error(f"Error checking Kubernetes permissions: {e}")
        return False

@router.get("/llm/{llm_name}/models")
async def get_models(request: Request, llm_name: str):
    """
    Endpoint to retrieve available LLM models.
    For ollama: requires 'url' query parameter
    For bedrock: requires 'region' and 'bearer_token'
    """
    try:
        user_id = await get_user_id_from_request(request)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if llm_name not in AVAILABLE_MODELS:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported LLM provider: {llm_name}")

        models = AVAILABLE_MODELS[llm_name]

        if llm_name == "ollama":
            ollama_url = request.query_params.get("url")
            
            if not ollama_url:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ollama URL is required")

            try:
                async with httpx.AsyncClient(timeout=2.0) as http_client:
                    response = await http_client.get(f"{ollama_url}/api/tags")
                    if response.status_code == 200:
                        ollama_data = response.json()
                        ollama_models = [model["name"] for model in ollama_data.get("models", [])]
                        models = ollama_models
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"Ollama server returned status {response.status_code}"
                        )
            except httpx.InvalidURL as e:
                logging.error(f"Invalid Ollama URL: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid Ollama URL: {ollama_url}"
                )
            except httpx.RequestError as e:
                logging.error(f"Failed to fetch Ollama models: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"The Ollama server is not available at the provide URL"
                )

        elif llm_name == "bedrock":
            region = request.query_params.get("region")
            bearer_token = request.query_params.get("bearerToken")
            
            if not region:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="AWS region is required for Bedrock")
            
            if not bearer_token:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The Bearer Token is required for Bedrock")
            
            # Use bearer token authentication
            logging.info("Using bearer token authentication for Bedrock")
            try:
                # Use direct HTTP request with bearer token
                async with httpx.AsyncClient(timeout=2.0) as http_client:
                    headers = {"Authorization": f"Bearer {bearer_token}"}
                    response = await http_client.get(
                        f"https://bedrock.{region}.amazonaws.com/foundation-models",
                        headers=headers
                    )
                    if response.status_code == 200:
                        data = response.json()
                        bedrock_models = [model['modelId'] for model in data.get('modelSummaries', [])]
                        
                        # Extract regional prefix from region and ensure model IDs are prefixed with the region if not already
                        # e.g. 'eu-west-1' -> 'eu.{modelId}', 'us-east-1' -> 'us.{modelId}'
                        region_prefix = region.split('-')[0]
                        if region_prefix not in ['us', 'eu', 'ap', 'ca']:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Invalid Bedrock region format: {region}"
                            )
                        bedrock_models = [
                            model if model.startswith(f"{region_prefix}.") else f"{region_prefix}.{model}"
                            for model in bedrock_models
                        ]
                        
                        models = bedrock_models
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED if response.status_code == 401 else status.HTTP_502_BAD_GATEWAY,
                            detail=f"Bedrock authentication failed. Make sure the Region and Bearer Token are valid and have the necessary permissions."
                        )
            except InvalidRegionError as e:
                logging.error(f"Invalid AWS region: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid AWS region: {region}"
                )
            except httpx.InvalidURL as e:
                logging.error(f"Invalid region format for Bedrock URL: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid AWS region: {region}"
                )
            except httpx.RequestError as e:
                logging.error(f"{e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"{e}"
                )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=models
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error retrieving models: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
    
@router.get("/settings")
async def get_settings(request: Request):
    """
    Endpoint to retrieve current agent settings.
    """
    try:
        user_id = await get_user_id_from_request(request)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        storage_type = request.app.memory_manager.storage_type.value
        
        settings = {"storageType": storage_type}
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=settings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error retrieving settings: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.put("/settings")
async def update_settings(settings: SettingsUpdate, request: Request):
    """
    Endpoint to update agent settings by patching the llm-config secret.
    Requires permission to patch secrets in the agent namespace.
    """
    try:
        user_id = await get_user_id_from_request(request)

        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        # Check if user has permission to update llm-config secret
        has_permission = await check_k8s_permission(
            user_id=user_id,
            verb="update",
            resource="secrets",
            namespace=AGENT_NAMESPACE
        )

        if not has_permission:
            logging.warning(f"User attempted to update settings without permission")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": f"User does not have permission to update settings in namespace {AGENT_NAMESPACE}"}
            )

        logging.debug(f"User is updating settings: {settings}")

        # Validate settings based on ACTIVE_CHATBOT
        updated_fields = settings.model_dump(exclude_none=True)
        active_chatbot = updated_fields.get("ACTIVE_LLM")
        
        if active_chatbot:
            if active_chatbot.lower() not in AVAILABLE_LLM_PROVIDERS:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": f"ACTIVE_LLM must be one of: {', '.join(sorted(AVAILABLE_LLM_PROVIDERS))}. Got: {active_chatbot}"}
                )
            
            # Validate based on the active chatbot type
            if active_chatbot.lower() == "ollama":
                if not updated_fields.get("OLLAMA_URL") or not updated_fields.get("OLLAMA_MODEL"):
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "OLLAMA_URL and OLLAMA_MODEL are required when ACTIVE_LLM is 'ollama'"}
                    )
            
            elif active_chatbot.lower() == "bedrock":
                if not updated_fields.get("AWS_REGION") or not updated_fields.get("BEDROCK_MODEL"):
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "AWS_REGION and BEDROCK_MODEL are required when ACTIVE_LLM is 'bedrock'"}
                    )
                
                # Check for authentication method
                has_bearer = updated_fields.get("AWS_BEARER_TOKEN_BEDROCK")
                
                if not has_bearer:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "AWS_BEARER_TOKEN_BEDROCK is required when ACTIVE_LLM is 'bedrock'"}
                    )
            
            elif active_chatbot.lower() == "openai":
                if not updated_fields.get("OPENAI_API_KEY") or not updated_fields.get("OPENAI_MODEL"):
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "OPENAI_API_KEY and OPENAI_MODEL are required when ACTIVE_LLM is 'openai'"}
                    )
            
            elif active_chatbot.lower() == "gemini":
                if not updated_fields.get("GOOGLE_API_KEY") or not updated_fields.get("GEMINI_MODEL"):
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "GOOGLE_API_KEY and GEMINI_MODEL are required when ACTIVE_LLM is 'gemini'"}
                    )

        try:
            try:
                # Load Kubernetes config
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                # Fall back to kubeconfig (for local development)
                k8s_config.load_kube_config()
            
            v1 = client.CoreV1Api()
            
            # Split fields into configmap vs secret groups
            all_fields = settings.model_dump(exclude_none=True)
            configmap_updates = {k: v for k, v in all_fields.items() if k in CONFIGMAP_FIELDS}
            secret_updates = {k: v for k, v in all_fields.items() if k not in CONFIGMAP_FIELDS}

            response_data = {}

            # Update ConfigMap fields
            if configmap_updates:
                configmap = v1.read_namespaced_config_map(SETTINGS_CONFIGMAP_NAME, AGENT_NAMESPACE)
                cm_data = configmap.data or {}
                existing_cm_keys = set(cm_data.keys())
                for field_name, value in configmap_updates.items():
                    if field_name in existing_cm_keys:
                        cm_data[field_name] = str(value)
                        logging.debug(f"Updated {field_name} in configmap")
                configmap.data = cm_data
                v1.patch_namespaced_config_map(SETTINGS_CONFIGMAP_NAME, AGENT_NAMESPACE, configmap)

            # Update Secret fields
            if secret_updates:
                secret = v1.read_namespaced_secret(SETTINGS_SECRET_NAME, AGENT_NAMESPACE)
                secret_data = secret.data or {}
                existing_secret_keys = set(secret_data.keys())
                for field_name, value in secret_updates.items():
                    if field_name in existing_secret_keys:
                        secret_data[field_name] = base64.b64encode(str(value).encode()).decode()
                        logging.debug(f"Updated {field_name} in secret")
                secret.data = secret_data
                v1.patch_namespaced_secret(SETTINGS_SECRET_NAME, AGENT_NAMESPACE, secret)
            
            # Reset LLMManager singleton to force reinitialization for consistency, but as of now we are redeploying the agent...
            LLMManager._instance = None

            # Re-fetch both resources to return the updated content
            updated_cm = v1.read_namespaced_config_map(SETTINGS_CONFIGMAP_NAME, AGENT_NAMESPACE)
            updated_secret = v1.read_namespaced_secret(SETTINGS_SECRET_NAME, AGENT_NAMESPACE)
            response_data.update(updated_cm.data or {})
            response_data.update(updated_secret.data or {})

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=response_data
            )

        except ApiException as e:
            logging.error(f"Kubernetes API error updating settings: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Failed to update settings: {str(e)}"}
            )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating settings: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Failed to update settings: {str(e)}"}
        )
