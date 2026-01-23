"""Load AIAgentConfig CRDs from Kubernetes cluster."""

import logging

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel
from kubernetes import client, config
from kubernetes.client.rest import ApiException

NAMESPACE = "cattle-ai-agent-system"
CRD_GROUP = "ai.cattle.io"
CRD_VERSION = "v1alpha1"
CRD_PLURAL = "aiagentconfigs"


class AuthenticationType(str, Enum):
    """Authentication types for agents."""
    NONE = "NONE"
    RANCHER = "RANCHER"
    BASIC = "BASIC"


class ToolActionType(str, Enum):
    """Action types for human validation tools."""
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


RANCHER_AGENT_PROMPT = """You are a helpful and expert AI assistant integrated directly into the Rancher UI. Your primary goal is to assist users in managing their Kubernetes clusters and resources through the Rancher interface. You are a trusted partner, providing clear, confident, and safe guidance.

## CORE DIRECTIVES

### UI-First Mentality
* NEVER suggest using `kubectl`, `helm`, or any other CLI tool UNLESS explicitely provided by the `retrieve_rancher_docs` tool.
* All actions and information should reflect what the user can see and click on inside the Rancher UI.

### Context Awareness
* Always consider the user's current context (cluster, project, or resource being viewed).
* If context is missing, ask clarifying questions before taking action.

## BUILDING USER TRUST

### 1. Reasoning Transparency
Always explain why you reached a conclusion, connecting it to observed data.
* Good: "The pod has restarted 12 times. This often indicates a crash loop."
* Bad: "The pod is unhealthy."

### 2. Confidence Indicators
Express certainty levels with clear language and a percentage.
- High certainty: "The error is definitively caused by a missing ConfigMap (95%)."
- Likely scenarios: "The memory growth strongly suggests a leak (80%)."
- Possible causes: "Pending status could be due to insufficient resources (60%)."

### 3. Graceful Boundaries
* If an issue requires deep expertise (e.g., complex networking, storage, security):
  - "This appears to require administrative privileges or deeper system access. Please contact your cluster administrator."
* If the request is off-topic:
  - "I can't help with that, but I can show you why a pod might be stuck in CrashLoopBackOff. How can I assist with your Rancher environment?"

## Tools usage
* If the tool fails, explain the failure and suggest manual step to assist the user to answer his original question and not to troubleshoot the tool failure.

## Docs
* When relevant, always provide links to Rancher or Kubernetes documentation.

## RESOURCE CREATION & MODIFICATION

* Always generate Kubernetes YAML in a markdown code block.
* Briefly explain the resource's purpose before showing YAML.

RESPONSE FORMAT
Summarize first: Provide a clear, human-readable overview of the resource's status or configuration.
The output should always be provided in Markdown format.

- Be concise: No unnecessary conversational fluff.  
- Always end with exactly three actionable suggestions:
  - Format: <suggestion>suggestion1</suggestion><suggestion>suggestion2</suggestion><suggestion>suggestion3</suggestion>
  - No markdown, no numbering, under 60 characters each.
  - The first two suggestions must be directly relevant to the current context. If none fallback to the next rule.
  - The third suggestion should be a 'discovery' action. It introduces a related but broader Rancher or Kubernetes topic, helping the user learn.
Examples: <suggestion>How do I scale a deployment?</suggestion><suggestion>Check the resource usage for this cluster</suggestion><suggestion>Show me the logs for the failing pod</suggestion>
"""


class HumanValidationTool(BaseModel):
    name: str
    type: ToolActionType


class AgentConfig(BaseModel):
    """Configuration for a single agent."""
    name: str 
    description: str 
    system_prompt: str 
    mcp_url: str
    authentication: AuthenticationType = AuthenticationType.NONE
    toolset: Optional[str] = None
    human_validation_tools: list[HumanValidationTool] = []


def _init_k8s_client():
    """Initialize Kubernetes client."""
    try:
        # Try in-cluster config first
        config.load_incluster_config()
    except config.ConfigException:
        # Fall back to kubeconfig
        config.load_kube_config()
    
    return client.CustomObjectsApi()


def _crd_to_agent_config(crd_obj: dict) -> AgentConfig:
    """Convert CRD object to AgentConfig."""
    spec = crd_obj.get("spec", {})
    
    # Convert human validation tools
    human_validation_tools = []
    for tool in spec.get("humanValidationTools", []):
        human_validation_tools.append(
            HumanValidationTool(
                name=tool["name"],
                type=ToolActionType[tool["type"]]
            )
        )
    
    return AgentConfig(
        name=spec.get("name", ""),
        description=spec.get("description", ""),
        system_prompt=spec.get("systemPrompt", ""),
        mcp_url=spec.get("mcpURL", ""),
        authentication=AuthenticationType[spec.get("authenticationType", "NONE")],
        toolset=spec.get("toolSet", None),
        human_validation_tools=human_validation_tools,
    )


def _create_default_agents(api: client.CustomObjectsApi):
    """Create default agents in the cluster."""
    default_agents = [
        {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": "AIAgentConfig",
            "metadata": {
                "name": "rancher",
                "namespace": NAMESPACE,
            },
            "spec": {
                "name": "Rancher",
                "description": "Manages Rancher resources and operations",
                "systemPrompt": RANCHER_AGENT_PROMPT,
                "mcpURL": "rancher-mcp-server.cattle-ai-agent-system.svc",
                "authenticationType": "RANCHER",
                "builtIn": True,
                "enabled": True,
                "toolSet": "rancher",
                "humanValidationTools": [
                    {"name": "createKubernetesResource", "type": "CREATE"},
                    {"name": "patchKubernetesResource", "type": "UPDATE"},
                ]
            }
        },
         {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": "AIAgentConfig",
            "metadata": {
                "name": "fleet",
                "namespace": NAMESPACE,
            },
            "spec": {
                "name": "Fleet",
                "description": "Manages Fleet resources such as GitRepos and Bundles",
                "systemPrompt": RANCHER_AGENT_PROMPT,
                "mcpURL": "rancher-mcp-server.cattle-ai-agent-system.svc",
                "authenticationType": "RANCHER",
                "builtIn": True,
                "enabled": False,
                "toolSet": "fleet",
            }
        }
    ]
    
    for agent in default_agents:
        try:
            api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=NAMESPACE,
                plural=CRD_PLURAL,
                body=agent,
            )
            logging.info(f"Created default agent: {agent['metadata']['name']}")
        except ApiException as e:
            if e.status == 409:
                logging.debug(f"Agent {agent['metadata']['name']} already exists")
            else:
                logging.error(f"Failed to create agent {agent['metadata']['name']}: {e}")


def load_agent_configs() -> List[AgentConfig]:
    """
    Load AIAgentConfig CRDs from the cattle-ai-agent-system namespace.
    
    If no agents exist, creates the default weather, math, and rancher agents.
    
    Returns:
        List of AgentConfig objects loaded from CRDs.
    """
    try:
        api = _init_k8s_client()
        
        # Try to list existing agents
        try:
            response = api.list_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=NAMESPACE,
                plural=CRD_PLURAL,
            )
            
            items = response.get("items", [])
            
            # If no agents exist, create defaults
            if not items:
                logging.info("No AIAgentConfig found, creating default agents")
                _create_default_agents(api)
                
                # Re-fetch after creation
                response = api.list_namespaced_custom_object(
                    group=CRD_GROUP,
                    version=CRD_VERSION,
                    namespace=NAMESPACE,
                    plural=CRD_PLURAL,
                )
                items = response.get("items", [])
            
            # Convert CRDs to AgentConfig objects, only enabled ones
            agent_configs = []
            for item in items:
                spec = item.get("spec", {})
                if spec.get("enabled", True): 
                    agent_configs.append(_crd_to_agent_config(item))
            
            logging.info(f"Loaded {len(agent_configs)} enabled agent configs from CRDs")
            return agent_configs
            
        except ApiException as e:
            if e.status == 404:
                logging.warning(f"Namespace {NAMESPACE} or CRD not found")
            else:
                logging.error(f"Failed to list AIAgentConfig: {e}")
            return []
            
    except Exception as e:
        logging.error(f"Failed to initialize Kubernetes client: {e}")
        return []
