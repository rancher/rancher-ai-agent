# AIAgentConfig CRD Generation

Generate the Kubernetes CRD for `AIAgentConfig`.

## Example AIAgentConfig

```yaml
apiVersion: ai.cattle.io/v1alpha1
kind: AIAgentConfig
metadata:
  name: rancher-core-agent
  namespace: cattle-ai-agent-system
spec:
  agent: "Rancher Core Agent"
  description: "AI agent for Rancher management"
  systemPrompt: "You are a helpful assistant for Rancher."
  mcpURL: "http://mcp-server:8080"
  authenticationType: RANCHER
  humanValidationTools:
    - name: "patch_resource"
      type: UPDATE
```
