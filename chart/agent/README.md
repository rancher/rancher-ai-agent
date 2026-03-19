# Rancher AI Agent Helm Chart

Helm chart for deploying the Rancher AI Agent and Rancher MCP server into the `cattle-ai-agent-system` namespace.

## Chart Metadata

- Name: `rancher-ai-agent`

## Install

```bash
helm upgrade rancher-ai-agent rancher-ai/agent \
  --namespace cattle-ai-agent-system \
  --create-namespace \
  -f values.yaml
```

## Values Schema Reference

The chart validates values using `values.schema.json` (JSON Schema draft-07).

<!-- BEGIN VALUES TABLE (auto-generated) -->
| Field | Type | Default | Description |
|---|---|---|---|
| `global.cattle.systemDefaultRegistry` | `string` | `""` | Optional global registry prefix used for chart images. |
| `imagePullSecrets` | `array<object>` | `[]` (unset) | Optional image pull secrets applied to both agent and MCP deployments. |
| `imagePullSecrets[].name` | `string` | n/a | Name of a Kubernetes secret used to pull private images. |
| `aiAgent.image.repository` | `string` | `rancher/rancher-ai-agent` | Container image repository for the Rancher AI Agent. |
| `aiAgent.image.tag` | `string` | `v1.0.0` | Container image tag for the Rancher AI Agent. |
| `aiAgent.image.pullPolicy` | `string` (`Always`|`IfNotPresent`|`Never`) | `IfNotPresent` | Image pull policy for the Rancher AI Agent container. |
| `mcp.readOnly` | `boolean` | `false` | Runs MCP server in read-only mode (`--read-only`) when enabled. |
| `mcp.image.repository` | `string` | `rancher/rancher-ai-mcp` | Container image repository for the MCP server. |
| `mcp.image.tag` | `string` | `v1.0.0` | Container image tag for the MCP server. |
| `mcp.image.pullPolicy` | `string` (`Always`|`IfNotPresent`|`Never`) | `IfNotPresent` | Image pull policy for the MCP server container. |
| `ollamaLlmModel` | `string` | `""` | Ollama model identifier exposed to the agent. |
| `geminiLlmModel` | `string` | `""` | Gemini model identifier exposed to the agent. |
| `openaiLlmModel` | `string` | `""` | OpenAI model identifier exposed to the agent. |
| `bedrockLlmModel` | `string` | `""` | AWS Bedrock model identifier exposed to the agent. |
| `ollamaUrl` | `string` | `""` | Base URL for Ollama API endpoint. |
| `googleApiKey` | `string` | `""` | API key for Google Gemini provider. |
| `openaiApiKey` | `string` | `""` | API key for OpenAI provider. |
| `openaiUrl` | `string` | `""` | Optional custom OpenAI-compatible base URL. |
| `awsBedrock.bearerToken` | `string` | `""` | Bearer token for Bedrock access, if required by your auth flow. |
| `awsBedrock.region` | `string` | `""` | AWS region used for Bedrock requests. |
| `activeLlm` | `string` (`ollama`|`gemini`|`openai`|`bedrock`) | `""` | Selects the active LLM provider used by the agent. |
| `rag.enabled` | `boolean` | `false` | Enables Retrieval-Augmented Generation (RAG) support. |
| `rag.embeddings_model` | `string` | `""` | Embeddings model used by RAG pipeline. |
| `rag.pvc` | `string` | `""` | Existing PVC name mounted at `/app/rag` for RAG data. |
| `langfuseSecretKey` | `string` | `""` | Langfuse secret key for tracing/observability integration. |
| `langfusePublicKey` | `string` | `""` | Langfuse public key for tracing/observability integration. |
| `langfuseHost` | `string` | `""` | Langfuse host URL. |
| `storage.enabled` | `boolean` | `false` | Enables database-backed storage in the agent. |
| `storage.connectionString` | `string` | `""` | Connection string used when `storage.enabled=true`. |
| `probes.startup.initialDelaySeconds` | `integer` (min `0`) | `5` | Startup probe initial delay. |
| `probes.startup.periodSeconds` | `integer` (min `1`) | `2` | Startup probe period. |
| `probes.startup.failureThreshold` | `integer` (min `1`) | `10` | Startup probe failure threshold. |
| `probes.liveness.initialDelaySeconds` | `integer` (min `0`) | `0` | Liveness probe initial delay. |
| `probes.liveness.periodSeconds` | `integer` (min `1`) | `15` | Liveness probe period. |
| `probes.liveness.timeoutSeconds` | `integer` (min `1`) | `5` | Liveness probe timeout. |
| `probes.liveness.failureThreshold` | `integer` (min `1`) | `3` | Liveness probe failure threshold. |
| `probes.readiness.initialDelaySeconds` | `integer` (min `0`) | `0` | Readiness probe initial delay. |
| `probes.readiness.periodSeconds` | `integer` (min `1`) | `10` | Readiness probe period. |
| `probes.readiness.timeoutSeconds` | `integer` (min `1`) | `5` | Readiness probe timeout. |
| `probes.readiness.failureThreshold` | `integer` (min `1`) | `2` | Readiness probe failure threshold. |
| `insecureSkipTls` | `boolean` | `false` | Disables MCP TLS verification and enables insecure MCP mode when true. |
| `llmMock.enabled` | `boolean` | `false` | Enables mock LLM endpoint usage in the agent. |
| `llmMock.url` | `string` | `""` | Mock LLM endpoint URL used when `llmMock.enabled=true`. |
| `log.level` | `string` | `info` | Log level passed to agent and MCP server. |
<!-- END VALUES TABLE (auto-generated) -->

## Notes

- Values like API keys are rendered into Kubernetes Secrets by chart templates.
- Leave provider-specific credentials unset when not using that provider.
- Set `activeLlm` to one of: `ollama`, `gemini`, `openai`, or `bedrock`.
