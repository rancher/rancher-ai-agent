# Contributing to Rancher AI Agent

> ⚠️ **Important Note:** This setup is strictly for local debugging and development purposes. Do not use these configurations in a production environment.

## Prerequisites

Installed:
- [python3](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [fastapi](https://fastapi.tiangolo.com/tutorial/#install-fastapi)

Before getting started, ensure you have the following running:
* **Rancher** (Local or accessible remote installation)
* **Rancher MCP** (Model Context Protocol) running locally or accessible via network

---

## How to Run the Rancher AI Agent Locally

### 1. Environment Variables Configuration

The following environment variables must be set at minimum to run the agent: `RANCHER_URL`, `RANCHER_API_TOKEN`, `MCP_URL`, `ACTIVE_LLM`, and the corresponding model variable for your chosen provider (e.g. `GEMINI_MODEL`). All other variables are optional unless noted. You can export these in your terminal, add them to a `.env` file, or include them in your IDE launch configuration.

**General**

| Variable | Example Value | Description |
| :--- | :--- | :--- |
| `INSECURE_SKIP_TLS` | `TRUE` | Skips TLS verification for local dev. |
| `ENABLE_TEST_UI` | `TRUE` | Enables the built-in testing user interface. |
| `LOG_LEVEL` | `INFO` | Log verbosity level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Defaults to `INFO`. |
| `RANCHER_URL` | `https://rancher.example.com` | URL of your Rancher instance. |
| `RANCHER_API_TOKEN` | `token-xxx` | Your Rancher API token (can be extracted from the `R_SESS` cookie). |
| `MCP_URL` | `http://localhost:9092` | Points to your running Rancher MCP instance. |

**LLM Provider**

| Variable | Example Value | Description |
| :--- | :--- | :--- |
| `ACTIVE_LLM` | `gemini` | The active LLM provider. One of: `ollama`, `gemini`, `openai`, `bedrock`. |
| `OLLAMA_MODEL` | `llama3` | Model name to use with Ollama. |
| `OLLAMA_URL` | `http://localhost:11434` | Base URL of the Ollama server. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model name to use with Gemini. For alternative models and providers, see the [LLM Configuration Chart](https://github.com/rancher/rancher-ai-agent/blob/main/chart/agent/templates/llm-config.yaml#L7). |
| `GOOGLE_API_KEY` | `AIza...` | API key for Google Gemini. |
| `OPENAI_MODEL` | `gpt-4o` | Model name to use with OpenAI. |
| `OPENAI_API_KEY` | `sk-...` | API key for OpenAI. |
| `OPENAI_URL` | `http://localhost:8080` | Optional custom OpenAI-compatible endpoint URL. |
| `BEDROCK_MODEL` | `us.anthropic.claude-sonnet-4-6` | Model ID to use with AWS Bedrock. |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for Bedrock. Required when `ACTIVE_LLM=bedrock`. |
| `AWS_BEARER_TOKEN_BEDROCK` | `ABSK...` | AWS bearer token for Bedrock authentication. |

**Memory / Persistence**

| Variable | Example Value | Description |
| :--- | :--- | :--- |
| `DB_ENABLED` | `false` | Set to `true` to use PostgreSQL for conversation memory instead of in-memory storage. |
| `DB_CONNECTION_STRING` | `postgresql://user:pass@localhost/db` | PostgreSQL connection string. Required when `DB_ENABLED=true`. |

**Observability (Optional)**

| Variable | Example Value | Description |
| :--- | :--- | :--- |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse server URL for LLM tracing. |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-...` | Langfuse public key. |
| `LANGFUSE_SECRET_KEY` | `sk-lf-...` | Langfuse secret key. |

**Testing / Mocking (Optional)**

| Variable | Example Value | Description |
| :--- | :--- | :--- |
| `LLM_MOCK_ENABLED` | `true` | Enables a mock LLM server instead of a real provider. |
| `LLM_MOCK_URL` | `http://localhost:9999` | URL of the mock LLM server. Required when `LLM_MOCK_ENABLED=true`. |

### 2. Prepare the Cluster

Ensure that wherever the execution commands are being run, your kubeconfig is pointed to the Rancher management cluster.

```bash
export KUBECONFIG=path/to/kubeconfig.yaml
```

Before running the agent, you need to apply the CRDs and create the namespace that will be used.

```bash
kubectl apply -f chart/agent/templates/crds/ai.cattle.io_aiagentconfigs.yaml
kubectl create ns cattle-ai-agent-system
```

### 3. Execution Commands

**Install dependencies:**

```bash
uv sync
```

**Start the application in development mode:**

```bash
fastapi dev app/main.py
```

**Accessing the local test UI**

Once the development server is up and running (and `ENABLE_TEST_UI` is set to `TRUE`), you can access and interact with the built-in agent testing interface by navigating to http://localhost:8000/ui in your web browser.

**Run tests:**

```bash
 uv run pytest tests/
```

### 4. Access Test UI (Optional)

After step 3, the agent is running and tests can be run, but if you would like to interact with the AI in the Rancher UI there are a few more steps.

#### Install UI Extension

1. In Rancher, go to the Extensions tab (it's a puzzle piece in the side menu)
2. Click the three dots in the top right
3. Select "Manage Repositories"
4. Create a new repository
5. Select Git Repository, name it anything you'd like. Set Git Repo URL to "https://github.com/torchiaf/rancher-ai-ui" and Git Branch to "gh-pages"
6. Go back to the Extensions tab and install the AI Assistant

> **Note:** The Git repo is a fork of the official UI repo and is highly experimental.

#### Access the Test UI

The Rancher AI Agent pod is not running, so you can't use the normal Liz menu from the UI. Instead, go to "http://localhost:8000/ui" to interact with your running AI Agent.

In order to submit context into this AI Agent, you can pass in JSON specifying the agent, context and prompt:

```json
{"prompt": "show deployments", "agent": "rancher", "context": {"cluster": "local"}}
```

## VS Code Debugging Configuration

To debug the application directly inside VS Code, add the following configuration to your .vscode/launch.json file.


```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug app/main.py",
            "type": "debugpy",
            "request": "launch",
            "module": "fastapi",
            "args": ["dev", "main.py"],
            "python": "${workspaceFolder}/.venv/bin/python",
            "cwd": "${workspaceFolder}/app",
            "console": "integratedTerminal",
            "envFile": "/home/dirwithcredentials", # Update the envFile path or the env object values to match your local credentials
            "justMyCode": false,
            "env": {
                "INSECURE_SKIP_TLS": "TRUE",
                "ENABLE_TEST_UI": "TRUE",
                "MCP_URL": "http://localhost:9092",
                "ACTIVE_LLM": "gemini",
                "GEMINI_MODEL": "gemini-2.5-flash",
                "RANCHER_URL": "yourrancherurl",
                "LOG_LEVEL": "DEBUG",
                "RANCHER_API_TOKEN": "token-xxx",
                "LANGFUSE_HOST": "http://0.0.0.0", # Optional if you want to use langfuse
                "LANGFUSE_PUBLIC_KEY": "xxx",
                "LANGFUSE_SECRET_KEY": "xxx"
            }
        }
    ]
}
```