# Contributing to Rancher AI Agent

> ⚠️ **Important Note:** This setup is strictly for local debugging and development purposes. Do not use these configurations in a production environment.

## Prerequisites

Before getting started, ensure you have the following installed and running:
* **Rancher** (Local or accessible remote installation)
* **Rancher MCP** (Model Context Protocol) running locally or accessible via network

---

## How to Run the Rancher AI Agent Locally

### 1. Environment Variables Configuration

You need to configure the following environment variables. You can export these in your terminal, add them to a `.env` file, or include them in your IDE launch configuration.

| Variable | Example Value | Description |
| :--- | :--- | :--- |
| `INSECURE_SKIP_TLS` | `TRUE` | Skips TLS verification for local dev. |
| `ENABLE_TEST_UI` | `TRUE` | Enables the built-in testing user interface. |
| `MCP_URL` | `http://localhost:9092` | Points to your running Rancher MCP instance. |
| `RANCHER_API_TOKEN` | `token-xxx` | Your Rancher API token (can be extracted from the `R_SESS` cookie). |
| `ACTIVE_LLM` | `gemini` | The active LLM provider to use. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | The specific model to target. For alternative models and providers, see the [LLM Configuration Chart](https://github.com/rancher/rancher-ai-agent/blob/main/chart/agent/templates/llm-config.yaml#L7). |

### 2. Execution Commands

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