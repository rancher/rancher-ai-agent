# Middleware

Middleware intercepts the agent execution lifecycle at specific points — before/after the LLM call, after the full agent turn, or around individual tool calls. They are passed to `create_agent()` via the `middleware=` parameter and execute in order.

This package uses [LangChain's agent middleware system](https://python.langchain.com/docs/how_to/agent_middleware/) (`langchain.agents.middleware`).

## Middleware Types

There are two ways to define middleware: **decorator-based** (factory functions) and **class-based**.

### Decorator-Based (Factory Functions)

Use decorators from `langchain.agents.middleware` to create middleware as factory functions. Each decorator corresponds to a lifecycle hook:

| Decorator | When it runs | Use case |
|-----------|-------------|----------|
| `@before_model` | Before the LLM is called | Inject system messages, short-circuit the LLM call |
| `@after_model` | After the LLM responds | Enrich or modify the AIMessage |
| `@after_agent` | After the full agent turn (model + tools) completes | Post-processing, dispatching events |
| `@wrap_tool_call` | Wraps each individual tool execution | Validation gates, error handling, artifact processing |

Example — a `@before_model` middleware:

```python
from typing import Any
from langchain.agents.middleware import AgentState, before_model
from langgraph.runtime import Runtime


def create_my_middleware():
    @before_model
    def my_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Return a dict to merge into state, or None to skip
        return None

    return my_hook
```

Example — a `@wrap_tool_call` middleware:

```python
from collections.abc import Callable
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command


def create_my_tool_middleware():
    @wrap_tool_call
    async def my_wrapper(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # Pre-processing
        result = await handler(request)
        # Post-processing
        return result

    return my_wrapper
```

### Class-Based

Subclass `AgentMiddleware` when your middleware needs custom state fields or multiple hooks in one unit:

```python
from typing import Any
from langchain.agents.middleware.types import AgentMiddleware
from typing_extensions import override


class MyMiddleware(AgentMiddleware):
    # Optional: extend the agent state schema
    # state_schema = MyExtendedState

    @override
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        return None

    @override
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        return None
```

## Adding a New Middleware

1. **Create a new file** in `app/services/agent/middleware/` (e.g., `my_feature.py`).

2. **Implement** the middleware using the appropriate hook type. Import shared constants from `._constants` if needed.

3. **Export** the middleware from `__init__.py`:
   ```python
   from .my_feature import create_my_feature_middleware
   ```

4. **Register** the middleware in the agent's middleware list. Middleware is registered in `supervisor.py` (for the supervisor agent) or `child.py` (for child agents) inside the `create_agent()` call:
   ```python
   middleware=[
       # ...existing middleware...
       create_my_feature_middleware(),
   ]
   ```
   Order matters — middleware executes in list order for `before_model` and in reverse order for `after_model`/`after_agent`.
