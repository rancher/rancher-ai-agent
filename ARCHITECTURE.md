# Architecture

The Rancher AI Agent uses a [subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents) architecture built on LangGraph. A single **supervisor agent** receives user requests and delegates work to one or more specialized **child agents** (loaded from AIAgentConfig CRDs). Each child agent is wrapped as a LangChain tool so the supervisor's LLM decides which agent(s) to invoke, coordinates their results, and synthesizes a final response.

Both supervisor and child agents are constructed with the same `create_agent` factory and share a middleware-based extension model for cross-cutting concerns.

---

## Middleware

Middleware intercepts the agent execution lifecycle at specific points — before/after the LLM call, after the full agent turn, or around individual tool calls. They are passed to `create_agent()` via the `middleware=` parameter and execute in order.

This project uses [LangChain's agent middleware system](https://python.langchain.com/docs/how_to/agent_middleware/) (`langchain.agents.middleware`).

### Middleware Types

There are two ways to define middleware: **decorator-based** (factory functions) and **class-based**.

#### Decorator-Based (Factory Functions)

Use decorators from `langchain.agents.middleware` to create middleware as factory functions. Each decorator corresponds to a lifecycle hook:

| Decorator | When it runs | Use case |
|-----------|-------------|----------|
| `@before_model` | Before the LLM is called | Inject system messages, short-circuit the LLM call |
| `@after_model` | After the LLM responds | Enrich or modify the AIMessage |
| `@after_agent` | After the full agent turn (model + tools) completes | Post-processing, dispatching events |
| `@wrap_tool_call` | Wraps each individual tool execution | Validation gates, error handling, artifact processing |

#### Class-Based

Subclass `AgentMiddleware` when your middleware needs custom state fields or multiple hooks in one unit.

### Adding a New Middleware

1. Create a new file in `app/services/agent/middleware/`.
2. Implement the middleware using the appropriate hook type. Import shared constants from `._constants` if needed.
3. Export the middleware from `__init__.py`.
4. Register the middleware in the agent's middleware list (`supervisor.py` or `child.py`) inside the `create_agent()` call.

---

## Human Validation (Human-in-the-Loop)

Human validation is the mechanism that pauses tool execution to ask the user for confirmation before performing a sensitive action (e.g. creating or modifying a Kubernetes resource).

### Child Agent Side

The `child_human_validation_middleware` (a `@wrap_tool_call` middleware in `app/services/agent/middleware/human_validation.py`) handles the gate at the child level:

1. Before executing a tool call, it checks if the tool is listed in the agent's `human_validation_tools` configuration.
2. If it is, the middleware invokes the corresponding **planning tool** (`<tool_name>Plan`) to produce a preview of the intended changes (e.g. a diff or resource manifest).
3. It then calls `langgraph.types.interrupt()` with the plan response, pausing the child graph and surfacing the confirmation prompt to the client.
4. When the graph is resumed:
   - If the user responds `"yes"`, the middleware proceeds to execute the real tool and dispatches any associated UI tools (e.g. YAML diff viewers).
   - Otherwise, it returns a cancellation `ToolMessage` (`INTERRUPT_CANCEL_MESSAGE`) and the tool call is skipped.

### Supervisor Side

Because the child agent is invoked via `ainvoke()` (not as a subgraph), a `GraphInterrupt` raised inside it is suppressed and `ainvoke()` returns normally. The supervisor handles this with a two-phase interrupt relay:

1. **Detecting a pending interrupt** — After invoking the child, the supervisor checks `aget_state()` on the child graph. If there are pending interrupts, it re-raises the interrupt at the supervisor level via `langgraph.types.interrupt(child_state.interrupts[0].value)`. This surfaces the confirmation prompt to the client through the supervisor's own graph.

2. **Resuming after user response** — On the next invocation, the supervisor's agent tool (`_invoke`) detects that the child has a pending interrupt (via `aget_state()`). It calls `langgraph.types.interrupt()` at the supervisor level to receive the user's `Command(resume=...)` value, then forwards it to the child graph with `ainvoke(Command(resume=resume_value))`.

3. **Handling cancellation** — If the resumed child returns an `INTERRUPT_CANCEL_MESSAGE`, the supervisor propagates it so that its own `cancel_human_validation_middleware` can end the graph gracefully.

### Cancellation via `cancel_human_validation_middleware`

The `cancel_human_validation_middleware` (`app/services/agent/middleware/cancel_check.py`) is a `@before_model` middleware registered on both the supervisor and child agents. After a human validation interrupt is resumed with a rejection (anything other than `"yes"`), the child returns a `ToolMessage` with `INTERRUPT_CANCEL_MESSAGE` as its content. On the next LLM turn, this middleware detects that the last message is a cancelled tool message and short-circuits the agent — it jumps directly to the `"end"` node with a cancellation reply, preventing any further LLM calls or tool executions.

This relay pattern allows human-in-the-loop confirmations to originate deep inside a child agent while the client interacts exclusively with the supervisor's interrupt surface.
