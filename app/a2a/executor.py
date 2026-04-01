import uuid
import logging
from typing_extensions import override
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TextPart,
)
from langchain_core.messages import HumanMessage
from ..dependencies import get_llm
from ..services.agent.factory import create_agent

logger = logging.getLogger(__name__)

STREAMABLE_NODES = ("agent", "model")


class RancherAgentExecutor(AgentExecutor):
    """A2A executor that delegates to the same LangGraph agent used by the WebSocket endpoint."""

    def __init__(self, app):
        """
        Args:
            app: The FastAPI application instance, used to access memory_manager and LLM.
        """
        self.app = app
        self._agent = None
        self._agents_metadata = None

    async def _get_or_create_agent(self):
        """Lazily create the LangGraph agent on first request."""
        if self._agent is None:

            llm = get_llm()
            checkpointer = self.app.memory_manager.get_checkpointer()
            self._agent, self._agents_metadata = await create_agent(llm=llm, checkpointer=checkpointer)
            logger.info("A2A agent created with metadata: %s", self._agents_metadata)
        return self._agent

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        try:
            user_input = context.get_user_input()
            task_id = context.task_id or str(uuid.uuid4())
            context_id = context.context_id or str(uuid.uuid4())

            logger.info(
                "A2A execute — task_id=%s, context_id=%s, input=%r",
                task_id,
                context_id,
                user_input,
            )

            agent = await self._get_or_create_agent()

            config = {
                "configurable": {
                    "thread_id": context_id,
                    "user_id": "a2a",
                    "request_id": task_id,
                    "request_metadata": {
                        "agent": "",
                        "user_input": user_input,
                        "context": {},
                        "labels": {},
                        "tags": [],
                    },
                    "agent": "",
                },
            }

            input_data = {
                "messages": [
                    HumanMessage(
                        content=user_input,
                        additional_kwargs={
                            "request_id": task_id,
                            "request_metadata": config["configurable"]["request_metadata"],
                        },
                    )
                ],
            }

            # Notify the client that the agent is working
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.working),
                    final=False,
                )
            )

            artifact_id = str(uuid.uuid4())
            accumulated_text = ""
            chunk_index = 0

            async for stream in agent.astream_events(
                input_data,
                config=config,
                stream_mode=["updates", "messages", "custom", "events"],
            ):
                if stream["event"] == "on_chat_model_stream":
                    text = _extract_streaming_text(stream)
                    if text:
                        accumulated_text += text
                        chunk_index += 1
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                artifact=Artifact(
                                    artifact_id=artifact_id,
                                    parts=[Part(root=TextPart(text=text))],
                                    name="response",
                                ),
                                append=chunk_index > 1,
                                last_chunk=False,
                            )
                        )

            # If we accumulated text, send a final artifact chunk
            if accumulated_text:
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        artifact=Artifact(
                            artifact_id=artifact_id,
                            parts=[Part(root=TextPart(text=""))],
                            name="response",
                        ),
                        append=True,
                        last_chunk=True,
                    )
                )

            # Send the completed status with the full message
            response_text = accumulated_text or "Agent did not produce a response."
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=Message(
                            role=Role.agent,
                            parts=[Part(root=TextPart(text=response_text))],
                            message_id=str(uuid.uuid4()),
                        ),
                    ),
                    final=True,
                )
            )

        except Exception:
            logger.exception(
                "A2A execute failed — task_id=%s, context_id=%s",
                context.task_id,
                context.context_id,
            )
            # Notify the client that the task failed
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id or "",
                    context_id=context.context_id or "",
                    status=TaskStatus(state=TaskState.failed),
                    final=True,
                )
            )
            raise

    @override
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception("cancel not supported")


def _extract_streaming_text(stream: dict) -> str | None:
    """Extract text content from a chat model stream event.

    Only extracts text from 'agent' or 'model' LangGraph nodes to avoid
    streaming intermediate processing steps.
    """
    node = stream.get("metadata", {}).get("langgraph_node")
    if node not in STREAMABLE_NODES:
        return None

    chunk = stream.get("data", {}).get("chunk")
    if not chunk or not chunk.content:
        return None

    content = chunk.content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    if isinstance(content, dict) and "text" in content:
        return content["text"]
    return str(content) if content is not None else ""