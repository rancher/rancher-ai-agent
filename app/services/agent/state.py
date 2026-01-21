"""Types for the chat agent state."""
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Sequence, TypedDict, Annotated

def add_messages_with_timestamp(left: Sequence[BaseMessage], right: Sequence[BaseMessage]):
    """Returns the result of add_messages after adding timestamps to messages."""
    new_messages = right if isinstance(right, list) else [right]
    
    # Add timestamps to new messages if not already present
    created_at = datetime.now().isoformat()

    for msg in new_messages:
        if hasattr(msg, 'additional_kwargs'):
            if "created_at" not in msg.additional_kwargs:
                msg.additional_kwargs["created_at"] = created_at

        elif isinstance(msg, dict):
            if "additional_kwargs" not in msg:
                msg["additional_kwargs"] = {}
            if "created_at" not in msg["additional_kwargs"]:
                msg["additional_kwargs"]["created_at"] = created_at

    return add_messages(left, new_messages)

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages_with_timestamp]
    summary: str