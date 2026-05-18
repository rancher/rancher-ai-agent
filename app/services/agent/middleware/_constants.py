INTERRUPT_CANCEL_MESSAGE = "tool execution cancelled by the user"


class ChildAgentCancelled(Exception):
    """Raised when a child agent's tool execution is cancelled by the user."""

    def __init__(self, agent_name: str, interrupt_info: dict | None = None):
        super().__init__(agent_name)
        self.agent_name = agent_name
        self.interrupt_info = interrupt_info
