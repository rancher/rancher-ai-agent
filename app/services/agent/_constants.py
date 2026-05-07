from .loader import AgentConfig


INTERRUPT_CANCEL_MESSAGE = "tool execution cancelled by the user"

class NoAgentAvailableError(Exception):
    """Exception raised when loading MCP tools fails."""
    pass

class NeedsOauth2(Exception):
    """Exception raised when OAuth2 authentication is required."""
    agent_cfg: AgentConfig
    
    def __init__(self, agent_cfg: AgentConfig):
        super().__init__(agent_cfg)
        self.agent_cfg = agent_cfg
