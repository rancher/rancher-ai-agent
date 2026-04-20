import re
import logging
import os
import certifi

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from .services.agent.loader import ensure_default_ai_agent_config_crds
from .services.memory import create_memory_manager
from .routers import agent, configuration, chat, websocket, ui
from .controllers.ai_agent_config import create_kopf_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class _NoisyEndpointFilter(logging.Filter):
    """Suppress uvicorn access log entries for noisy endpoints (probes, polling, etc.)."""
    _NOISY_PATHS = ("/v1/api/health", "/v1/api/readiness", "/v1/api/llm/bedrock/models")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._NOISY_PATHS)


class _SensitiveHeaderFilter(logging.Filter):
    """Redact sensitive HTTP headers (e.g. Authorization, X-Api-Key) from log messages.

    Attached to root logger handlers so it intercepts records propagated from
    any child logger (e.g. botocore.endpoint) that may emit raw HTTP requests
    containing bearer tokens or API keys at DEBUG level.
    """
    _HEADER_PATTERNS = [
        re.compile(
            r"""('Authorization'\s*:\s*b?['"])(.*?)(['"])""",
            re.IGNORECASE,
        ),
        re.compile(
            r"""('X-Api-Key'\s*:\s*b?['"])(.*?)(['"])""",
            re.IGNORECASE,
        ),
        re.compile(
            r"""(Authorization:\s*)(Bearer\s+)?\S+""",
            re.IGNORECASE,
        ),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.msg = record.getMessage()
            record.args = None
        msg = record.msg
        if isinstance(msg, str) and ("authorization" in msg.lower() or "x-api-key" in msg.lower()):
            for pattern in self._HEADER_PATTERNS:
                msg = pattern.sub(r"\1[REDACTED]\3" if pattern.groups >= 3 else r"\1[REDACTED]", msg)
            record.msg = msg
        return True

_sensitive_filter = _SensitiveHeaderFilter()
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_sensitive_filter)
logging.getLogger("uvicorn.access").addFilter(_NoisyEndpointFilter())
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
        logging.getLogger().setLevel(LOG_LEVEL)

        configs = ensure_default_ai_agent_config_crds()
        logging.info(f"Startup: {len(configs)} AIAgentConfig CRDs in the cluster.")

        app.memory_manager = await create_memory_manager()
        app.kopf_manager = create_kopf_manager()
        app.kopf_manager.start()

        app.state.ready = True

    except ValueError as e:
        app.state.ready = False
        logging.critical(e)
        raise e
    
    yield

    app.kopf_manager.stop()
    await app.memory_manager.destroy()
    
app = FastAPI(lifespan=lifespan)

app.include_router(websocket.router)
app.include_router(agent.router)
app.include_router(configuration.router)
app.include_router(chat.router)

if os.environ.get("ENABLE_TEST_UI", "").lower() == "true":
    app.include_router(ui.router)