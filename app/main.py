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

# This will be removed once https://github.com/modelcontextprotocol/python-sdk/pull/1177 is merged
class SimpleTruststore:
    def get_default(self):
        """Get the default Python truststore"""
        return certifi.where()

    def create_combined(self, company_cert_path, output_path):
        """Create truststore with public CAs + company cert"""
        with open(output_path, "w") as combined:
            # Add public CAs 
            with open(certifi.where(), "r") as public_cas:
                combined.write(public_cas.read())

            # Add MCP self-signed cert
            with open(company_cert_path, "r") as company:
                combined.write("\n" + company.read())

        return output_path
    
    def use_truststore(self, truststore_path):
        """Set the global truststore"""
        os.environ["SSL_CERT_FILE"] = truststore_path

    def set_truststore(self):
        company_cert_path = "/etc/tls/tls.crt"
        output_path = "/cert/combined.crt"

        if os.path.exists(company_cert_path):
            truststore_path = self.create_combined(
                company_cert_path=company_cert_path, output_path=output_path
            )
            self.use_truststore(truststore_path=truststore_path)
        else:
            logging.warning(f"Company cert not found at {company_cert_path}, skipping truststore setup.")

        
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
        logging.getLogger().setLevel(LOG_LEVEL)
        if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() != "true":
            SimpleTruststore().set_truststore()

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