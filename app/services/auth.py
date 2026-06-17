import logging
import os
import ssl
from urllib.parse import urlparse
import httpx
from fastapi import Request

SA_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"


def _parse_insecure_skip_tls() -> bool:
    raw = os.environ.get("INSECURE_SKIP_TLS", "false").strip().lower()
    if raw in ("true", "false"):
        return raw == "true"
    raise ValueError(f"INSECURE_SKIP_TLS must be 'true' or 'false', got: {raw!r}")


def _build_ssl_context() -> ssl.SSLContext | bool:
    if _parse_insecure_skip_tls():
        logging.warning("TLS verification disabled by INSECURE_SKIP_TLS")
        return False

    ctx = ssl.create_default_context()

    ssl_cert_file = os.environ.get("SSL_CERT_FILE")
    if ssl_cert_file:
        if not os.path.isfile(ssl_cert_file):
            raise FileNotFoundError(
                f"SSL_CERT_FILE is set but file does not exist: {ssl_cert_file}"
            )
        ctx.load_verify_locations(ssl_cert_file)

    if os.path.isfile(SA_CA_PATH):
        ctx.load_verify_locations(SA_CA_PATH)

    return ctx


async def get_user_id(host: str, token: str) -> str | None:
    """
    Retrieves the user ID from the Rancher API using the session token.
    """
    url = f"{host}/v3/users?me=true"

    try:
        verify = _build_ssl_context()
    except Exception as e:
        logging.error("TLS configuration error: %s", e)
        return None

    try:
        async with httpx.AsyncClient(timeout=5.0, verify=verify) as client:
            resp = await client.get(url, headers={
                "Cookie": f"R_SESS={token}",
                "Accept": "application/json",
            })
            payload = resp.json()

            if (
                payload.get("type") == "error"
                or resp.status_code != 200
                or "data" not in payload
                or len(payload["data"]) == 0
            ):
                logging.error("user API returned error: %s - %s", resp.status_code, payload)
                raise Exception("Failed to retrieve user ID from Rancher API")

            user_id = payload["data"][0]["id"]
            if user_id:
                logging.info("user API returned: %s - userId %s", resp.status_code, user_id)
                return user_id

    except Exception as e:
        logging.error("user API call failed: %s", e)

    return None


async def get_user_id_from_request(request: Request) -> str | None:
    """
    Retrieves the user ID from the Rancher API using the session token from the request cookies.
    """
    token = request.cookies.get("R_SESS")
    if not token:
        return None

    rancher_url = os.environ.get("RANCHER_URL", "")
    if rancher_url:
        parsed = urlparse(rancher_url)
        scheme = parsed.scheme or "https"
        host = f"{scheme}://{parsed.netloc or rancher_url}"
    else:
        host = "https://rancher.cattle-system.svc"

    return await get_user_id(host, token)
