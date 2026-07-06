import logging
import ssl
import httpx
import os
import urllib3
from urllib.parse import urlparse
from fastapi import Request
from typing import cast
from kubernetes import client, config

_ssl_context = None
_ssl_context_loaded = False

_rancher_url = None
_rancher_url_loaded = False


def _load_cacerts_ssl_context():
    """Load CA certificate from the Rancher cacerts Setting in the cluster."""
    global _ssl_context, _ssl_context_loaded

    if _ssl_context_loaded:
        return _ssl_context

    try:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        api = client.CustomObjectsApi()
        result = cast(dict, api.get_cluster_custom_object(
            group="management.cattle.io",
            version="v3",
            plural="settings",
            name="internal-cacerts",
        ))
        ca_pem: str = result.get("value", "")
        if ca_pem:
            ctx = ssl.create_default_context()
            ctx.load_verify_locations(cadata=ca_pem)
            _ssl_context = ctx
            logging.info("Loaded CA certificate from internal-cacerts setting")
        else:
            logging.warning("internal-cacerts setting is empty, using default system CAs")
    except Exception as e:
        logging.error("Failed to load internal-cacerts from cluster: %s", e)

    _ssl_context_loaded = True
    return _ssl_context


def _reset_cacerts_cache():
    """Invalidate the cached CA certificate so it will be reloaded on next use."""
    global _ssl_context, _ssl_context_loaded
    _ssl_context = None
    _ssl_context_loaded = False


def _load_rancher_url() -> str | None:
    """Load the Rancher server URL from the internal-server-url Setting in the cluster."""
    global _rancher_url, _rancher_url_loaded

    if _rancher_url_loaded:
        return _rancher_url

    try:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        api = client.CustomObjectsApi()
        result = cast(dict, api.get_cluster_custom_object(
            group="management.cattle.io",
            version="v3",
            plural="settings",
            name="internal-server-url",
        ))
        url: str = result.get("value", "")
        if url:
            _rancher_url = url
            logging.info("Loaded Rancher URL from internal-server-url setting: %s", url)
        else:
            logging.warning("internal-server-url setting is empty")
    except Exception as e:
        logging.error("Failed to load Rancher URL from cluster: %s", e)

    _rancher_url_loaded = True
    return _rancher_url


def _is_tls_error(exc: Exception) -> bool:
    """Return True if the exception is (or wraps) an SSL/TLS error."""
    cause = getattr(exc, "__cause__", None) or exc
    return isinstance(cause, ssl.SSLError)


def _get_tls_verify():
    """Return the appropriate verify parameter for httpx."""
    if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == 'true':
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return False

    ctx = _load_cacerts_ssl_context()
    if ctx is not None:
        return ctx
    return True


async def get_user_id(host: str, token: str) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token.
    """
    url = f"{host}/v3/users?me=true"
    headers = {
        "Cookie": f"R_SESS={token}",
        "Accept": "application/json",
    }
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=5.0, verify=_get_tls_verify()) as http_client:
                resp = await http_client.get(url, headers=headers)
                payload = resp.json()

                if (payload.get("type") == "error") or (resp.status_code != 200) or ("data" not in payload) or (len(payload["data"]) == 0):
                    logging.error("user API returned error: %s - %s", resp.status_code, payload)
                    raise Exception("Failed to retrieve user ID from Rancher API")

                user_id = payload["data"][0]["id"]

                if user_id:
                    logging.info("user API returned: %s - userId %s", resp.status_code, user_id)
                    return user_id
                break
        except httpx.ConnectError as e:
            if attempt == 0 and _is_tls_error(e):
                logging.warning("TLS error connecting to Rancher API, reloading CA certificate and retrying")
                _reset_cacerts_cache()
                continue
            logging.error("user API call failed: %s", e)
            break
        except Exception as e:
            logging.error("user API call failed: %s", e)
            break

    return None

async def get_user_id_from_request(request: Request) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token from the request cookies.
    """
    rancher_url = os.environ.get("RANCHER_URL", "").strip()
    if not rancher_url:
        rancher_url = _load_rancher_url()
    if not rancher_url:
        logging.error("Rancher URL is not configured and could not be fetched from the cluster")
        return None
    token = request.cookies.get("R_SESS")

    if not token:
        logging.warning("R_SESS cookie not found")
        return None

    return await get_user_id(rancher_url, token)