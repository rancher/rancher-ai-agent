"""Tests for app.services.auth"""

import ssl
import os
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

import app.services.auth as auth_module
from app.services.auth import (
    _load_cacerts_ssl_context,
    _reset_cacerts_cache,
    _load_rancher_url,
    _is_tls_error,
    _get_tls_verify,
    get_user_id,
    get_user_id_from_token,
)

SAMPLE_CA_PEM = """\
-----BEGIN CERTIFICATE-----
MIIBvjCCAWOgAwIBAgIBADAKBggqhkjOPQQDAjBGMRwwGgYDVQQKExNkeW5hbWlj
bGlzdGVuZXItb3JnMSYwJAYDVQQDDB1keW5hbWljbGlzdGVuZXItY2FAMTc4MzA2
NTg1ODAeFw0yNjA3MDMwNzA0MThaFw0zNjA2MzAwNzA0MThaMEYxHDAaBgNVBAoT
E2R5bmFtaWNsaXN0ZW5lci1vcmcxJjAkBgNVBAMMHWR5bmFtaWNsaXN0ZW5lci1j
YUAxNzgzMDY1ODU4MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE3A4JBnBPtksq
dBGgtagh7eRYQhDTNacQw20+GzMSyMfZEyEq5hlatQRAnvClGfDMiqlEOwNjCFBJ
Inyq01wOsqNCMEAwDgYDVR0PAQH/BAQDAgKkMA8GA1UdEwEB/wQFMAMBAf8wHQYD
VR0OBBYEFDr8qyaBeKs/CLAo7iu3o/RGW9miMAoGCCqGSM49BAMCA0kAMEYCIQDB
u6+Vx8Ec1X7I2HopsHi09rNBjqG6WiLrhYmsbijKcAIhALu5DUkom6TgMTRYVNeK
p36pHH4Sra16Ld8jkPw+AxUI
-----END CERTIFICATE-----
"""


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset module-level caches before and after every test."""
    _reset_cacerts_cache()
    auth_module._rancher_url = None
    auth_module._rancher_url_loaded = False
    yield
    _reset_cacerts_cache()
    auth_module._rancher_url = None
    auth_module._rancher_url_loaded = False


# ---------------------------------------------------------------------------
# _is_tls_error
# ---------------------------------------------------------------------------

class TestIsTlsError:
    def test_returns_true_for_ssl_error_cause(self):
        ssl_err = ssl.SSLError("certificate verify failed")
        exc = httpx.ConnectError("TLS handshake failed")
        exc.__cause__ = ssl_err
        assert _is_tls_error(exc) is True

    def test_returns_true_for_ssl_error_directly(self):
        exc = ssl.SSLError("certificate verify failed")
        assert _is_tls_error(exc) is True

    def test_returns_false_for_non_tls_error(self):
        exc = ConnectionRefusedError("connection refused")
        assert _is_tls_error(exc) is False

    def test_returns_false_when_cause_is_not_ssl(self):
        inner = OSError("network error")
        exc = httpx.ConnectError("connect failed")
        exc.__cause__ = inner
        assert _is_tls_error(exc) is False


# ---------------------------------------------------------------------------
# _reset_cacerts_cache
# ---------------------------------------------------------------------------

class TestResetCacertsCache:
    def test_resets_ssl_context_and_loaded_flag(self):
        auth_module._ssl_context = MagicMock()
        auth_module._ssl_context_loaded = True
        _reset_cacerts_cache()
        assert auth_module._ssl_context is None
        assert auth_module._ssl_context_loaded is False


# ---------------------------------------------------------------------------
# _load_cacerts_ssl_context
# ---------------------------------------------------------------------------

class TestLoadCacertsSslContext:
    def _make_k8s_mock(self, ca_pem: str):
        mock_api = MagicMock()
        mock_api.get_cluster_custom_object.return_value = {"value": ca_pem}
        mock_client = MagicMock()
        mock_client.CustomObjectsApi.return_value = mock_api
        return mock_client

    def test_returns_cached_context_without_k8s_call(self):
        fake_ctx = MagicMock(spec=ssl.SSLContext)
        auth_module._ssl_context = fake_ctx
        auth_module._ssl_context_loaded = True
        with patch("app.services.auth.client") as mock_client:
            result = _load_cacerts_ssl_context()
        assert result is fake_ctx
        mock_client.CustomObjectsApi.assert_not_called()

    def test_loads_ca_from_cluster_and_caches(self):
        mock_ssl_ctx = MagicMock(spec=ssl.SSLContext)
        mock_client = self._make_k8s_mock(SAMPLE_CA_PEM)
        with patch("app.services.auth.config") as mock_config, \
             patch("app.services.auth.client", mock_client), \
             patch("app.services.auth.ssl.create_default_context", return_value=mock_ssl_ctx):
            result = _load_cacerts_ssl_context()
        mock_config.load_incluster_config.assert_called_once()
        mock_ssl_ctx.load_verify_locations.assert_called_once_with(cadata=SAMPLE_CA_PEM)
        assert result is mock_ssl_ctx
        assert auth_module._ssl_context is mock_ssl_ctx
        assert auth_module._ssl_context_loaded is True

    def test_falls_back_to_kube_config_on_config_exception(self):
        from kubernetes.config import ConfigException
        mock_ssl_ctx = MagicMock(spec=ssl.SSLContext)
        mock_client = self._make_k8s_mock(SAMPLE_CA_PEM)
        with patch("app.services.auth.config") as mock_config, \
             patch("app.services.auth.client", mock_client), \
             patch("app.services.auth.ssl.create_default_context", return_value=mock_ssl_ctx):
            mock_config.ConfigException = ConfigException
            mock_config.load_incluster_config.side_effect = ConfigException("not in cluster")
            result = _load_cacerts_ssl_context()
        mock_config.load_kube_config.assert_called_once()
        assert result is mock_ssl_ctx

    def test_returns_none_when_cacerts_value_is_empty(self):
        mock_client = MagicMock()
        mock_client.CustomObjectsApi.return_value.get_cluster_custom_object.return_value = {"value": ""}
        with patch("app.services.auth.config"), \
             patch("app.services.auth.client", mock_client):
            result = _load_cacerts_ssl_context()
        assert result is None
        assert auth_module._ssl_context_loaded is True

    def test_returns_none_and_sets_loaded_on_k8s_error(self):
        mock_client = MagicMock()
        mock_client.CustomObjectsApi.return_value.get_cluster_custom_object.side_effect = Exception("k8s unavailable")
        with patch("app.services.auth.config"), \
             patch("app.services.auth.client", mock_client):
            result = _load_cacerts_ssl_context()
        assert result is None
        assert auth_module._ssl_context_loaded is True

    def test_does_not_call_k8s_twice(self):
        mock_client = self._make_k8s_mock(SAMPLE_CA_PEM)
        mock_ssl_ctx = MagicMock(spec=ssl.SSLContext)
        with patch("app.services.auth.config"), \
             patch("app.services.auth.client", mock_client), \
             patch("app.services.auth.ssl.create_default_context", return_value=mock_ssl_ctx):
            _load_cacerts_ssl_context()
            _load_cacerts_ssl_context()
        assert mock_client.CustomObjectsApi.call_count == 1


# ---------------------------------------------------------------------------
# _load_rancher_url
# ---------------------------------------------------------------------------

class TestLoadRancherUrl:
    def _make_k8s_mock(self, url: str):
        mock_api = MagicMock()
        mock_api.get_cluster_custom_object.return_value = {"value": url}
        mock_client = MagicMock()
        mock_client.CustomObjectsApi.return_value = mock_api
        return mock_client

    def test_returns_cached_url_without_k8s_call(self):
        auth_module._rancher_url = "https://cached.example.com"
        auth_module._rancher_url_loaded = True
        with patch("app.services.auth.client") as mock_client:
            result = _load_rancher_url()
        assert result == "https://cached.example.com"
        mock_client.CustomObjectsApi.assert_not_called()

    def test_loads_url_from_cluster_and_caches(self):
        mock_client = self._make_k8s_mock("https://10.43.31.188")
        with patch("app.services.auth.config") as mock_config, \
             patch("app.services.auth.client", mock_client):
            result = _load_rancher_url()
        mock_config.load_incluster_config.assert_called_once()
        mock_client.CustomObjectsApi.return_value.get_cluster_custom_object.assert_called_once_with(
            group="management.cattle.io",
            version="v3",
            plural="settings",
            name="internal-server-url",
        )
        assert result == "https://10.43.31.188"
        assert auth_module._rancher_url == "https://10.43.31.188"
        assert auth_module._rancher_url_loaded is True

    def test_falls_back_to_kube_config_on_config_exception(self):
        from kubernetes.config import ConfigException
        mock_client = self._make_k8s_mock("https://10.43.31.188")
        with patch("app.services.auth.config") as mock_config, \
             patch("app.services.auth.client", mock_client):
            mock_config.ConfigException = ConfigException
            mock_config.load_incluster_config.side_effect = ConfigException("not in cluster")
            result = _load_rancher_url()
        mock_config.load_kube_config.assert_called_once()
        assert result == "https://10.43.31.188"

    def test_returns_none_when_value_is_empty(self):
        mock_client = MagicMock()
        mock_client.CustomObjectsApi.return_value.get_cluster_custom_object.return_value = {"value": ""}
        with patch("app.services.auth.config"), \
             patch("app.services.auth.client", mock_client):
            result = _load_rancher_url()
        assert result is None
        assert auth_module._rancher_url_loaded is True

    def test_returns_none_and_sets_loaded_on_k8s_error(self):
        mock_client = MagicMock()
        mock_client.CustomObjectsApi.return_value.get_cluster_custom_object.side_effect = Exception("k8s unavailable")
        with patch("app.services.auth.config"), \
             patch("app.services.auth.client", mock_client):
            result = _load_rancher_url()
        assert result is None
        assert auth_module._rancher_url_loaded is True

    def test_does_not_call_k8s_twice(self):
        mock_client = self._make_k8s_mock("https://10.43.31.188")
        with patch("app.services.auth.config"), \
             patch("app.services.auth.client", mock_client):
            _load_rancher_url()
            _load_rancher_url()
        assert mock_client.CustomObjectsApi.call_count == 1


# ---------------------------------------------------------------------------
# _get_tls_verify
# ---------------------------------------------------------------------------

class TestGetTlsVerify:
    def test_returns_false_when_insecure_skip_tls_true(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "true"}):
            result = _get_tls_verify()
        assert result is False

    def test_returns_false_when_insecure_skip_tls_true_uppercase(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "TRUE"}):
            result = _get_tls_verify()
        assert result is False

    def test_returns_ssl_context_when_cacerts_loaded(self):
        fake_ctx = MagicMock(spec=ssl.SSLContext)
        with patch("app.services.auth._load_cacerts_ssl_context", return_value=fake_ctx), \
             patch.dict(os.environ, {"INSECURE_SKIP_TLS": "false"}):
            result = _get_tls_verify()
        assert result is fake_ctx

    def test_returns_true_when_no_cacerts_and_tls_enabled(self):
        with patch("app.services.auth._load_cacerts_ssl_context", return_value=None), \
             patch.dict(os.environ, {"INSECURE_SKIP_TLS": "false"}):
            result = _get_tls_verify()
        assert result is True


# ---------------------------------------------------------------------------
# get_user_id
# ---------------------------------------------------------------------------

def _make_response(status_code: int, payload: dict):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = payload
    return mock_resp


class TestGetUserId:
    def _patch_client(self, response):
        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=response)
        mock_async_ctx = AsyncMock()
        mock_async_ctx.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_async_ctx.__aexit__ = AsyncMock(return_value=False)
        return mock_async_ctx, mock_http_client

    @pytest.mark.asyncio
    async def test_returns_user_id_on_success(self):
        response = _make_response(200, {"data": [{"id": "user-123"}]})
        mock_ctx, _ = self._patch_client(response)
        with patch("app.services.auth.httpx.AsyncClient", return_value=mock_ctx), \
             patch("app.services.auth._get_tls_verify", return_value=True):
            result = await get_user_id("https://rancher.example.com", "token-abc")
        assert result == "user-123"

    @pytest.mark.asyncio
    async def test_returns_none_on_api_error_response(self):
        response = _make_response(200, {"type": "error", "message": "unauthorized"})
        mock_ctx, _ = self._patch_client(response)
        with patch("app.services.auth.httpx.AsyncClient", return_value=mock_ctx), \
             patch("app.services.auth._get_tls_verify", return_value=True):
            result = await get_user_id("https://rancher.example.com", "token-abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_non_200_status(self):
        response = _make_response(401, {})
        mock_ctx, _ = self._patch_client(response)
        with patch("app.services.auth.httpx.AsyncClient", return_value=mock_ctx), \
             patch("app.services.auth._get_tls_verify", return_value=True):
            result = await get_user_id("https://rancher.example.com", "bad-token")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_data(self):
        response = _make_response(200, {"data": []})
        mock_ctx, _ = self._patch_client(response)
        with patch("app.services.auth.httpx.AsyncClient", return_value=mock_ctx), \
             patch("app.services.auth._get_tls_verify", return_value=True):
            result = await get_user_id("https://rancher.example.com", "token-abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_generic_exception(self):
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=Exception("network failure"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("app.services.auth.httpx.AsyncClient", return_value=mock_ctx), \
             patch("app.services.auth._get_tls_verify", return_value=True):
            result = await get_user_id("https://rancher.example.com", "token-abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_retries_on_tls_error_and_succeeds(self):
        """On TLS ConnectError, cache is reset and the second attempt succeeds."""
        ssl_err = ssl.SSLError("certificate verify failed")
        tls_connect_err = httpx.ConnectError("TLS handshake failed")
        tls_connect_err.__cause__ = ssl_err

        success_response = _make_response(200, {"data": [{"id": "user-456"}]})
        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=tls_connect_err)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)

        success_http_client = AsyncMock()
        success_http_client.get = AsyncMock(return_value=success_response)
        success_ctx = AsyncMock()
        success_ctx.__aenter__ = AsyncMock(return_value=success_http_client)
        success_ctx.__aexit__ = AsyncMock(return_value=False)

        side_effects = [fail_ctx, success_ctx]
        with patch("app.services.auth.httpx.AsyncClient", side_effect=side_effects), \
             patch("app.services.auth._get_tls_verify", return_value=True), \
             patch("app.services.auth._reset_cacerts_cache") as mock_reset:
            result = await get_user_id("https://rancher.example.com", "token-abc")

        assert result == "user-456"
        mock_reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_retry_on_non_tls_connect_error(self):
        """A ConnectError that is NOT a TLS error should not trigger a retry."""
        connect_err = httpx.ConnectError("connection refused")
        connect_err.__cause__ = ConnectionRefusedError("connection refused")

        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=connect_err)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.auth.httpx.AsyncClient", return_value=fail_ctx) as mock_client, \
             patch("app.services.auth._get_tls_verify", return_value=True), \
             patch("app.services.auth._reset_cacerts_cache") as mock_reset:
            result = await get_user_id("https://rancher.example.com", "token-abc")

        assert result is None
        mock_reset.assert_not_called()
        assert mock_client.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_none_when_tls_retry_also_fails(self):
        """If the retry after CA reload also raises TLS error, return None."""
        ssl_err = ssl.SSLError("certificate verify failed")

        def make_tls_ctx():
            err = httpx.ConnectError("TLS handshake failed")
            err.__cause__ = ssl_err
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(side_effect=err)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        with patch("app.services.auth.httpx.AsyncClient", side_effect=[make_tls_ctx(), make_tls_ctx()]), \
             patch("app.services.auth._get_tls_verify", return_value=True), \
             patch("app.services.auth._reset_cacerts_cache") as mock_reset:
            result = await get_user_id("https://rancher.example.com", "token-abc")

        assert result is None
        mock_reset.assert_called_once()


# ---------------------------------------------------------------------------
# get_user_id_from_token
# ---------------------------------------------------------------------------

class TestGetUserIdFromToken:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            result = await get_user_id_from_token({})
        assert result is None

    @pytest.mark.asyncio
    async def test_delegates_to_get_user_id(self):
        with patch.dict(os.environ, {"RANCHER_URL": "https://rancher.example.com"}), \
             patch("app.services.auth.get_user_id", new=AsyncMock(return_value="user-789")) as mock_get:
            result = await get_user_id_from_token({"R_SESS": "my-session-token"})
        mock_get.assert_called_once_with("https://rancher.example.com", "my-session-token")
        assert result == "user-789"

    @pytest.mark.asyncio
    async def test_api_token_env_takes_precedence_over_cookie(self):
        with patch.dict(os.environ, {"RANCHER_URL": "https://rancher.example.com", "RANCHER_API_TOKEN": "env-token"}), \
             patch("app.services.auth.get_user_id", new=AsyncMock(return_value="user-789")) as mock_get:
            await get_user_id_from_token({"R_SESS": "cookie-token"})
        mock_get.assert_called_once_with("https://rancher.example.com", "env-token")

    @pytest.mark.asyncio
    async def test_returns_none_when_cluster_lookup_fails(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("app.services.auth._load_rancher_url", return_value=None), \
             patch("app.services.auth.get_user_id", new=AsyncMock()) as mock_get:
            result = await get_user_id_from_token({"R_SESS": "token"})
        assert result is None
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_rancher_url_from_cluster_when_env_not_set(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("app.services.auth._load_rancher_url", return_value="https://10.43.31.188"), \
             patch("app.services.auth.get_user_id", new=AsyncMock(return_value="user-001")) as mock_get:
            await get_user_id_from_token({"R_SESS": "token"})
        mock_get.assert_called_once_with("https://10.43.31.188", "token")

    @pytest.mark.asyncio
    async def test_env_var_takes_precedence_over_cluster_url(self):
        with patch.dict(os.environ, {"RANCHER_URL": "https://rancher.example.com"}), \
             patch("app.services.auth._load_rancher_url") as mock_load, \
             patch("app.services.auth.get_user_id", new=AsyncMock(return_value="user-002")) as mock_get:
            await get_user_id_from_token({"R_SESS": "token"})
        mock_load.assert_not_called()
        mock_get.assert_called_once_with("https://rancher.example.com", "token")
