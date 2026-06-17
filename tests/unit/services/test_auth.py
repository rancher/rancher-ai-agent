import os
import ssl
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.auth import _build_ssl_context, _parse_insecure_skip_tls, get_user_id, get_user_id_from_request


class TestParseInsecureSkipTls:
    def test_true(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "true"}):
            assert _parse_insecure_skip_tls() is True

    def test_false(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "false"}):
            assert _parse_insecure_skip_tls() is False

    def test_unset_defaults_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _parse_insecure_skip_tls() is False

    def test_garbage_raises(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "yes"}):
            with pytest.raises(ValueError, match="must be 'true' or 'false'"):
                _parse_insecure_skip_tls()


class TestBuildSSLContext:
    def test_default_returns_verified_context(self):
        with patch.dict(os.environ, {}, clear=True):
            ctx = _build_ssl_context()
            assert isinstance(ctx, ssl.SSLContext)
            assert ctx.check_hostname is True
            assert ctx.verify_mode == ssl.CERT_REQUIRED
            assert len(ctx.get_ca_certs()) > 0

    def test_insecure_skip_returns_false(self):
        with patch.dict(os.environ, {"INSECURE_SKIP_TLS": "true"}, clear=True):
            assert _build_ssl_context() is False

    def test_ssl_cert_file_loads_custom_ca(self, tmp_path):
        ca = tmp_path / "ca.pem"
        ca.write_text(TEST_CA_PEM)
        with patch.dict(os.environ, {"SSL_CERT_FILE": str(ca)}, clear=True):
            assert isinstance(_build_ssl_context(), ssl.SSLContext)

    def test_ssl_cert_file_missing_raises(self):
        with patch.dict(os.environ, {"SSL_CERT_FILE": "/no/such/file.pem"}, clear=True):
            with pytest.raises(FileNotFoundError, match="does not exist"):
                _build_ssl_context()


class TestGetUserId:
    @pytest.mark.asyncio
    async def test_returns_user_id_on_success(self):
        result = await get_user_id("https://rancher.example.com", "tok")
        assert result == "user-abc123"

    @pytest.mark.asyncio
    async def test_returns_none_on_api_error(self):
        self._mock_response.status_code = 401
        self._mock_response.json.return_value = {"type": "error"}
        result = await get_user_id("https://rancher.example.com", "bad")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self):
        self._mock_client.get.side_effect = Exception("Connection refused")
        result = await get_user_id("https://rancher.example.com", "tok")
        assert result is None

    @pytest.fixture(autouse=True)
    def _patch_httpx(self):
        self._mock_response = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"data": [{"id": "user-abc123"}]}),
        )
        self._mock_client = AsyncMock()
        self._mock_client.get.return_value = self._mock_response
        with patch("app.services.auth.httpx.AsyncClient") as cls:
            cls.return_value.__aenter__ = AsyncMock(return_value=self._mock_client)
            cls.return_value.__aexit__ = AsyncMock(return_value=False)
            yield


class TestGetUserIdFromRequest:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_cookie(self):
        request = MagicMock(cookies={})
        assert await get_user_id_from_request(request) is None

    @pytest.mark.asyncio
    async def test_uses_rancher_url_env(self):
        request = MagicMock(cookies={"R_SESS": "tok"})
        with patch.dict(os.environ, {"RANCHER_URL": "https://rancher.prod.example.com"}), \
             patch("app.services.auth.get_user_id", new_callable=AsyncMock, return_value="u1") as mock_get:
            await get_user_id_from_request(request)
            mock_get.assert_called_once_with("https://rancher.prod.example.com", "tok")

    @pytest.mark.asyncio
    async def test_normalizes_url_without_scheme(self):
        request = MagicMock(cookies={"R_SESS": "tok"})
        with patch.dict(os.environ, {"RANCHER_URL": "rancher.example.com"}), \
             patch("app.services.auth.get_user_id", new_callable=AsyncMock, return_value="u1") as mock_get:
            await get_user_id_from_request(request)
            mock_get.assert_called_once_with("https://rancher.example.com", "tok")

    @pytest.mark.asyncio
    async def test_falls_back_to_internal_svc(self):
        request = MagicMock(cookies={"R_SESS": "tok"})
        with patch.dict(os.environ, {}, clear=True), \
             patch("app.services.auth.get_user_id", new_callable=AsyncMock, return_value="u1") as mock_get:
            await get_user_id_from_request(request)
            mock_get.assert_called_once_with("https://rancher.cattle-system.svc", "tok")


# openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
#   -keyout /dev/null -out - -days 3650 -nodes -subj '/CN=test-ca'
TEST_CA_PEM = """\
-----BEGIN CERTIFICATE-----
MIIBeTCCAR+gAwIBAgIUS6up8UzE6XhsSSn6DsLbYcuwCF8wCgYIKoZIzj0EAwIw
EjEQMA4GA1UEAwwHdGVzdC1jYTAeFw0yNjA1MjkxMjE0MzNaFw0yNjA1MzAxMjE0
MzNaMBIxEDAOBgNVBAMMB3Rlc3QtY2EwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNC
AASG176iTWW2SuUOtZnloMy1+xfNH/ZYOsjsc69XgU8OPK9yG1K8sIsx+KfOc9Rv
cEL9YP6RljfHn4Y9vOsHJZhDo1MwUTAdBgNVHQ4EFgQUpUAe+TFKeo2qxMELe53R
ZouQ6R0wHwYDVR0jBBgwFoAUpUAe+TFKeo2qxMELe53RZouQ6R0wDwYDVR0TAQH/
BAUwAwEB/zAKBggqhkjOPQQDAgNIADBFAiA9yYtS/Os2ETOt2F0c834l8WvTUMCo
XIuSPIOGvZDpEgIhAJ5+iN+UoIyC7VtHJZL6WT/NVmXKgO08e6aOGfiXdPuX
-----END CERTIFICATE-----"""
