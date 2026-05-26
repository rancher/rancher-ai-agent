"""Tests for app.services.oauth2.client"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from app.services.oauth2.client import OAuthClient
from app.services.oauth2.models import OAuthDiscoveryError


class TestOAuthClientInit:
    def test_basic_init(self):
        client = OAuthClient(client_id="test-id")
        assert client.client_id == "test-id"
        assert client.client_secret == ""
        assert client.scope is None

    def test_init_with_all_params(self):
        client = OAuthClient(client_id="test-id", client_secret="secret", scope="read write")
        assert client.client_id == "test-id"
        assert client.client_secret == "secret"
        assert client.scope == "read write"


class TestGeneratePkcePair:
    def test_returns_verifier_and_challenge(self):
        client = OAuthClient(client_id="test")
        verifier, challenge = client.generate_pkce_pair()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(verifier) > 0
        assert len(challenge) > 0

    def test_verifier_is_url_safe(self):
        client = OAuthClient(client_id="test")
        verifier, _ = client.generate_pkce_pair()
        # url-safe base64 characters only
        assert all(c.isalnum() or c in "-_" for c in verifier)

    def test_challenge_has_no_padding(self):
        client = OAuthClient(client_id="test")
        _, challenge = client.generate_pkce_pair()
        assert "=" not in challenge

    def test_different_each_call(self):
        client = OAuthClient(client_id="test")
        pair1 = client.generate_pkce_pair()
        pair2 = client.generate_pkce_pair()
        assert pair1[0] != pair2[0]
        assert pair1[1] != pair2[1]


class TestGetAuthUrl:
    @pytest.mark.asyncio
    async def test_returns_url_verifier_state(self):
        client = OAuthClient(client_id="test-id", scope="read")
        url, verifier, state = await client.get_auth_url(
            auth_endpoint="https://auth.example.com/authorize",
            redirect_uri="https://app.example.com/callback",
        )
        assert "https://auth.example.com/authorize" in url
        assert "client_id=test-id" in url
        assert "code_challenge_method=S256" in url
        assert isinstance(verifier, str)
        assert isinstance(state, str)


class TestDynamicRegistration:
    @pytest.mark.asyncio
    @patch("app.services.oauth2.client._get_tls_verify", return_value=True)
    async def test_successful_registration(self, _mock_tls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "client_id": "registered-id",
            "client_secret": "registered-secret",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("app.services.oauth2.client.httpx.AsyncClient") as mock_http:
            mock_http_instance = AsyncMock()
            mock_http_instance.post = AsyncMock(return_value=mock_response)
            mock_http_instance.__aenter__ = AsyncMock(return_value=mock_http_instance)
            mock_http_instance.__aexit__ = AsyncMock(return_value=False)
            mock_http.return_value = mock_http_instance

            client = await OAuthClient.from_dynamic_registration(
                registration_endpoint="https://auth.example.com/register",
                redirect_uri="https://app.example.com/callback",
                scope="read",
            )
            assert client.client_id == "registered-id"
            assert client.client_secret == "registered-secret"
            assert client.scope == "read"

    @pytest.mark.asyncio
    @patch("app.services.oauth2.client._get_tls_verify", return_value=True)
    async def test_registration_failure(self, _mock_tls):
        with patch("app.services.oauth2.client.httpx.AsyncClient") as mock_http:
            mock_http_instance = AsyncMock()
            mock_http_instance.post = AsyncMock(side_effect=httpx.RequestError("fail"))
            mock_http_instance.__aenter__ = AsyncMock(return_value=mock_http_instance)
            mock_http_instance.__aexit__ = AsyncMock(return_value=False)
            mock_http.return_value = mock_http_instance

            with pytest.raises(OAuthDiscoveryError, match="Dynamic client registration failed"):
                await OAuthClient.from_dynamic_registration(
                    registration_endpoint="https://auth.example.com/register",
                    redirect_uri="https://app.example.com/callback",
                )


class TestFetchToken:
    @pytest.mark.asyncio
    async def test_fetch_token(self):
        client = OAuthClient(client_id="test-id")
        expected_token = {"access_token": "at-123", "token_type": "bearer"}
        client.client.fetch_token = AsyncMock(return_value=expected_token)

        token = await client.fetch_token(
            token_endpoint="https://auth.example.com/token",
            authorization_response="https://app.example.com/callback?code=abc",
            redirect_uri="https://app.example.com/callback",
            verifier="test-verifier",
        )
        assert token == expected_token


class TestRefreshToken:
    @pytest.mark.asyncio
    async def test_refresh_token(self):
        client = OAuthClient(client_id="test-id")
        expected_token = {"access_token": "new-at", "refresh_token": "new-rt"}
        client.client.refresh_token = AsyncMock(return_value=expected_token)

        token = await client.refresh_token(
            token_endpoint="https://auth.example.com/token",
            refresh_token="old-rt",
        )
        assert token == expected_token
