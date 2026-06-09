"""Tests for app.services.oauth2.models"""

from app.services.oauth2.models import (
    OAuthDiscoveryError,
    ResourceMetadata,
    AuthorizationServerMetadata,
    OAuthDiscoveryResult,
    OAuthClientCredentials,
)


def test_oauth_discovery_error_is_exception():
    err = OAuthDiscoveryError("test error")
    assert isinstance(err, Exception)
    assert str(err) == "test error"


def test_resource_metadata_defaults():
    rm = ResourceMetadata()
    assert rm.resource == ""
    assert rm.authorization_servers == []
    assert rm.scopes_supported == []
    assert rm.bearer_methods_supported == []


def test_resource_metadata_with_values():
    rm = ResourceMetadata(
        resource="https://mcp.example.com",
        authorization_servers=["https://auth.example.com"],
        scopes_supported=["read", "write"],
        bearer_methods_supported=["header"],
    )
    assert rm.resource == "https://mcp.example.com"
    assert rm.authorization_servers == ["https://auth.example.com"]
    assert rm.scopes_supported == ["read", "write"]
    assert rm.bearer_methods_supported == ["header"]


def test_authorization_server_metadata_defaults():
    asm = AuthorizationServerMetadata()
    assert asm.issuer == ""
    assert asm.authorization_endpoint == ""
    assert asm.token_endpoint == ""
    assert asm.registration_endpoint is None
    assert asm.scopes_supported == []
    assert asm.response_types_supported == []
    assert asm.code_challenge_methods_supported == []


def test_authorization_server_metadata_with_values():
    asm = AuthorizationServerMetadata(
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        registration_endpoint="https://auth.example.com/register",
        scopes_supported=["openid"],
        response_types_supported=["code"],
        code_challenge_methods_supported=["S256"],
    )
    assert asm.issuer == "https://auth.example.com"
    assert asm.registration_endpoint == "https://auth.example.com/register"


def test_oauth_discovery_result_required_fields():
    asm = AuthorizationServerMetadata(
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
    )
    result = OAuthDiscoveryResult(auth_server_metadata=asm)
    assert result.auth_server_metadata.authorization_endpoint == "https://auth.example.com/authorize"
    assert result.auth_server_metadata.token_endpoint == "https://auth.example.com/token"
    assert result.required_scopes == []
    assert result.resource_metadata is None


def test_oauth_client_credentials_defaults():
    creds = OAuthClientCredentials(client_id="my-client")
    assert creds.client_id == "my-client"
    assert creds.client_secret == ""
    assert creds.scopes == ""


def test_oauth_client_credentials_with_all_fields():
    creds = OAuthClientCredentials(
        client_id="my-client",
        client_secret="my-secret",
        scopes="read write",
    )
    assert creds.client_id == "my-client"
    assert creds.client_secret == "my-secret"
    assert creds.scopes == "read write"


def test_oauth_discovery_result_round_trip():
    original = OAuthDiscoveryResult(
        auth_server_metadata=AuthorizationServerMetadata(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            registration_endpoint="https://auth.example.com/register",
            scopes_supported=["read"],
            response_types_supported=["code"],
            code_challenge_methods_supported=["S256"],
        ),
        required_scopes=["read"],
        resource_metadata=ResourceMetadata(
            resource="https://mcp.example.com",
            authorization_servers=["https://auth.example.com"],
            scopes_supported=["read"],
            bearer_methods_supported=["header"],
        ),
    )

    d = original.to_dict()
    restored = OAuthDiscoveryResult.from_dict(d)

    assert restored.required_scopes == original.required_scopes
    assert restored.resource_metadata.resource == original.resource_metadata.resource
    assert restored.auth_server_metadata.issuer == original.auth_server_metadata.issuer
    assert restored.auth_server_metadata.authorization_endpoint == original.auth_server_metadata.authorization_endpoint
    assert restored.auth_server_metadata.token_endpoint == original.auth_server_metadata.token_endpoint


def test_oauth_discovery_result_from_dict_minimal():
    d = {
        "auth_server_metadata": {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        },
    }
    result = OAuthDiscoveryResult.from_dict(d)
    assert result.auth_server_metadata.authorization_endpoint == "https://auth.example.com/authorize"
    assert result.auth_server_metadata.token_endpoint == "https://auth.example.com/token"
    assert result.resource_metadata is None


def test_resource_metadata_round_trip():
    original = ResourceMetadata(
        resource="https://mcp.example.com",
        authorization_servers=["https://auth.example.com"],
        scopes_supported=["read"],
        bearer_methods_supported=["header"],
    )
    d = original.to_dict()
    restored = ResourceMetadata.from_dict(d)
    assert restored.resource == original.resource
    assert restored.authorization_servers == original.authorization_servers


def test_authorization_server_metadata_round_trip():
    original = AuthorizationServerMetadata(
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        registration_endpoint="https://auth.example.com/register",
    )
    d = original.to_dict()
    restored = AuthorizationServerMetadata.from_dict(d)
    assert restored.issuer == original.issuer
    assert restored.registration_endpoint == original.registration_endpoint
