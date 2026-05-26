"""Tests for app.services.oauth2.cookies"""

from app.services.oauth2.cookies import (
    OAUTH_COOKIE_PREFIX,
    generate_oauth_cookie_key,
    get_oauth_cookie_names,
)


def test_oauth_cookie_prefix():
    assert OAUTH_COOKIE_PREFIX == "mcp_oauth"


def test_generate_oauth_cookie_key_deterministic():
    key1 = generate_oauth_cookie_key("https://auth.example.com/authorize")
    key2 = generate_oauth_cookie_key("https://auth.example.com/authorize")
    assert key1 == key2


def test_generate_oauth_cookie_key_is_8_chars():
    key = generate_oauth_cookie_key("https://auth.example.com/authorize")
    assert len(key) == 8


def test_generate_oauth_cookie_key_different_endpoints():
    key1 = generate_oauth_cookie_key("https://auth1.example.com/authorize")
    key2 = generate_oauth_cookie_key("https://auth2.example.com/authorize")
    assert key1 != key2


def test_get_oauth_cookie_names():
    names = get_oauth_cookie_names("abcd1234")
    assert names["access_token"] == "mcp_oauth_at_abcd1234"
    assert names["refresh_token"] == "mcp_oauth_rt_abcd1234"


def test_get_oauth_cookie_names_empty_key():
    names = get_oauth_cookie_names("")
    assert names["access_token"] == "mcp_oauth_at_"
    assert names["refresh_token"] == "mcp_oauth_rt_"
