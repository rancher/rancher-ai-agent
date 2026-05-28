"""Tests for app.services.oauth2.cookies"""

from app.services.oauth2.cookies import (
    OAUTH_COOKIE_PREFIX,
    get_oauth_cookie_names,
)


def test_oauth_cookie_prefix():
    assert OAUTH_COOKIE_PREFIX == "mcp_oauth"


def test_get_oauth_cookie_names():
    names = get_oauth_cookie_names("my-agent")
    assert names["access_token"] == "mcp_oauth_at_my-agent"
    assert names["refresh_token"] == "mcp_oauth_rt_my-agent"


def test_get_oauth_cookie_names_empty_key():
    names = get_oauth_cookie_names("")
    assert names["access_token"] == "mcp_oauth_at_"
    assert names["refresh_token"] == "mcp_oauth_rt_"
