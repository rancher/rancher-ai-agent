"""OAuth cookie utilities."""


OAUTH_COOKIE_PREFIX = "mcp_oauth"


def get_oauth_cookie_names(cookie_key: str) -> dict[str, str]:
    """
    Get the cookie names for a given OAuth cookie key.
    Returns a dict with keys: access_token, refresh_token.
    """
    return {
        "access_token": f"{OAUTH_COOKIE_PREFIX}_at_{cookie_key}",
        "refresh_token": f"{OAUTH_COOKIE_PREFIX}_rt_{cookie_key}",
    }
