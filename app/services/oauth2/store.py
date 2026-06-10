import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_TTL_SECONDS = 600  # 10 minutes


def _make_key(session_token: str, key: str) -> str:
    return f"{session_token}:{key}"


@dataclass
class _StateEntry:
    """Bundles an OAuth state payload with the monotonic timestamp of its creation.

    Used to enforce TTL-based eviction without a separate timestamps dict.
    """

    data: dict[str, Any]
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class _TokenEntry:
    """Bundles an OAuth token value with the monotonic timestamp of its creation.

    Used to enforce TTL-based eviction without a separate timestamps dict.
    """

    token: str
    created_at: float = field(default_factory=time.monotonic)


class OAuthTokenStore:
    """In-process store for short-lived OAuth state and access/refresh tokens.

    Two kinds of entries are managed:

    * **State entries** – keyed by the random ``state`` parameter generated
      during the authorization redirect.  They hold the PKCE code verifier,
      the agent name, and redirect URI needed to complete the code exchange
      in the callback handler.

    * **Token entries** – keyed by the cookie name.  They hold the raw
      access/refresh token string produced by the callback and consumed once
      by the WebSocket handler (which cannot see cookies set after the
      handshake).

    Both entry types are **single-use**: ``pop_state`` and ``pop_token``
    delete the entry on first read so a token can never be replayed.

    A background asyncio task runs every 60 s and evicts any entry that has
    not been consumed within ``_TTL_SECONDS`` (10 minutes), preventing
    unbounded growth from abandoned flows.  The task is started lazily on the
    first write, so no explicit lifecycle management is required.
    """

    def __init__(self):
        self._state_store: dict[str, _StateEntry] = {}
        self._token_store: dict[str, _TokenEntry] = {}
        self._cleanup_task: asyncio.Task | None = None

    def _ensure_cleanup_running(self) -> None:
        """Start the background eviction task if it is not already running.

        Called before every write.  Safe to call outside an async context
        (e.g. during testing or startup) — it will simply return without
        creating a task.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = loop.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Periodically call ``_evict_expired`` until the task is cancelled."""
        while True:
            await asyncio.sleep(60)
            self._evict_expired()

    def _evict_expired(self) -> None:
        """Remove all state and token entries that have exceeded the TTL."""
        now = time.monotonic()
        for k in [k for k, e in self._state_store.items() if now - e.created_at > _TTL_SECONDS]:
            del self._state_store[k]
            logger.debug(f"Evicted expired OAuth state: {k}")
        for k in [k for k, e in self._token_store.items() if now - e.created_at > _TTL_SECONDS]:
            del self._token_store[k]
            logger.debug(f"Evicted expired OAuth token: {k}")

    def set_state(self, state: str, data: dict[str, Any], session_token: str) -> None:
        """Store ``data`` under ``state``, overwriting any previous entry."""
        self._ensure_cleanup_running()
        self._state_store[_make_key(session_token, state)] = _StateEntry(data=data)

    def pop_state(self, state: str, session_token: str) -> dict[str, Any] | None:
        """Remove and return the data for ``state``, or ``None`` if absent."""
        entry = self._state_store.pop(_make_key(session_token, state), None)
        return entry.data if entry else None

    def set_token(self, cookie_name: str, token: str, session_token: str) -> None:
        """Store ``token`` under ``cookie_name``, overwriting any previous entry."""
        self._ensure_cleanup_running()
        self._token_store[_make_key(session_token, cookie_name)] = _TokenEntry(token=token)

    def pop_token(self, cookie_name: str, session_token: str) -> str | None:
        """Remove and return the token for ``cookie_name``, or ``None`` if absent."""
        entry = self._token_store.pop(_make_key(session_token, cookie_name), None)
        return entry.token if entry else None


oauth_store = OAuthTokenStore()
