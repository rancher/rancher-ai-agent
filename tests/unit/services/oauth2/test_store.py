"""Tests for app.services.oauth2.store"""

import asyncio
import time

import pytest

from app.services.oauth2.store import OAuthTokenStore, _StateEntry, _TokenEntry, _TTL_SECONDS, _make_key

SESSION = "sess-abc"


class TestStateEntry:
    def test_stores_data_and_timestamp(self):
        entry = _StateEntry(data={"verifier": "abc"})
        assert entry.data == {"verifier": "abc"}
        assert isinstance(entry.created_at, float)

    def test_created_at_reflects_current_time(self):
        before = time.monotonic()
        entry = _StateEntry(data={})
        after = time.monotonic()
        assert before <= entry.created_at <= after


class TestTokenEntry:
    def test_stores_token_and_timestamp(self):
        entry = _TokenEntry(token="tok123")
        assert entry.token == "tok123"
        assert isinstance(entry.created_at, float)

    def test_created_at_reflects_current_time(self):
        before = time.monotonic()
        entry = _TokenEntry(token="t")
        after = time.monotonic()
        assert before <= entry.created_at <= after


class TestOAuthTokenStore:
    # --- state ---

    def test_pop_state_returns_stored_data(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"verifier": "abc", "cookie_key": "k"}, SESSION)
        assert store.pop_state("s1", SESSION) == {"verifier": "abc", "cookie_key": "k"}

    def test_pop_state_is_one_time_use(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"x": 1}, SESSION)
        store.pop_state("s1", SESSION)
        assert store.pop_state("s1", SESSION) is None

    def test_pop_state_missing_key_returns_none(self):
        store = OAuthTokenStore()
        assert store.pop_state("nonexistent", SESSION) is None

    def test_set_state_overwrites_previous_entry(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"x": 1}, SESSION)
        store.set_state("s1", {"x": 2}, SESSION)
        assert store.pop_state("s1", SESSION) == {"x": 2}

    def test_pop_state_isolated_by_session(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"x": 1}, "session-a")
        assert store.pop_state("s1", "session-b") is None
        assert store.pop_state("s1", "session-a") == {"x": 1}

    # --- token ---

    def test_pop_token_returns_stored_token(self):
        store = OAuthTokenStore()
        store.set_token("access_cookie", "token_value", SESSION)
        assert store.pop_token("access_cookie", SESSION) == "token_value"

    def test_pop_token_is_one_time_use(self):
        store = OAuthTokenStore()
        store.set_token("access_cookie", "token_value", SESSION)
        store.pop_token("access_cookie", SESSION)
        assert store.pop_token("access_cookie", SESSION) is None

    def test_pop_token_missing_key_returns_none(self):
        store = OAuthTokenStore()
        assert store.pop_token("nonexistent", SESSION) is None

    def test_set_token_overwrites_previous_entry(self):
        store = OAuthTokenStore()
        store.set_token("c", "old", SESSION)
        store.set_token("c", "new", SESSION)
        assert store.pop_token("c", SESSION) == "new"

    def test_pop_token_isolated_by_session(self):
        store = OAuthTokenStore()
        store.set_token("c", "tok", "session-a")
        assert store.pop_token("c", "session-b") is None
        assert store.pop_token("c", "session-a") == "tok"

    # --- eviction ---

    def test_evict_expired_removes_old_state(self):
        store = OAuthTokenStore()
        store.set_state("old", {"x": 1}, SESSION)
        store._state_store[_make_key(SESSION, "old")].created_at -= _TTL_SECONDS + 1
        store._evict_expired()
        assert _make_key(SESSION, "old") not in store._state_store

    def test_evict_expired_removes_old_token(self):
        store = OAuthTokenStore()
        store.set_token("old_cookie", "tok", SESSION)
        store._token_store[_make_key(SESSION, "old_cookie")].created_at -= _TTL_SECONDS + 1
        store._evict_expired()
        assert _make_key(SESSION, "old_cookie") not in store._token_store

    def test_evict_expired_keeps_fresh_entries(self):
        store = OAuthTokenStore()
        store.set_state("fresh_state", {"x": 1}, SESSION)
        store.set_token("fresh_token", "val", SESSION)
        store._evict_expired()
        assert _make_key(SESSION, "fresh_state") in store._state_store
        assert _make_key(SESSION, "fresh_token") in store._token_store

    def test_evict_expired_only_removes_expired_entries(self):
        store = OAuthTokenStore()
        store.set_state("expired", {"x": 1}, SESSION)
        store.set_state("fresh", {"y": 2}, SESSION)
        store._state_store[_make_key(SESSION, "expired")].created_at -= _TTL_SECONDS + 1
        store._evict_expired()
        assert _make_key(SESSION, "expired") not in store._state_store
        assert _make_key(SESSION, "fresh") in store._state_store

    # --- cleanup task lifecycle ---

    def test_no_cleanup_task_outside_async_context(self):
        store = OAuthTokenStore()
        store.set_state("s", {"a": 1}, SESSION)
        store.set_token("c", "t", SESSION)
        assert store._cleanup_task is None

    @pytest.mark.asyncio
    async def test_cleanup_task_started_on_first_set_state(self):
        store = OAuthTokenStore()
        assert store._cleanup_task is None
        store.set_state("s", {"x": 1}, SESSION)
        assert store._cleanup_task is not None
        assert not store._cleanup_task.done()
        store._cleanup_task.cancel()

    @pytest.mark.asyncio
    async def test_cleanup_task_started_on_first_set_token(self):
        store = OAuthTokenStore()
        assert store._cleanup_task is None
        store.set_token("c", "t", SESSION)
        assert store._cleanup_task is not None
        assert not store._cleanup_task.done()
        store._cleanup_task.cancel()

    @pytest.mark.asyncio
    async def test_cleanup_task_not_restarted_while_running(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"x": 1}, SESSION)
        task1 = store._cleanup_task
        store.set_state("s2", {"y": 2}, SESSION)
        assert store._cleanup_task is task1
        task1.cancel()

    @pytest.mark.asyncio
    async def test_cleanup_task_restarted_after_done(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"x": 1}, SESSION)
        task1 = store._cleanup_task
        task1.cancel()
        # Let the event loop process the cancellation
        await asyncio.sleep(0)
        store.set_state("s2", {"y": 2}, SESSION)
        assert store._cleanup_task is not task1
        store._cleanup_task.cancel()
