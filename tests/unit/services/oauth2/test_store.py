"""Tests for app.services.oauth2.store"""

import asyncio
import time

import pytest

from app.services.oauth2.store import OAuthTokenStore, _StateEntry, _TokenEntry, _TTL_SECONDS


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
        store.set_state("s1", {"verifier": "abc", "cookie_key": "k"})
        assert store.pop_state("s1") == {"verifier": "abc", "cookie_key": "k"}

    def test_pop_state_is_one_time_use(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"x": 1})
        store.pop_state("s1")
        assert store.pop_state("s1") is None

    def test_pop_state_missing_key_returns_none(self):
        store = OAuthTokenStore()
        assert store.pop_state("nonexistent") is None

    def test_set_state_overwrites_previous_entry(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"x": 1})
        store.set_state("s1", {"x": 2})
        assert store.pop_state("s1") == {"x": 2}

    # --- token ---

    def test_pop_token_returns_stored_token(self):
        store = OAuthTokenStore()
        store.set_token("access_cookie", "token_value")
        assert store.pop_token("access_cookie") == "token_value"

    def test_pop_token_is_one_time_use(self):
        store = OAuthTokenStore()
        store.set_token("access_cookie", "token_value")
        store.pop_token("access_cookie")
        assert store.pop_token("access_cookie") is None

    def test_pop_token_missing_key_returns_none(self):
        store = OAuthTokenStore()
        assert store.pop_token("nonexistent") is None

    def test_set_token_overwrites_previous_entry(self):
        store = OAuthTokenStore()
        store.set_token("c", "old")
        store.set_token("c", "new")
        assert store.pop_token("c") == "new"

    # --- eviction ---

    def test_evict_expired_removes_old_state(self):
        store = OAuthTokenStore()
        store.set_state("old", {"x": 1})
        store._state_store["old"].created_at -= _TTL_SECONDS + 1
        store._evict_expired()
        assert "old" not in store._state_store

    def test_evict_expired_removes_old_token(self):
        store = OAuthTokenStore()
        store.set_token("old_cookie", "tok")
        store._token_store["old_cookie"].created_at -= _TTL_SECONDS + 1
        store._evict_expired()
        assert "old_cookie" not in store._token_store

    def test_evict_expired_keeps_fresh_entries(self):
        store = OAuthTokenStore()
        store.set_state("fresh_state", {"x": 1})
        store.set_token("fresh_token", "val")
        store._evict_expired()
        assert "fresh_state" in store._state_store
        assert "fresh_token" in store._token_store

    def test_evict_expired_only_removes_expired_entries(self):
        store = OAuthTokenStore()
        store.set_state("expired", {"x": 1})
        store.set_state("fresh", {"y": 2})
        store._state_store["expired"].created_at -= _TTL_SECONDS + 1
        store._evict_expired()
        assert "expired" not in store._state_store
        assert "fresh" in store._state_store

    # --- cleanup task lifecycle ---

    def test_no_cleanup_task_outside_async_context(self):
        store = OAuthTokenStore()
        store.set_state("s", {"a": 1})
        store.set_token("c", "t")
        assert store._cleanup_task is None

    @pytest.mark.asyncio
    async def test_cleanup_task_started_on_first_set_state(self):
        store = OAuthTokenStore()
        assert store._cleanup_task is None
        store.set_state("s", {"x": 1})
        assert store._cleanup_task is not None
        assert not store._cleanup_task.done()
        store._cleanup_task.cancel()

    @pytest.mark.asyncio
    async def test_cleanup_task_started_on_first_set_token(self):
        store = OAuthTokenStore()
        assert store._cleanup_task is None
        store.set_token("c", "t")
        assert store._cleanup_task is not None
        assert not store._cleanup_task.done()
        store._cleanup_task.cancel()

    @pytest.mark.asyncio
    async def test_cleanup_task_not_restarted_while_running(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"x": 1})
        task1 = store._cleanup_task
        store.set_state("s2", {"y": 2})
        assert store._cleanup_task is task1
        task1.cancel()

    @pytest.mark.asyncio
    async def test_cleanup_task_restarted_after_done(self):
        store = OAuthTokenStore()
        store.set_state("s1", {"x": 1})
        task1 = store._cleanup_task
        task1.cancel()
        # Let the event loop process the cancellation
        await asyncio.sleep(0)
        store.set_state("s2", {"y": 2})
        assert store._cleanup_task is not task1
        store._cleanup_task.cancel()
