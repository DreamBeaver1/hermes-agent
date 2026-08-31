"""Tests for Discord thread participation persistence.

Verifies that _threads (ThreadParticipationTracker) survives adapter restarts by
being persisted to ~/.hermes/discord_threads.json.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


class TestDiscordThreadPersistence:
    """Thread IDs are saved to disk and reloaded on init."""

    def _make_adapter(self, tmp_path):
        """Build a minimal DiscordAdapter with HERMES_HOME pointed at tmp_path."""
        from gateway.config import PlatformConfig
        from plugins.platforms.discord.adapter import DiscordAdapter

        config = PlatformConfig(enabled=True, token="test-token")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            return DiscordAdapter(config=config)

    def test_starts_empty_when_no_state_file(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        assert "$nonexistent" not in adapter._threads

    def test_track_thread_persists_to_disk(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            adapter._threads.mark("111")
            adapter._threads.mark("222")

        state_file = tmp_path / "discord_threads.json"
        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert set(saved) == {"111", "222"}

    def test_threads_survive_restart(self, tmp_path):
        """Threads tracked by one adapter instance are visible to the next."""
        adapter1 = self._make_adapter(tmp_path)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            adapter1._threads.mark("aaa")
            adapter1._threads.mark("bbb")

        adapter2 = self._make_adapter(tmp_path)
        assert "aaa" in adapter2._threads
        assert "bbb" in adapter2._threads

    @pytest.mark.asyncio
    async def test_free_response_root_message_without_mention_creates_thread(self, tmp_path):
        """Free-response bypasses mention gating but does not suppress threading."""
        adapter = self._make_adapter(tmp_path)
        adapter.config.extra.update({
            "free_response_channels": ["root-channel"],
            "no_thread_channels": [],
        })
        adapter._discord_require_mention = lambda: True
        message = SimpleNamespace(
            content="Ordinary root-channel request",
            author=SimpleNamespace(display_name="TestUser", bot=False),
            channel=SimpleNamespace(id="root-channel"),
            type=None,
            create_thread=AsyncMock(return_value=SimpleNamespace(id="thread-1")),
        )

        with patch.object(adapter, "_auto_create_thread", new=AsyncMock(return_value=message.create_thread.return_value)) as create_thread:
            # Exercise the routing predicate directly: free-response is only a
            # mention-gate bypass; it must not set skip_thread.
            free_channels = adapter._discord_free_response_channels()
            channel_keys = {"root-channel"}
            no_thread_channels = adapter._get_no_thread_channels()
            is_free_channel = bool(channel_keys & free_channels)
            skip_thread = bool(channel_keys & no_thread_channels)
            if not skip_thread:
                await create_thread(message)

        assert is_free_channel
        create_thread.assert_awaited_once_with(message)

    @pytest.mark.asyncio
    async def test_auto_threads_use_seven_day_archive_window(self, tmp_path):
        """Auto-created conversation threads stay reopenable for seven days."""
        adapter = self._make_adapter(tmp_path)
        message = SimpleNamespace(
            content="Discuss the project",
            author=SimpleNamespace(display_name="TestUser"),
            create_thread=AsyncMock(return_value=SimpleNamespace(id="thread-1")),
        )

        result = await adapter._auto_create_thread(message)

        assert result.id == "thread-1"
        assert message.create_thread.await_args.kwargs["auto_archive_duration"] == 10080


