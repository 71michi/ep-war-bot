"""Discord helper utilities.

These are intentionally small and side-effect free to keep them unit-testable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import discord


logger = logging.getLogger("warbot.discord_utils")


async def safe_delete_message(message: Optional[discord.Message]) -> None:
    """Delete a Discord message (best-effort)."""
    if message is None:
        return
    try:
        await message.delete()
    except (discord.NotFound, discord.Forbidden):
        return
    except Exception:
        logger.debug("Failed to delete message", exc_info=True)


async def delete_message_later(message: Optional[discord.Message], delay_sec: int) -> None:
    """Delete a message after delay_sec seconds (best-effort)."""
    if message is None:
        return
    try:
        await asyncio.sleep(max(0, int(delay_sec)))
        await safe_delete_message(message)
    except Exception:
        # Never crash the bot because of a cleanup task.
        logger.debug("Failed to auto-delete message", exc_info=True)
