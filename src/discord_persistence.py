import os
import io
import time
import logging
from typing import Optional, Tuple, List

import discord

logger = logging.getLogger("warbot.persist")

TAG_WARS = "[WARSTORE]"
TAG_WARS_BACKUP = "[WARSTORE_BACKUP]"
TAG_ROSTER_OVERRIDES = "[ROSTER_OVERRIDES]"
TAG_ROSTER_REMOVED = "[ROSTER_REMOVED]"

def _is_tag(msg: discord.Message, tag: str) -> bool:
    try:
        c = (msg.content or "").strip()
        return c.startswith(tag)
    except Exception:
        return False

async def get_storage_channel(client: discord.Client, channel_id: int) -> Optional[discord.TextChannel]:
    """Fetch a TextChannel by id. Returns None if not found / no access."""
    if not channel_id:
        return None
    ch = client.get_channel(channel_id)
    if isinstance(ch, discord.TextChannel):
        return ch
    try:
        fetched = await client.fetch_channel(channel_id)
        return fetched if isinstance(fetched, discord.TextChannel) else None
    except Exception as e:
        logger.warning("Cannot fetch storage channel %s: %r", channel_id, e)
        return None

async def _find_latest_snapshot_message(channel: discord.TextChannel, tag: str, *, history_limit: int = 200) -> Optional[discord.Message]:
    """Find latest message with given tag and at least one attachment.
    Prefer pinned, but fall back to recent history.
    """
    # Prefer pins
    try:
        pins = await channel.pins()
        tagged = [m for m in pins if _is_tag(m, tag) and m.attachments]
        if tagged:
            tagged.sort(key=lambda m: m.created_at, reverse=True)
            return tagged[0]
    except Exception:
        pass

    # Fallback: scan recent history
    try:
        async for m in channel.history(limit=history_limit, oldest_first=False):
            if _is_tag(m, tag) and m.attachments:
                logger.warning("Restoring %s from recent history (not pinned)", tag)
                return m
    except Exception as e:
        logger.warning("History scan failed in %s: %r", channel.id, e)
    return None

async def restore_snapshot(channel: discord.TextChannel, tag: str, dest_path: str, *, history_limit: int = 200) -> bool:
    """Restore a snapshot attachment into dest_path. Returns True if restored."""
    msg = await _find_latest_snapshot_message(channel, tag, history_limit=history_limit)
    if msg is None:
        return False
    att = msg.attachments[0]
    try:
        data = await att.read()
        # Write atomically
        tmp = dest_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest_path)
        logger.info("Restored snapshot %s -> %s (%d bytes)", tag, os.path.basename(dest_path), len(data))
        return True
    except Exception as e:
        logger.error("Failed to restore %s: %r", tag, e)
        return False

async def _latest_pinned_by_tag(channel: discord.TextChannel, tag: str) -> Optional[discord.Message]:
    try:
        pins = await channel.pins()
        tagged = [m for m in pins if _is_tag(m, tag) and m.attachments]
        if not tagged:
            return None
        tagged.sort(key=lambda m: m.created_at, reverse=True)
        return tagged[0]
    except Exception:
        return None

async def upload_snapshot(
    channel: discord.TextChannel,
    tag: str,
    src_path: str,
    *,
    max_bytes: int = 7000000,
    pin: bool = True,
    keep_backups: int = 2,
) -> Optional[discord.Message]:
    """Upload src_path as a Discord attachment under a tag.
    For WARSTORE we keep a backup pinned message before replacing.
    """
    try:
        size = os.path.getsize(src_path)
    except Exception:
        size = -1
    if size <= 0:
        logger.warning("Skip upload %s: empty file %s", tag, src_path)
        return None
    if size > max_bytes:
        logger.error("Skip upload %s: %s is too large (%d > %d)", tag, src_path, size, max_bytes)
        return None

    prev = await _latest_pinned_by_tag(channel, tag) if pin else None

    # For WARSTORE: turn previous into BACKUP instead of deleting.
    if prev is not None and tag == TAG_WARS:
        try:
            old_content = prev.content or ""
            if old_content.startswith(TAG_WARS):
                new_content = old_content.replace(TAG_WARS, TAG_WARS_BACKUP, 1)
            else:
                new_content = f"{TAG_WARS_BACKUP} {old_content}".strip()
            await prev.edit(content=new_content[:1900])
        except Exception:
            pass
        try:
            await prev.unpin()
        except Exception:
            pass

        # prune old backups
        try:
            pins = await channel.pins()
            backups = [m for m in pins if _is_tag(m, TAG_WARS_BACKUP) and m.attachments]
            backups.sort(key=lambda m: m.created_at, reverse=True)
            for old in backups[keep_backups:]:
                try:
                    await old.unpin()
                except Exception:
                    pass
                try:
                    await old.delete()
                except Exception:
                    pass
        except Exception:
            pass

    # For other tags: remove previous pinned
    if prev is not None and tag != TAG_WARS:
        try:
            await prev.unpin()
        except Exception:
            pass
        try:
            await prev.delete()
        except Exception:
            pass

    filename = os.path.basename(src_path)
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    content = f"{tag} updated={ts} bytes={size} gz=0"
    try:
        with open(src_path, "rb") as f:
            file = discord.File(fp=io.BytesIO(f.read()), filename=filename)
        msg = await channel.send(content=content, file=file)
        if pin:
            try:
                await msg.pin()
            except Exception:
                pass
        return msg
    except Exception as e:
        logger.error("Upload snapshot failed (%s): %r", tag, e)
        return None
