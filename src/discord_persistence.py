import gzip
import io
import logging
import os
import time
from typing import Optional, Tuple

import discord

logger = logging.getLogger("warbot.persist")

# Prefixes used in pinned messages inside the storage channel.
TAG_WARS = "[WARSTORE]"
TAG_ROSTER_OVERRIDES = "[ROSTER_OVERRIDES]"
TAG_ROSTER_REMOVED = "[ROSTER_REMOVED]"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


async def get_storage_channel(client: discord.Client, channel_id: int) -> Optional[discord.TextChannel]:
    """Return a TextChannel if accessible."""
    if not channel_id:
        return None

    ch = client.get_channel(channel_id)
    if isinstance(ch, discord.TextChannel):
        return ch

    try:
        fetched = await client.fetch_channel(channel_id)
        if isinstance(fetched, discord.TextChannel):
            return fetched
    except Exception:
        logger.exception("Cannot fetch storage channel: %s", channel_id)
    return None


async def _find_pinned_with_prefix(channel: discord.TextChannel, prefix: str) -> Optional[discord.Message]:
    """Find the newest pinned message with given prefix."""
    try:
        pins = await channel.pins()
    except Exception:
        logger.exception("Failed to list pinned messages")
        return None

    best: Optional[discord.Message] = None
    for m in pins:
        if not (m.content or "").startswith(prefix):
            continue
        if not m.attachments:
            continue
        if best is None:
            best = m
        else:
            # Use Discord created_at ordering
            try:
                if m.created_at and best.created_at and m.created_at > best.created_at:
                    best = m
            except Exception:
                best = m
    return best


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _maybe_gzip(data: bytes, max_bytes: int) -> Tuple[bytes, bool]:
    """If data exceeds max_bytes, gzip it. Returns (payload, gzipped)."""
    if len(data) <= max_bytes:
        return data, False
    out = gzip.compress(data)
    return out, True


async def upload_snapshot(
    channel: discord.TextChannel,
    prefix: str,
    local_path: str,
    max_upload_bytes: int = 7_000_000,
) -> Optional[int]:
    """Upload local_path as a pinned snapshot message.

    Strategy:
    - Remove older pinned snapshot with the same prefix (delete message).
    - Upload a new message with file attachment and pin it.

    Returns message.id on success.
    """
    if not os.path.exists(local_path):
        logger.warning("Snapshot missing on disk: %s", local_path)
        return None

    # Delete previous snapshot (if any)
    prev = await _find_pinned_with_prefix(channel, prefix)
    if prev is not None:
        try:
            await prev.unpin()
        except Exception:
            pass
        try:
            await prev.delete()
        except Exception:
            pass

    data = _read_file_bytes(local_path)
    payload, gz = _maybe_gzip(data, max_upload_bytes)

    fname = os.path.basename(local_path)
    if gz:
        fname = fname + ".gz"

    file_obj = discord.File(io.BytesIO(payload), filename=fname)
    meta = f"{prefix} updated={_now_iso()} bytes={len(data)} gz={1 if gz else 0}"

    try:
        msg = await channel.send(content=meta, file=file_obj)
        try:
            await msg.pin()
        except Exception:
            # Pin isn't strictly required, but it helps fast restore.
            pass
        logger.info("Uploaded snapshot %s as msg_id=%s", prefix, msg.id)
        return msg.id
    except Exception:
        logger.exception("Failed to upload snapshot: %s", prefix)
        return None


async def restore_snapshot(
    channel: discord.TextChannel,
    prefix: str,
    dest_path: str,
) -> bool:
    """Restore snapshot from pinned messages into dest_path.

    Returns True if restored.
    """
    msg = await _find_pinned_with_prefix(channel, prefix)
    if msg is None:
        logger.info("No pinned snapshot for %s", prefix)
        return False

    att = msg.attachments[0]
    try:
        data = await att.read()
    except Exception:
        logger.exception("Failed to download snapshot attachment: %s", prefix)
        return False

    is_gz = (att.filename or "").endswith(".gz")
    if is_gz:
        try:
            data = gzip.decompress(data)
        except Exception:
            logger.exception("Failed to gunzip snapshot: %s", prefix)
            return False

    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(data)
        logger.info("Restored snapshot %s -> %s (%d bytes)", prefix, dest_path, len(data))
        return True
    except Exception:
        logger.exception("Failed to write restored snapshot: %s", dest_path)
        return False
