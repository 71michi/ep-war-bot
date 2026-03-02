import asyncio
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


def _normalize_attachment_stem(filename: str) -> str:
    """Normalize Discord attachment filenames.

    Users sometimes upload files that their OS renamed to e.g.:
      - wars_store (1).json
      - wars_store (2).json.gz

    This helper returns a stable stem for matching.
    """
    fn = (filename or "").strip()
    if fn.endswith(".gz"):
        fn = fn[:-3]
    if fn.endswith(".json"):
        fn = fn[:-5]

    # Strip trailing " (N)" pattern.
    if fn.endswith(")") and " (" in fn:
        base, maybe = fn.rsplit(" (", 1)
        num = maybe[:-1]
        if num.isdigit():
            fn = base
    return fn


async def _find_pinned_by_attachment_stem(channel: discord.TextChannel, wanted_stem: str) -> Optional[discord.Message]:
    """Fallback: find newest pinned message that has an attachment matching wanted_stem.

    This is used when the pinned message was created manually and doesn't start with our prefix tag.
    """
    try:
        pins = await channel.pins()
    except Exception:
        logger.exception("Failed to list pinned messages")
        return None

    best: Optional[discord.Message] = None
    for m in pins:
        if not m.attachments:
            continue
        att = m.attachments[0]
        stem = _normalize_attachment_stem(att.filename or "")
        if stem != wanted_stem:
            continue
        if best is None:
            best = m
        else:
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

    # Safety: avoid accidentally overwriting a good WARSTORE snapshot with an empty/partial file.
    # This can happen on free hosting tiers if the local filesystem resets and a command triggers
    # persistence before restore completes.
    prev = await _find_pinned_with_prefix(channel, prefix)
    if prefix == TAG_WARS and prev is not None:
        try:
            att = prev.attachments[0]
            prev_bytes = await att.read()
            if (att.filename or "").endswith(".gz"):
                try:
                    prev_bytes = gzip.decompress(prev_bytes)
                except Exception:
                    # If we cannot decompress, fall back to size check only.
                    pass

            # If the new local file is suspiciously small compared to the pinned snapshot,
            # do NOT overwrite it.
            # We use a conservative threshold (25%) to tolerate small growth/shrink.
            new_bytes = _read_file_bytes(local_path)
            if len(new_bytes) < max(5_000, int(0.25 * len(prev_bytes))):
                logger.warning(
                    "Refusing to overwrite %s: local snapshot too small (local=%d, pinned=%d)",
                    prefix,
                    len(new_bytes),
                    len(prev_bytes),
                )
                return prev.id
        except Exception:
            # On any error, keep old behavior (attempt overwrite)
            pass

    # Delete previous snapshot (if any) AFTER safety check.
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

    # If there is no tagged snapshot, try to recover from a manually pinned file
    # with the expected filename (including OS-added " (1)" suffixes).
    if msg is None:
        wanted_stem = _normalize_attachment_stem(os.path.basename(dest_path))
        msg = await _find_pinned_by_attachment_stem(channel, wanted_stem)
        if msg is not None:
            logger.warning(
                "Restoring %s from manually pinned attachment (no %s tag found)",
                wanted_stem,
                prefix,
            )

    # Last-resort fallback: the storage message might not be pinned.
    # Try scanning recent history for the newest matching message.
    if msg is None:
        try:
            wanted_stem = _normalize_attachment_stem(os.path.basename(dest_path))
            async for m in channel.history(limit=100):
                if not m.attachments:
                    continue
                att0 = m.attachments[0]
                stem = _normalize_attachment_stem(att0.filename or "")
                if (m.content or "").startswith(prefix) or stem == wanted_stem:
                    msg = m
                    logger.warning("Restoring %s from recent history (not pinned)", prefix)
                    break
        except Exception:
            pass

    if msg is None:
        logger.info("No snapshot for %s (pinned or recent)", prefix)
        return False

    # UX guard: if WARSTORE exists but is not pinned, warn the operator.
    # This prevents "0 wars after restart" situations.
    try:
        if prefix == TAG_WARS and not getattr(msg, "pinned", False):
            await channel.send(
                "⚠️ **WARSTORE nie jest przypięty.** Po restarcie bota lista wojen może być pusta. "
                "Przypnij wiadomość z `[WARSTORE]` i załącznikiem `wars_store.json` w kanale #warbot-storage.",
                delete_after=15,
            )
    except Exception:
        # Best-effort only.
        pass

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
