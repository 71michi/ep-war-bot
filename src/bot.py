import os
import re
import asyncio
import logging
import time
import uuid
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from aiohttp import web
import discord

from .config import env_str, env_int, env_bool
from .logging_setup import setup_logging, set_trace_id, reset_trace_id, get_trace_id
from .openai_parser import parse_war_from_images, WarSummary, PlayerScore
from .nicknames import normalize_with_aliases, normalize_display, roster_match, canonical_key, is_clean_display
from .war_store import WarStore
from .discord_persistence import (
    get_storage_channel, restore_snapshot, upload_snapshot,
    TAG_WARS, TAG_ROSTER_OVERRIDES, TAG_ROSTER_REMOVED,
)
from .discord_utils import delete_message_later
from rapidfuzz import fuzz, process

setup_logging()
logger = logging.getLogger("warbot")


# ----------------------------
# Duplicate-response guard
# ----------------------------
#
# On some hosting setups (especially during redeploys) it's possible to have a
# short overlap where two bot instances are connected to Discord at the same
# time. Then both can process the same command message and send duplicate
# replies.
#
# We mitigate this by attempting to delete command messages *immediately* to
# "claim" them. If another instance already handled the command, the delete
# will fail with NotFound and we skip processing.
EARLY_DELETE_COMMANDS = env_bool("EARLY_DELETE_COMMANDS", True)

# In practice, Discord may occasionally deliver the same message event more than once
# (e.g. around reconnect/resume), and on free hosting tiers you might also get short
# overlaps of two bot processes. To avoid duplicate replies we implement a small
# idempotency cache (per-process) plus a best-effort cross-process "claim".
COMMAND_DEDUP_TTL_SEC = env_int("COMMAND_DEDUP_TTL_SEC", 120)
COMMAND_DEDUP_MAX = env_int("COMMAND_DEDUP_MAX", 5000)
USE_REACTION_CLAIM = env_bool("USE_REACTION_CLAIM", True)
COMMAND_CLAIM_REACTION = env_str("COMMAND_CLAIM_REACTION", "🔒")

# msg_id -> first_seen_ts (LRU)
_COMMAND_DEDUP_CACHE: "OrderedDict[int, float]" = OrderedDict()


def _command_seen_recently(msg_id: int) -> bool:
    """Return True if msg_id was processed recently (idempotency guard)."""
    now = time.time()

    # purge old
    if COMMAND_DEDUP_TTL_SEC > 0:
        cutoff = now - float(COMMAND_DEDUP_TTL_SEC)
        # OrderedDict: pop from left while old
        while _COMMAND_DEDUP_CACHE:
            k0, ts0 = next(iter(_COMMAND_DEDUP_CACHE.items()))
            if ts0 >= cutoff:
                break
            _COMMAND_DEDUP_CACHE.popitem(last=False)

    if msg_id in _COMMAND_DEDUP_CACHE:
        # touch
        _COMMAND_DEDUP_CACHE.move_to_end(msg_id)
        return True

    _COMMAND_DEDUP_CACHE[msg_id] = now
    _COMMAND_DEDUP_CACHE.move_to_end(msg_id)
    # cap
    while len(_COMMAND_DEDUP_CACHE) > max(1, int(COMMAND_DEDUP_MAX)):
        _COMMAND_DEDUP_CACHE.popitem(last=False)
    return False


async def _try_claim_command_message(message: discord.Message) -> bool:
    """Best-effort cross-instance de-duplication.

    If the bot has permission to delete messages, we try to delete the user's
    command message *immediately*. Only one instance can successfully delete;
    the others will receive NotFound and must skip processing.

    Returns:
        True  -> proceed with handling
        False -> skip (likely handled by another instance)
    """

    # Per-process idempotency guard (handles duplicate delivery within a single instance)
    try:
        if _command_seen_recently(int(message.id)):
            logger.info("Duplicate command delivery detected -> skip: msg_id=%s", message.id)
            return False
    except Exception:
        # never block
        pass

    # Cross-process claim: best-effort reaction lock (works even without Manage Messages)
    if USE_REACTION_CLAIM and COMMAND_CLAIM_REACTION:
        try:
            await message.add_reaction(COMMAND_CLAIM_REACTION)
            logger.debug("Claimed command message by reaction: msg_id=%s emoji=%s", message.id, COMMAND_CLAIM_REACTION)
        except discord.NotFound:
            # likely already deleted by another instance
            logger.info("Command message missing while claiming (duplicate?) -> skip: msg_id=%s", message.id)
            return False
        except discord.Forbidden:
            # Missing Add Reactions permission; continue to other methods.
            logger.debug("Reaction-claim forbidden; will try early delete if enabled: msg_id=%s", message.id)
        except discord.HTTPException as e:
            # If another instance already added this reaction, Discord returns 400; treat as claimed elsewhere.
            if getattr(e, "status", None) == 400:
                logger.info("Reaction-claim already exists (duplicate instance?) -> skip: msg_id=%s", message.id)
                return False
            logger.debug("Reaction-claim HTTPException; continuing: msg_id=%s", message.id)

    if not EARLY_DELETE_COMMANDS:
        return True

    try:
        await message.delete()
        logger.debug("Claimed command message by early delete: msg_id=%s", message.id)
        return True
    except discord.NotFound:
        logger.info("Command message already deleted (duplicate instance?) -> skip: msg_id=%s", message.id)
        return False
    except discord.Forbidden:
        # Missing Manage Messages permission; we can't claim. Fall back to normal handling.
        logger.debug("EARLY_DELETE_COMMANDS: missing permissions; cannot claim msg_id=%s", message.id)
        return True
    except discord.HTTPException:
        # Any transient HTTP error: don't block execution.
        logger.debug("EARLY_DELETE_COMMANDS: HTTPException; cannot claim msg_id=%s", message.id)
        return True


def _base36(n: int) -> str:
    """Encode non-negative int to base36 (0-9A-Z)."""
    if n <= 0:
        return "0"
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out: List[str] = []
    x = int(n)
    while x:
        x, r = divmod(x, 36)
        out.append(chars[r])
    return "".join(reversed(out))


def make_war_id(ref_message_id: int) -> str:
    """Deterministic war ID derived from the Discord message id containing the screenshots.

    IMPORTANT: For the *same* screenshot message, this must be stable across repeated LISTWAR.
    """
    b36 = _base36(int(ref_message_id)).upper()
    # Keep it short but globally unique enough (Discord snowflakes are globally unique anyway).
    short = b36[-10:] if len(b36) > 10 else b36
    return f"WAR-{short}"


def make_war_id_with_seq(ref_message_id: int, seq: int) -> str:
    """War ID with a per-ref-message sequence suffix.

    - seq<=1: keep legacy stable ID (WAR-XXXX)
    - seq>1 : WAR-XXXX-2, WAR-XXXX-3, ...
    """
    base = make_war_id(ref_message_id)
    try:
        s = int(seq)
    except Exception:
        s = 1
    return base if s <= 1 else f"{base}-{s}"


def _flush_logs() -> None:
    """Best-effort flush of all handlers.

    We flush both root handlers (console + per-trace capture) and the 'warbot'
    handlers (file handler).
    """
    seen: set[int] = set()
    for lg in (logging.getLogger(), logging.getLogger("warbot")):
        for h in list(lg.handlers):
            hid = id(h)
            if hid in seen:
                continue
            seen.add(hid)
            try:
                h.flush()
            except Exception:
                pass


def _new_trace_id(prefix: str = "msg") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}-{int(time.time())}"

DISCORD_TOKEN = env_str("DISCORD_TOKEN", "")
WATCH_CHANNEL_ID = env_int("WATCH_CHANNEL_ID", 0)
# Default to a slightly better model for more robust OCR / vision parsing.
# You can override via OPENAI_MODEL in .env (e.g. set gpt-4o-mini for cheaper runs).
OPENAI_MODEL = env_str("OPENAI_MODEL", "gpt-4o")
ALIASES_PATH = env_str("ALIASES_PATH", "aliases.json")
ROSTER_PATH = env_str("ROSTER_PATH", "roster.json")
ROSTER_OVERRIDES_PATH = env_str("ROSTER_OVERRIDES_PATH", "roster_overrides.json")
ROSTER_REMOVED_PATH = env_str("ROSTER_REMOVED_PATH", "roster_removed.json")
WAR_STORE_PATH = env_str("WAR_STORE_PATH", "wars_store.json")
DISCORD_STORAGE_CHANNEL_ID = env_int("DISCORD_STORAGE_CHANNEL_ID", 0)
DISCORD_PERSIST_MAX_BYTES = env_int("DISCORD_PERSIST_MAX_BYTES", 7000000)

# LISTWAR: inform user that processing has started (helps UX on slow hosting / OpenAI latency)
LISTWAR_PROGRESS_ENABLED = env_bool("LISTWAR_PROGRESS_ENABLED", True)

# HELP message auto-delete (seconds)
HELP_AUTO_DELETE_SEC = env_int("HELP_AUTO_DELETE_SEC", 30)

# In free hosting tiers with ephemeral filesystem (e.g. Render Free), you can
# keep progress by persisting JSON snapshots into a private Discord channel.
_STORAGE_CHANNEL: Optional[discord.TextChannel] = None
_STORAGE_LOCK = asyncio.Lock()

WEB_INDEX_PATH = os.path.join(os.path.dirname(__file__), "web", "index.html")

WAR_STORE = WarStore(WAR_STORE_PATH)
WAR_STORE_LOCK = asyncio.Lock()


async def _ensure_storage_channel(client: discord.Client) -> Optional[discord.TextChannel]:
    """Lazy-init the Discord storage channel (for persistence on free tiers)."""
    global _STORAGE_CHANNEL
    if DISCORD_STORAGE_CHANNEL_ID == 0:
        return None
    if _STORAGE_CHANNEL is not None:
        return _STORAGE_CHANNEL
    async with _STORAGE_LOCK:
        if _STORAGE_CHANNEL is not None:
            return _STORAGE_CHANNEL
        ch = await get_storage_channel(client, DISCORD_STORAGE_CHANNEL_ID)
        _STORAGE_CHANNEL = ch
        return ch


async def _restore_progress_from_discord(client: discord.Client) -> None:
    """Restore persisted JSON snapshots from a private Discord channel.

    This solves the "Render Free resets filesystem" problem without paid disks/DB.
    """
    ch = await _ensure_storage_channel(client)
    if ch is None:
        return

    restored_any = False
    try:
        restored_any |= await restore_snapshot(ch, TAG_WARS, WAR_STORE_PATH)
        restored_any |= await restore_snapshot(ch, TAG_ROSTER_OVERRIDES, ROSTER_OVERRIDES_PATH)
        restored_any |= await restore_snapshot(ch, TAG_ROSTER_REMOVED, ROSTER_REMOVED_PATH)
    except Exception:
        logger.exception("Restore from Discord failed")
        return

    if restored_any:
        # Invalidate caches so we read restored data.
        try:
            await asyncio.to_thread(WAR_STORE.load)
        except Exception:
            pass
        try:
            _invalidate_roster_cache()
        except Exception:
            pass
        logger.info("Restored progress from Discord snapshots")


async def _persist_progress_to_discord(client: discord.Client, what: str = "all") -> None:
    """Persist JSON snapshots to Discord.

    what: 'all' | 'wars' | 'roster'
    """
    ch = await _ensure_storage_channel(client)
    if ch is None:
        return

    try:
        if what in {"all", "wars"}:
            await upload_snapshot(ch, TAG_WARS, WAR_STORE_PATH, max_upload_bytes=DISCORD_PERSIST_MAX_BYTES)
        if what in {"all", "roster"}:
            await upload_snapshot(ch, TAG_ROSTER_OVERRIDES, ROSTER_OVERRIDES_PATH, max_upload_bytes=DISCORD_PERSIST_MAX_BYTES)
            await upload_snapshot(ch, TAG_ROSTER_REMOVED, ROSTER_REMOVED_PATH, max_upload_bytes=DISCORD_PERSIST_MAX_BYTES)
    except Exception:
        logger.exception("Persist to Discord failed (what=%s)", what)


# When running on Render Free (no shell / no FS access), you can still get the full debug trace.
# By default we attach per-run debug logs as a text file on Discord.
SEND_LOG_TO_DISCORD = env_int("SEND_LOG_TO_DISCORD", 1) == 1
# If enabled, we only send logs when at least one player remained UNKNOWN (or on exceptions).
# This is the recommended setting on free hosting tiers.
SEND_LOG_ONLY_ON_UNKNOWN = env_int("SEND_LOG_ONLY_ON_UNKNOWN", 1) == 1
# Backwards compatibility (deprecated): older packages used SEND_LOG_ONLY_ON_WARN.
SEND_LOG_ONLY_ON_WARN = env_int("SEND_LOG_ONLY_ON_WARN", 0) == 1
# Safety cap so we don't exceed Discord upload limits.
DISCORD_LOG_MAX_BYTES = env_int("DISCORD_LOG_MAX_BYTES", 1_500_000)


def _redact_secrets(text: str) -> str:
    """Best-effort redaction (avoid leaking tokens in case a library logs them)."""
    if not text:
        return ""
    # OpenAI keys often look like: sk-...
    text = re.sub(r"\bsk-[A-Za-z0-9]{10,}\b", "sk-***REDACTED***", text)
    # Discord bot tokens are base64-ish and dot-separated.
    text = re.sub(r"\b([MN][A-Za-z\d_-]{20,}\.[A-Za-z\d_-]{6,}\.[A-Za-z\d_-]{20,})\b", "***REDACTED_DISCORD_TOKEN***", text)
    # Generic Bearer tokens
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*", "Bearer ***REDACTED***", text)
    return text


class _PerTraceCaptureFilter(logging.Filter):
    def __init__(self, trace_id: str):
        super().__init__()
        self._trace_id = trace_id

    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure trace_id exists even if other handlers didn't inject it.
        if not hasattr(record, "trace_id"):
            try:
                record.trace_id = get_trace_id()  # type: ignore[attr-defined]
            except Exception:
                record.trace_id = "-"  # type: ignore[attr-defined]

        if getattr(record, "trace_id", "-") != self._trace_id:
            return False

        # Keep the debug trace readable:
        # - Always include our own logs (warbot.*)
        # - Include WARNING/ERROR from any other library (useful for diagnosing crashes)
        if record.name.startswith("warbot"):
            return True
        return record.levelno >= logging.WARNING


def _attach_per_trace_logger(trace_id: str) -> Tuple[logging.Handler, io.StringIO]:
    """Attach a temporary DEBUG handler capturing only this trace_id."""
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s"))
    h.addFilter(_PerTraceCaptureFilter(trace_id))
    logging.getLogger().addHandler(h)
    return h, stream


# ----------------------------
# Keepalive server for Render
# ----------------------------
_keepalive_started = False


async def start_keepalive_server():
    """HTTP server used for /health and a simple dashboard.

    Render free tier may sleep; this endpoint still helps to wake it.
    On Oracle Always Free you can keep the process running 24/7.
    """
    global _keepalive_started
    if _keepalive_started:
        return
    _keepalive_started = True

    app = web.Application()

    # Serve static web assets (icons, css, etc.)
    web_root = os.path.join(os.path.dirname(__file__), "web")
    app.router.add_static("/static/", web_root, show_index=False)

    async def health(_request):
        return web.Response(text="ok", content_type="text/plain")

    async def index(_request):
        try:
            with open(WEB_INDEX_PATH, "r", encoding="utf-8") as f:
                html = f.read()
        except Exception:
            html = "<h1>Dashboard missing</h1><p>Brak pliku index.html</p>"
        return web.Response(text=html, content_type="text/html")

    async def api_wars(request):
        include_drafts = request.query.get("include_drafts") in {"1", "true", "yes"}
        status_q = (request.query.get("status") or "").strip().lower()
        limit_raw = request.query.get("limit")
        limit = None
        if limit_raw is not None and limit_raw.strip() != "":
            try:
                limit = max(1, int(limit_raw))
            except Exception:
                limit = None

        async with WAR_STORE_LOCK:
            wars = await asyncio.to_thread(WAR_STORE.get_wars, True)

        # Default: only confirmed wars are shown on the website.
        if not include_drafts:
            wars = [w for w in wars if str(w.get("status") or "confirmed").lower() == "confirmed"]

        # Optional: explicit status filter
        if status_q:
            wars = [w for w in wars if str(w.get("status") or "").lower() == status_q]

        if limit is not None:
            wars = wars[:limit]

        # Normalize mode for stable UI filtering (legacy records may have mixed case).
        for w in wars:
            m = w.get("mode")
            if isinstance(m, str):
                mm = m.strip().upper()
                w["mode"] = mm

        return web.json_response({"wars": wars, "count": len(wars)})

    async def api_roster(_request):
        roster = load_roster()
        return web.json_response({"roster": roster, "count": len(roster)})

    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/api/wars", api_wars)
    app.router.add_get("/api/roster", api_roster)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.getenv("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logger.info("HTTP listening on :%s (/, /health, /api/wars, /api/roster)", port)


# ----------------------------
# Roster helpers
# ----------------------------

# (mtime_base, mtime_overrides, mtime_removed, merged_roster)
_roster_cache: Tuple[float, float, float, List[str]] = (0.0, 0.0, 0.0, [])
_roster_write_lock = asyncio.Lock()


def _load_roster_file(path: str) -> List[str]:
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        roster = data.get("roster", []) if isinstance(data, dict) else []
        return [str(x) for x in roster if str(x).strip()]
    except FileNotFoundError:
        return []
    except Exception:
        logger.exception("Failed to read roster file: %s", path)
        return []


def load_roster() -> List[str]:
    """Load effective roster.

    Sources (in order):
      1) roster.json
      2) roster_overrides.json (added via ADDROSTER)
      3) roster_removed.json (removed via REMOVEROSTER)

    The returned list is de-duplicated (case-insensitive) and then filtered by
    removed names.
    """
    global _roster_cache

    try:
        st_base = os.stat(ROSTER_PATH)
        mtime_base = float(st_base.st_mtime)
    except FileNotFoundError:
        mtime_base = 0.0
    except Exception:
        mtime_base = 0.0

    try:
        st_ov = os.stat(ROSTER_OVERRIDES_PATH)
        mtime_ov = float(st_ov.st_mtime)
    except FileNotFoundError:
        mtime_ov = 0.0
    except Exception:
        mtime_ov = 0.0

    try:
        st_rm = os.stat(ROSTER_REMOVED_PATH)
        mtime_rm = float(st_rm.st_mtime)
    except FileNotFoundError:
        mtime_rm = 0.0
    except Exception:
        mtime_rm = 0.0

    if (
        _roster_cache[0] == mtime_base
        and _roster_cache[1] == mtime_ov
        and _roster_cache[2] == mtime_rm
        and _roster_cache[3]
    ):
        return _roster_cache[3]

    base = _load_roster_file(ROSTER_PATH)
    ov = _load_roster_file(ROSTER_OVERRIDES_PATH)
    removed = _load_roster_file(ROSTER_REMOVED_PATH)

    merged: List[str] = []
    seen: set[str] = set()
    for name in (base + ov):
        nm = str(name).strip()
        if not nm:
            continue
        key = nm.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(nm)

    removed_keys = {str(x).strip().casefold() for x in removed if str(x).strip()}
    if removed_keys:
        merged = [nm for nm in merged if nm.casefold() not in removed_keys]

    _roster_cache = (mtime_base, mtime_ov, mtime_rm, merged)
    return merged


def _invalidate_roster_cache() -> None:
    global _roster_cache
    _roster_cache = (0.0, 0.0, 0.0, [])


async def add_to_roster_overrides(names: List[str]) -> List[str]:
    """Add names to roster_overrides.json. Returns the list of actually added names."""
    import json

    clean: List[str] = []
    for n in names:
        nm = str(n).strip()
        if not nm:
            continue
        if len(nm) > 64:
            nm = nm[:64]
        clean.append(nm)
    if not clean:
        return []

    async with _roster_write_lock:
        existing = _load_roster_file(ROSTER_OVERRIDES_PATH)
        existing_keys = {x.casefold() for x in existing}

        added: List[str] = []
        for nm in clean:
            k = nm.casefold()
            if k in existing_keys:
                continue
            existing.append(nm)
            existing_keys.add(k)
            added.append(nm)

        if not added:
            return []

        payload = {"roster": existing}
        tmp_path = ROSTER_OVERRIDES_PATH + ".tmp"
        os.makedirs(os.path.dirname(ROSTER_OVERRIDES_PATH) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, ROSTER_OVERRIDES_PATH)

        _invalidate_roster_cache()
        logger.info("ADDROSTER: added=%s (overrides now=%d)", added, len(existing))
        return added


async def unremove_from_roster(names: List[str]) -> List[str]:
    """Remove given names from roster_removed.json (case-insensitive).

    Returns the list of actually un-removed names (as stored in the file).
    """
    import json

    clean = [str(n).strip()[:64] for n in names if str(n).strip()]
    if not clean:
        return []

    async with _roster_write_lock:
        removed = _load_roster_file(ROSTER_REMOVED_PATH)
        if not removed:
            return []

        target = {c.casefold() for c in clean}
        kept: List[str] = []
        unremoved: List[str] = []
        for nm in removed:
            if nm.casefold() in target:
                unremoved.append(nm)
            else:
                kept.append(nm)

        if not unremoved:
            return []

        payload = {"roster": kept}
        tmp_path = ROSTER_REMOVED_PATH + ".tmp"
        os.makedirs(os.path.dirname(ROSTER_REMOVED_PATH) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, ROSTER_REMOVED_PATH)

        _invalidate_roster_cache()
        logger.info("UNREMOVE: removed from roster_removed=%s (now=%d)", unremoved, len(kept))
        return unremoved


async def remove_from_roster(names: List[str]) -> List[str]:
    """Add names to roster_removed.json.

    Additionally, if a name exists in roster_overrides.json, we remove it from
    overrides to keep the overrides file small.

    Returns the list of names that were newly added to removed.
    """
    import json

    clean = [str(n).strip()[:64] for n in names if str(n).strip()]
    if not clean:
        return []

    async with _roster_write_lock:
        # Resolve names against the effective roster first (case-insensitive),
        # so the removed file keeps the canonical casing.
        effective = load_roster()  # cached; safe
        eff_map = {nm.casefold(): nm for nm in effective}

        removed = _load_roster_file(ROSTER_REMOVED_PATH)
        removed_keys = {nm.casefold() for nm in removed}

        newly_added: List[str] = []
        for raw in clean:
            key = raw.casefold()
            nm = eff_map.get(key, raw)
            k = nm.casefold()
            if k in removed_keys:
                continue
            removed.append(nm)
            removed_keys.add(k)
            newly_added.append(nm)

        if not newly_added:
            return []

        # Remove from overrides (if present)
        overrides = _load_roster_file(ROSTER_OVERRIDES_PATH)
        ov_keep = [nm for nm in overrides if nm.casefold() not in removed_keys]
        if ov_keep != overrides:
            tmp_ov = ROSTER_OVERRIDES_PATH + ".tmp"
            os.makedirs(os.path.dirname(ROSTER_OVERRIDES_PATH) or ".", exist_ok=True)
            with open(tmp_ov, "w", encoding="utf-8") as f:
                json.dump({"roster": ov_keep}, f, ensure_ascii=False, indent=2)
            os.replace(tmp_ov, ROSTER_OVERRIDES_PATH)
            logger.info("REMOVEROSTER: removed %d entries from overrides", len(overrides) - len(ov_keep))

        # Write removed
        tmp_rm = ROSTER_REMOVED_PATH + ".tmp"
        os.makedirs(os.path.dirname(ROSTER_REMOVED_PATH) or ".", exist_ok=True)
        with open(tmp_rm, "w", encoding="utf-8") as f:
            json.dump({"roster": removed}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_rm, ROSTER_REMOVED_PATH)

        _invalidate_roster_cache()
        logger.info("REMOVEROSTER: added=%s (removed now=%d)", newly_added, len(removed))
        return newly_added


def resolve_to_roster(name_raw: str, name_norm_from_model: Optional[str], roster: List[str]) -> Optional[str]:
    """Return a roster name or None (strict roster-only).

    Uses the same candidate logic as the batch resolver (aliases/exact/fuzzy+variants).
    """
    if not roster:
        return None

    roster_lower = {r.lower(): r for r in roster}
    roster_keys = [canonical_key(r) for r in roster]

    tmp = WarLine(rank=0, points=0, name_raw=name_raw, name_norm_model=name_norm_from_model)
    cands = _candidates_for_line(tmp, roster, roster_lower, roster_keys, min_fuzzy_score=88)
    if not cands:
        return None
    # We only take the top candidate; ambiguity is handled better in batch mode.
    return cands[0][1]



# ----------------------------
# War post model + rendering
# ----------------------------


@dataclass
class WarLine:
    rank: int
    points: int
    name_raw: str
    # Optional hint from the vision model; trusted only if it exists in roster.
    name_norm_model: Optional[str] = None

    # Final, roster-only display name; or "UNKNOWN" if we could not resolve.
    name_display: str = ""

    # When UNKNOWN, we keep the raw string for warnings/manual fixes.
    unknown_raw: Optional[str] = None

    # When nick is clean but not in roster, we keep it and warn 'poza rosterem'.
    out_of_roster_raw: Optional[str] = None

@dataclass
class WarPost:
    summary: WarSummary
    expected_max_rank: int
    # Stable identifier for this war list (derived from the Discord message id with screenshots)
    war_id: str = ""

    # Discord message id that contained the screenshots (used for stable ID + audit)
    ref_message_id: Optional[int] = None
    # Jump URL to the screenshot message (handy for debug / traceability)
    ref_jump_url: Optional[str] = None
    # Metadata for auditing
    guild_id: Optional[int] = None
    channel_id: Optional[int] = None
    created_at_ts: Optional[int] = None
    lines_by_rank: Dict[int, WarLine] = field(default_factory=dict)

    def missing_ranks(self) -> List[int]:
        return [r for r in range(1, self.expected_max_rank + 1) if r not in self.lines_by_rank]

    def unknown_ranks(self) -> List[int]:
        return [r for r, ln in self.lines_by_rank.items() if ln.name_display == "UNKNOWN"]


    def out_of_roster_ranks(self) -> List[int]:
        return [r for r, ln in self.lines_by_rank.items() if getattr(ln, 'out_of_roster_raw', None)]

    def total_points_sum(self) -> int:
        """Sum of points for all parsed players in the list.

        Used to validate against the alliance total shown on the war summary screen.
        """
        s = 0
        for ln in self.lines_by_rank.values():
            try:
                s += int(getattr(ln, 'points', 0) or 0)
            except Exception:
                continue
        return int(s)


# message_id -> WarPost
WAR_POSTS: Dict[int, WarPost] = {}

# screenshot_message_id -> (summary, players, expected_max_rank)
# Used to avoid re-parsing the same screenshot message when LISTWAR is called multiple times.
WAR_PARSE_CACHE: Dict[int, Tuple[WarSummary, List[PlayerScore], int]] = {}


def _generate_key_variants(key: str) -> List[str]:
    """Generate a small set of OCR-robust variants for a canonical key.

    IMPORTANT: We generate *combinations* of a few transforms (2 rounds) so we can
    recover cases like stylized "ɭαяσ" -> canonical "lars" -> variants "laro" -> "jaro".
    """
    if not key:
        return []

    # NOTE: We cap the variant set for speed/cost, but some "must-have" variants
    # should never be dropped (they fix recurring stylized/OCR cases).
    MUST_KEEP: set[str] = {
        key,
        "jaro",
        "zawisza",
        "bush22",
        "washi",
        "madevil",
        "thevil",
        "legendarny",
    }

    variants: set[str] = {key}

    def _step(v: str) -> set[str]:
        out: set[str] = set()

        # Collapse repeated letters: leggendarny -> legendarny
        out.add(re.sub(r"(.)\1+", r"\1", v))

        # Common OCR confusions in these fonts
        if "k" in v:
            out.add(v.replace("k", "h"))
        if "h" in v:
            out.add(v.replace("h", "k"))

        # n<->v confusion (Madevil: madenil -> madevil)
        if "n" in v:
            out.add(v.replace("n", "v"))
        if "v" in v:
            out.add(v.replace("v", "n"))

        # g<->w confusion at start (Washi: gashi -> washi)
        if v.startswith("g"):
            out.add("w" + v[1:])
        if v.startswith("w"):
            out.add("g" + v[1:])

        # aw<->iw (Zawisza variants often lose the 'a')
        if "iw" in v:
            out.add(v.replace("iw", "aw"))
        if "aw" in v:
            out.add(v.replace("aw", "iw"))

        # First letter confusions (Jaro case)
        if v.startswith("r"):
            out.add("j" + v[1:])
        if v.startswith("j"):
            out.add("r" + v[1:])
        # Sometimes OCR adds an extra leading 'l' before a real 'j' (e.g. ljao)
        if v.startswith("lj") and len(v) >= 3:
            out.add(v[1:])  # drop the leading 'l'
        if v.startswith("l"):
            out.add("j" + v[1:])

        # Very short / corrupted Jaro reads -> jaro
        # Includes cases like "яaд" -> canonical "rad".
        if v in {
            "rao", "ras", "raco",
            "jao", "jas",
            "lao", "las", "laao", "laas",
            "iaaso", "iaso", "iaao",
            "rars",
            "rad", "raad", "jad", "jaad",
        }:
            out.add("jaro")

        # Last letter confusion i<->a (Washi/Waśka case)
        if v.endswith("i"):
            out.add(v[:-1] + "a")
        if v.endswith("a"):
            out.add(v[:-1] + "i")

        # Stylized 's' used as 'o' at end (ɭαяσ case)
        if v.endswith("s"):
            out.add(v[:-1] + "o")
        if v.endswith("o"):
            out.add(v[:-1] + "s")

        # Legendarny family: egerdarny/leggendarny -> legendarny
        if v.endswith("darny") and v != "legendarny":
            out.add("legendarny")

        # Bush22: common stylized small-caps OCR (ʙʏʜ22 / ʙᴜʜ22 / ʙʏʍ22) -> bush22
        # (Sometimes 's' is dropped entirely -> buh22/buw22)
        if v in {"byh22", "buh22", "byw22", "buw22"}:
            out.add("bush22")

        # Washi: occasionally last 'i' gets dropped -> gash / wash
        if v in {"gash", "wash"}:
            out.add(v + "i")
            out.add("washi")

        # Madevil: missing 'd' or 'l' -> maevi / madevi
        # Some fonts cause "madevil" to become "madevul" (u for i)
        if v in {"maevi", "madevi", "maevil", "madevul", "magasyl", "magasy1"}:
            out.add("madevil")

        # If the string strongly looks like madev(i/u)l, try the i<->u swap only there.
        if v.startswith("madev") and "u" in v:
            out.add(v.replace("u", "i"))
        if v.startswith("madev") and "i" in v:
            out.add(v.replace("i", "u"))

        # Thevil: OCR sometimes reads 'h' as 'f' (tfevil / tfevi)
        if v.startswith("tf") and len(v) > 2:
            out.add("th" + v[2:])
        if v in {"thevi", "theva", "tkevi", "tkeva", "tfevi", "tfeva", "thęvi", "thęva", "tfevil", "tkevil"}:
            out.add("thevil")

        # Zawisza: common shortened forms
        if v in {"zawsza", "zsza", "zsz"}:
            out.add("zawisza")

        # Zawisza: if OCR drops middle, try force canonical when pattern matches
        if v.startswith("z") and v.endswith("sza") and v != "zawisza" and len(v) <= 7:
            out.add("zawisza")

        return {x for x in out if x}

    # Two rounds to allow chaining (e.g. lars -> laro -> jaro)
    for _ in range(2):
        cur = list(variants)
        for v in cur:
            variants.update(_step(v))

        # Cap the set to avoid blowups (roster is small; we don't need many variants).
        # IMPORTANT: do not drop MUST_KEEP variants when trimming.
        CAP = 60
        if len(variants) > CAP:
            keep = {v for v in variants if v in MUST_KEEP}
            # Always keep the original key as well.
            keep.add(key)
            rest = [v for v in variants if v not in keep]
            rest_sorted = sorted(rest, key=lambda x: (abs(len(x) - len(key)), len(x), x))
            variants = keep.union(rest_sorted[: max(0, CAP - len(keep))])

    out = sorted({v for v in variants if v})
    return out


def _candidates_for_line(
    line: WarLine,
    roster: List[str],
    roster_lower: Dict[str, str],
    roster_keys: List[str],
    min_fuzzy_score: int = 88,
) -> List[Tuple[float, str]]:
    """Return sorted candidates (score, roster_name). Higher is better."""
    candidates: Dict[str, float] = {}

    cleaned = normalize_display(line.name_raw)
    target_key = canonical_key(cleaned)
    logger.debug(
        "Roster resolver: rank=%s raw=%r cleaned=%r canonical=%r model_norm=%r",
        getattr(line, "rank", "?"),
        line.name_raw,
        cleaned,
        target_key,
        line.name_norm_model,
    )

    # 1) Model-provided normalized name (only if it's a roster member)
    if line.name_norm_model:
        nn = line.name_norm_model.strip()
        if nn and nn.lower() in roster_lower:
            resolved = roster_lower[nn.lower()]
            candidates[resolved] = 1000.0
            logger.debug("  candidate via model_norm: %s score=1000", resolved)

    # 2) Aliases
    mapped = normalize_with_aliases(line.name_raw, ALIASES_PATH)
    if mapped and mapped.lower() in roster_lower:
        resolved = roster_lower[mapped.lower()]
        candidates[resolved] = max(candidates.get(resolved, 0.0), 950.0)
        logger.debug("  candidate via alias: raw=%r -> %r -> %s score=950", line.name_raw, mapped, resolved)

    # 3) Exact match after cleanup
    if cleaned.lower() in roster_lower:
        resolved = roster_lower[cleaned.lower()]
        candidates[resolved] = max(candidates.get(resolved, 0.0), 925.0)
        logger.debug("  candidate via exact(cleaned): %s score=925", resolved)

    # 4) Fuzzy on canonical keys (+ OCR variants)
    # We use a robust scorer: max(ratio, partial_ratio) so that near-misses like
    # "zisza" -> "zawisza" or "theviet" -> "thevil" still match.
    def _robust_scorer(a: str, b: str, **_kwargs) -> float:
        try:
            return float(max(fuzz.ratio(a, b), fuzz.partial_ratio(a, b)))
        except Exception:
            return float(fuzz.ratio(a, b))
    if target_key:
        base_last = target_key[-1]
        best_by_idx: Dict[int, float] = {}

        variants = _generate_key_variants(target_key)
        logger.debug("  fuzzy canonical variants: %s", variants)

        for k in variants:
            for _rk, score, idx in process.extract(k, roster_keys, scorer=_robust_scorer, limit=6):
                if score is None:
                    continue
                s = float(score)

                # Tie-break bonus: prefer candidates ending with the same last char as the *original* key.
                try:
                    if roster_keys[idx].endswith(base_last):
                        s += 2.0
                except Exception:
                    pass

                prev = best_by_idx.get(idx, 0.0)
                if s > prev:
                    best_by_idx[idx] = s

        for idx, s in best_by_idx.items():
            if s >= float(min_fuzzy_score):
                candidates[roster[idx]] = max(candidates.get(roster[idx], 0.0), s)
                logger.debug("  candidate via fuzzy: %s score=%.1f", roster[idx], s)

    out = [(score, name) for name, score in candidates.items()]
    out.sort(key=lambda x: (-x[0], x[1].lower()))
    if out:
        logger.debug("  candidates sorted: %s", out[:6])
    else:
        logger.debug("  no roster candidates")
    return out


def apply_roster_mapping(lines_by_rank: Dict[int, WarLine], roster: List[str]) -> None:
    """Resolve all lines to roster names with global de-duplication."""
    if not lines_by_rank:
        return

    if not roster:
        for ln in lines_by_rank.values():
            ln.name_display = "UNKNOWN"
            ln.unknown_raw = ln.name_raw
        return

    roster_lower = {r.lower(): r for r in roster}
    roster_keys = [canonical_key(r) for r in roster]
    roster_exact = {normalize_display(r): r for r in roster}

    cand_by_rank: Dict[int, List[Tuple[float, str]]] = {}
    for r, ln in lines_by_rank.items():
        cand_by_rank[r] = _candidates_for_line(ln, roster, roster_lower, roster_keys, min_fuzzy_score=88)

    assigned: Dict[int, str] = {}
    used: set[str] = set()
    remaining = set(cand_by_rank.keys())

    # Phase 0: lock exact-case matches (prevents random order issues with de-dup)
    for r, ln in lines_by_rank.items():
        cleaned = normalize_display(ln.name_raw)
        exact = roster_exact.get(cleaned)
        if exact and exact not in used:
            assigned[r] = exact
            used.add(exact)
            remaining.discard(r)
            logger.debug("Pre-assign exact-case: rank %s cleaned=%r -> %s", r, cleaned, exact)

    # Greedy max-first assignment to avoid duplicates (Washi vs Waśka etc.)
    while remaining:
        best_rank: Optional[int] = None
        best_score: float = -1.0

        for r in remaining:
            cands = cand_by_rank.get(r) or []
            if not cands:
                continue
            score, _name = cands[0]
            if score > best_score or (score == best_score and (best_rank is None or r < best_rank)):
                best_score = score
                best_rank = r

        if best_rank is None:
            break

        cands = cand_by_rank.get(best_rank) or []
        if not cands:
            remaining.remove(best_rank)
            continue

        score, name = cands[0]
        logger.debug("Assign pass: best_rank=%s candidate=%s score=%.1f", best_rank, name, score)
        if name not in used:
            assigned[best_rank] = name
            used.add(name)
            remaining.remove(best_rank)
            logger.debug("  assigned rank %s -> %s", best_rank, name)
        else:
            logger.debug("  candidate %s already used -> trying next candidate for rank %s", name, best_rank)
            cand_by_rank[best_rank] = cands[1:]
            if not cand_by_rank[best_rank]:
                remaining.remove(best_rank)
                logger.debug("  no more candidates for rank %s -> UNKNOWN", best_rank)

    for r, ln in lines_by_rank.items():
        if r in assigned:
            ln.name_display = assigned[r]
            ln.unknown_raw = None
            ln.out_of_roster_raw = None
            logger.debug("Final roster mapping: rank %s raw=%r -> %s", r, ln.name_raw, ln.name_display)
        else:
            cleaned = normalize_display(ln.name_raw)
            if is_clean_display(ln.name_raw):
                if cleaned in roster_exact:
                    # Clean nick that *is* in roster, but we couldn't assign it uniquely.
                    # Treat as UNKNOWN to force manual correction instead of misleading "poza rosterem".
                    ln.name_display = "UNKNOWN"
                    ln.unknown_raw = cleaned
                    ln.out_of_roster_raw = None
                    logger.debug(
                        "Final roster mapping: rank %s raw=%r -> UNKNOWN_DUPLICATE_IN_ROSTER(%s)",
                        r,
                        ln.name_raw,
                        cleaned,
                    )
                else:
                    # Show clean nick even if it's not in roster (but warn).
                    ln.name_display = cleaned
                    ln.unknown_raw = None
                    ln.out_of_roster_raw = cleaned
                    logger.debug(
                        "Final roster mapping: rank %s raw=%r -> OUT_OF_ROSTER(%s)",
                        r,
                        ln.name_raw,
                        cleaned,
                    )
            else:
                ln.name_display = "UNKNOWN"
                ln.unknown_raw = cleaned or ln.name_raw
                ln.out_of_roster_raw = None
                logger.debug("Final roster mapping: rank %s raw=%r -> UNKNOWN", r, ln.name_raw)


def build_post(summary: WarSummary, players: List[PlayerScore], expected_max_rank: Optional[int]) -> WarPost:
    roster = load_roster()

    logger.info(
        "Build post: players=%d expected_max_rank=%s roster=%d",
        len(players or []),
        expected_max_rank,
        len(roster or []),
    )

    by_rank: Dict[int, WarLine] = {}

    def better(a: WarLine, b: WarLine) -> WarLine:
        a_score = (100 if a.name_norm_model else 0) + min(len(a.name_raw or ""), 40) + int(a.points)
        b_score = (100 if b.name_norm_model else 0) + min(len(b.name_raw or ""), 40) + int(b.points)
        return a if a_score >= b_score else b

    for p in players:
        if not isinstance(p.rank, int) or p.rank <= 0 or p.rank > 200:
            continue
        if not isinstance(p.points, int) or p.points < 0 or p.points > 9999:
            continue

        ln = WarLine(
            rank=p.rank,
            points=p.points,
            name_raw=p.name_raw,
            name_norm_model=p.name_norm,
        )

        if p.rank in by_rank:
            prev = by_rank[p.rank]
            chosen = better(prev, ln)
            by_rank[p.rank] = chosen
            logger.debug(
                "Duplicate rank=%d: prev(raw=%r pts=%d model_norm=%r) vs new(raw=%r pts=%d model_norm=%r) -> keep(raw=%r pts=%d)",
                p.rank,
                prev.name_raw,
                prev.points,
                prev.name_norm_model,
                ln.name_raw,
                ln.points,
                ln.name_norm_model,
                chosen.name_raw,
                chosen.points,
            )
        else:
            by_rank[p.rank] = ln

    mx = expected_max_rank if expected_max_rank and expected_max_rank > 0 else None
    if not mx:
        mx = max(by_rank.keys()) if by_rank else 0
    mx = int(mx) if mx else 0

    post = WarPost(summary=summary, expected_max_rank=mx, lines_by_rank=by_rank)

    logger.info("Post ranks: unique=%d max=%d", len(post.lines_by_rank), post.expected_max_rank)

    apply_roster_mapping(post.lines_by_rank, roster)

    missing = post.missing_ranks()
    unknown = post.unknown_ranks()
    out_of_roster = post.out_of_roster_ranks()
    if missing or unknown or out_of_roster:
        logger.warning("Post validation: missing=%s unknown=%s out_of_roster=%s", missing, unknown, out_of_roster)
    else:
        logger.info("Post validation: OK")

    return post


def render_post(post: WarPost) -> str:
    s = post.summary

    # Score diff: positive => win, negative => loss. Render wants a visible sign.
    diff = int(s.our_score) - int(s.opponent_score)
    if diff > 0:
        badge = "🟢"
    elif diff < 0:
        badge = "🔴"
    else:
        badge = "⚪"

    header = (
        f"**Wojna zakończona: {badge} {s.result} ({diff:+d})**\n"
        f"**{s.our_alliance}** {s.our_score} — {s.opponent_score} **{s.opponent_alliance}**\n"
    )
    if s.war_mode:
        mode_disp = s.war_mode.strip().upper() if isinstance(s.war_mode, str) else str(s.war_mode)
        header += f"Tryb: **{mode_disp}**" + (" (BETA)\n" if s.beta_badge else "\n")
    if post.war_id:
        header += f"ID: `{post.war_id}`\n"
    header += "\n"

    lines_out: List[str] = []
    for r in sorted(post.lines_by_rank.keys()):
        ln = post.lines_by_rank[r]
        name_disp = ln.name_display
        # Make UNKNOWN lines self-explanatory in the main list as well.
        if name_disp == "UNKNOWN":
            raw = (ln.unknown_raw or ln.name_raw or "").strip()
            raw = raw.replace("`", "'")
            if raw:
                name_disp = f'{name_disp} _(nierozpoznany: "{raw}")_'
        if getattr(ln, 'out_of_roster_raw', None):
            name_disp = f"{name_disp} _(poza rosterem)_"
        lines_out.append(f"[{ln.rank:02d}] {name_disp} — {ln.points}")

    msg = header + "\n".join(lines_out)

    missing = post.missing_ranks()
    unknown = post.unknown_ranks()
    out_of_roster = post.out_of_roster_ranks()
    points_sum = post.total_points_sum()
    try:
        alliance_total = int(s.our_score)
    except Exception:
        alliance_total = 0
    sum_mismatch = bool(alliance_total and points_sum != alliance_total)

    if missing or unknown or out_of_roster or sum_mismatch:
        warn_lines: List[str] = ["", "⚠️ **Wymagane poprawki**"]
        if missing:
            warn_lines.append("• Brakujące pozycje: " + ", ".join(str(x) for x in missing))
        if unknown:
            # include raw to make manual correction easier
            parts = []
            for r in sorted(unknown):
                raw = post.lines_by_rank[r].unknown_raw or ""
                raw = raw.replace("`", "'")
                parts.append(f"{r}(\"{raw}\")")
            warn_lines.append("• UNKNOWN (nieznane nicki): " + ", ".join(parts))
        if out_of_roster:
            parts2 = []
            for r in sorted(out_of_roster):
                raw = post.lines_by_rank[r].out_of_roster_raw or post.lines_by_rank[r].name_display or ""
                raw = raw.replace("`", "'")
                parts2.append(f"{r}(\"{raw}\")")
            warn_lines.append("• Poza rosterem (gracz nie widnieje w rosterze): " + ", ".join(parts2))

        if sum_mismatch:
            diff_pts = points_sum - alliance_total
            warn_lines.append("• Suma punktów graczy = %d, suma sojuszu (z podsumowania) = %d (różnica: %+d)" % (points_sum, alliance_total, diff_pts))

        warn_lines.append("")
        warn_lines.append(
            "Reply na tę wiadomość w formacie: `23 ropuch13 250` **albo** `23 Legendarny` (bez punktów = zachowaj istniejące)."
        )
        warn_lines.append(
            "Możesz też dodać nowy nick do rosteru: `ADDROSTER Krati` (albo wiele: `ADDROSTER Krati, NowyGracz`)."
        )
        warn_lines.append(
            "Możesz też usunąć nick z rosteru (rotacja): `REMOVEROSTER Krati` (albo wiele: `REMOVEROSTER Krati, NowyGracz`)."
        )
        warn_lines.append("Po przetworzeniu poprawki bot usunie Twoją wiadomość.")
        msg += "\n" + "\n".join(warn_lines)

    msg += "\n\n" + "✅ Gdy lista jest poprawna: odpowiedz na tę wiadomość komendą `ADDWAR`, aby dodać wojnę do strony."

    return msg


# ---------------- Web dashboard store helpers ----------------

def _post_to_store_record(
    post: WarPost,
    ref_msg: Optional[discord.Message] = None,
    bot_msg: Optional[discord.Message] = None,
    *,
    store_status: Optional[str] = None,
    confirmed_at_ts: Optional[int] = None,
    confirmed_by_user_id: Optional[int] = None,
    confirmed_by_user_tag: Optional[str] = None,
) -> dict:
    """Convert a WarPost into a JSON record for wars_store.json.

    Status convention:
      - draft: generated by LISTWAR (not shown on the website)
      - confirmed: approved by ADDWAR (shown on the website)
    """
    s = post.summary
    diff = int(s.our_score) - int(s.opponent_score)
    points_sum = post.total_points_sum()
    try:
        alliance_total = int(s.our_score)
    except Exception:
        alliance_total = 0
    points_sum_matches_alliance = bool(alliance_total and points_sum == alliance_total)
    now_ts = int(time.time())

    ref_id = str(ref_msg.id) if ref_msg is not None else (str(post.ref_message_id) if post.ref_message_id else None)
    created_ts = now_ts
    created_iso = None
    ref_jump = None
    guild_id = None
    channel_id = None
    if ref_msg is not None:
        try:
            created_ts = int(ref_msg.created_at.timestamp())
            created_iso = ref_msg.created_at.isoformat()
        except Exception:
            created_ts = now_ts
            created_iso = None
        try:
            ref_jump = ref_msg.jump_url
        except Exception:
            ref_jump = None
        try:
            guild_id = int(ref_msg.guild.id) if getattr(ref_msg, "guild", None) else None
        except Exception:
            guild_id = None
        try:
            channel_id = int(ref_msg.channel.id)
        except Exception:
            channel_id = None
    else:
        created_ts = int(post.created_at_ts or now_ts)
        ref_jump = post.ref_jump_url
        guild_id = post.guild_id
        channel_id = post.channel_id

    bot_id = None
    bot_jump = None
    if bot_msg is not None:
        bot_id = str(bot_msg.id)
        try:
            bot_jump = bot_msg.jump_url
        except Exception:
            bot_jump = None

    players: List[dict] = []
    unknown: List[dict] = []
    out_of_roster: List[dict] = []
    missing = post.missing_ranks()

    for rank in sorted(post.lines_by_rank.keys()):
        ln = post.lines_by_rank[rank]
        row_status = "ok"
        if (ln.name_display or "").strip().upper() == "UNKNOWN":
            row_status = "unknown"
            unknown.append({"rank": rank, "raw": ln.name_raw})
        elif ln.out_of_roster_raw:
            row_status = "out_of_roster"
            out_of_roster.append({"rank": rank, "raw": ln.name_raw})

        players.append({
            "rank": rank,
            "name": ln.name_display,
            "raw": ln.name_raw,
            "points": int(ln.points),
            "status": row_status,
        })

    unknown_count = len(unknown)
    out_of_roster_count = len(out_of_roster)
    missing_count = len(missing)

    mode_norm: Optional[str] = None
    if s.war_mode and isinstance(s.war_mode, str):
        mode_norm = s.war_mode.strip().upper()
        if not mode_norm:
            mode_norm = None

    rec: Dict[str, Any] = {
        "war_id": post.war_id,
        "ref_message_id": ref_id,
        "ref_jump_url": ref_jump,
        "created_at_ts": created_ts,
        "created_at_iso": created_iso,
        "updated_at_ts": now_ts,
        "result": s.result,
        "our_score": int(s.our_score),
        "our_alliance": s.our_alliance,
        "opponent_alliance": s.opponent_alliance,
        "opponent_score": int(s.opponent_score),
        # Backwards compatible aliases (older UI keys)
        "enemy_score": int(s.opponent_score),
        "diff": int(diff),
        "players_points_sum": int(points_sum),
        "players_points_sum_matches_alliance": bool(points_sum_matches_alliance),
        "mode": mode_norm,
        "players": players,
        "unknown": unknown,
        "out_of_roster": out_of_roster,
        "missing": [{"rank": r} for r in missing],
        "unknown_count": unknown_count,
        "out_of_roster_count": out_of_roster_count,
        "missing_count": missing_count,
        "bot_message_id": bot_id,
        "bot_jump_url": bot_jump,
        "guild_id": guild_id,
        "channel_id": channel_id,
        "status": (store_status or "confirmed"),
        "confirmed_at_ts": confirmed_at_ts,
        "confirmed_by_user_id": confirmed_by_user_id,
        "confirmed_by_user_tag": confirmed_by_user_tag,
    }
    return rec


async def _war_store_upsert_from_post(post: WarPost, ref_msg: Optional[discord.Message] = None, bot_msg: Optional[discord.Message] = None) -> None:
    rec = _post_to_store_record(post, ref_msg=ref_msg, bot_msg=bot_msg)
    async with WAR_STORE_LOCK:
        await asyncio.to_thread(WAR_STORE.upsert_war, post.war_id, rec)


def chunk_message(msg: str, limit: int = 1900) -> List[str]:
    chunks: List[str] = []
    while len(msg) > limit:
        cut = msg.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(msg[:cut])
        msg = msg[cut:].lstrip("\n")
    chunks.append(msg)
    return chunks


def _set_current_status_block(text: str, status_line: str) -> str:
    """Ensure LISTWAR message ends with a single '**Aktualny status:** ...' line."""
    base = (text or "").rstrip()
    # Remove any previous status block (best-effort) and re-append as the final line.
    base = re.sub(r"\n\n\*\*Aktualny status:\*\*.*\Z", "", base, flags=re.DOTALL)
    base = base.rstrip()
    return base + "\n\n**Aktualny status:** " + status_line


async def _try_update_listwar_status_message(
    discord_client: discord.Client,
    war_id: str,
    status_line: str,
    ref_msg: Optional[discord.Message] = None,
) -> bool:
    """Update the LISTWAR-generated bot message with current status (no spam in channel).

    Returns True if we managed to update a message, False otherwise.
    """
    # If we already have the referenced bot message, update it directly.
    if ref_msg is not None:
        try:
            new_content = _set_current_status_block(ref_msg.content or "", status_line)
            if new_content != (ref_msg.content or ""):
                await ref_msg.edit(content=new_content)
            return True
        except Exception:
            logger.debug("Failed to edit LISTWAR message for status update", exc_info=True)

    # Otherwise, try to locate the original LISTWAR message from store metadata.
    try:
        async with WAR_STORE_LOCK:
            rec = await asyncio.to_thread(WAR_STORE.get_war, war_id)
        if not isinstance(rec, dict):
            return False
        msg_id = rec.get("ref_message_id") or rec.get("bot_message_id")
        ch_id = rec.get("channel_id")
        if not msg_id or not ch_id:
            return False

        channel = discord_client.get_channel(int(ch_id))
        if channel is None:
            try:
                channel = await discord_client.fetch_channel(int(ch_id))
            except Exception:
                channel = None
        if channel is None:
            return False

        try:
            m = await channel.fetch_message(int(msg_id))
        except Exception:
            return False

        new_content = _set_current_status_block(m.content or "", status_line)
        if new_content != (m.content or ""):
            await m.edit(content=new_content)
        return True
    except Exception:
        logger.debug("Failed to locate/update LISTWAR message for war_id=%s", war_id, exc_info=True)
        return False


def is_image(att: discord.Attachment) -> bool:
    if att.content_type:
        return att.content_type.startswith("image/")
    return att.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))


async def parse_images_in_thread(images: List[bytes], trace_id: str):
    # Pass trace_id explicitly so the OpenAI thread gets the same id.
    return await asyncio.to_thread(parse_war_from_images, images, OPENAI_MODEL, trace_id=trace_id)


def _parse_manual_corrections(text: str) -> List[Tuple[int, str, Optional[int]]]:
    """Parse manual correction lines.

    Supported formats (one per line, can be many lines):
      - "23 ropuch13 250"  -> set rank=23, nick=ropuch13, points=250
      - "23 Legendarny"   -> set rank=23, nick=Legendarny, keep existing points (if present)
    """
    out: List[Tuple[int, str, Optional[int]]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Normalize separators
        line = line.replace("—", " ").replace("-", " ").replace(":", " ")
        toks = [t for t in line.split() if t]
        if len(toks) < 2:
            continue

        # Rank
        t0 = toks[0].strip("[](){}")
        try:
            rank = int(t0)
        except ValueError:
            continue

        # Points (optional)
        points: Optional[int] = None
        t_last = toks[-1].strip("[](){}")
        if len(toks) >= 3:
            try:
                points = int(t_last)
            except ValueError:
                points = None

        if points is None:
            nick = " ".join(toks[1:]).strip()
        else:
            nick = " ".join(toks[1:-1]).strip()
        if not nick:
            continue

        if rank <= 0 or rank > 200:
            continue
        if points is not None:
            if points < 0 or points > 9999:
                continue

        out.append((rank, nick, points))
    return out


def _parse_addroster(text: str) -> List[str]:
    """Parse an ADDROSTER command.

    Supported examples:
      - "ADDROSTER Krati"
      - "ADDROSTER: Krati, NowyGracz"
      - "ADDROSTER\nKrati\nNowyGracz"
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    first = lines[0]
    m = re.match(r"^\s*ADDROSTER\b\s*[:\-]?\s*(.*)$", first, flags=re.IGNORECASE)
    if not m:
        return []

    rest_first = (m.group(1) or "").strip()
    rest_lines: List[str] = []
    if rest_first:
        rest_lines.append(rest_first)
    if len(lines) > 1:
        rest_lines.extend(lines[1:])

    raw_joined = "\n".join(rest_lines).strip()
    if not raw_joined:
        return []

    # Split by commas/newlines, keep internal spaces.
    parts: List[str] = []
    for chunk in raw_joined.split("\n"):
        for p in chunk.split(","):
            nm = p.strip()
            if nm:
                parts.append(nm)
    return parts


async def try_apply_addroster_command(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """Handle reply command: ADDROSTER <name> [...].

    - Updates roster_overrides.json (persistent)
    - If the replied-to message is a known war post in this runtime, re-renders it.
    - Deletes the user's command message (best effort).
    """
    if not message.reference or not message.reference.message_id:
        return False

    names = _parse_addroster(message.content or "")
    if not names:
        return False

    # Fetch referenced message
    ref_msg: Optional[discord.Message] = None
    if isinstance(message.reference.resolved, discord.Message):
        ref_msg = message.reference.resolved
    else:
        try:
            ref_msg = await message.channel.fetch_message(message.reference.message_id)
        except Exception:
            ref_msg = None
    if not ref_msg:
        return False
    if not discord_client.user:
        return False
    if ref_msg.author.id != discord_client.user.id:
        return False

    # If a name was previously removed via REMOVEROSTER, allow ADDROSTER to
    # bring it back by deleting it from roster_removed.json.
    unremoved = await unremove_from_roster(names)
    added = await add_to_roster_overrides(names)
    if added or unremoved:
        try:
            await _persist_progress_to_discord(discord_client, what="roster")
        except Exception:
            pass
    if not (added or unremoved):
        # Nothing new -> treat as handled to avoid noise.
        try:
            await ref_msg.add_reaction("ℹ️")
        except Exception:
            pass
        try:
            await message.delete()
        except Exception:
            pass
        return True

    # If we can, re-render the referenced war post.
    post = WAR_POSTS.get(ref_msg.id)
    if post:
        roster = load_roster()
        apply_roster_mapping(post.lines_by_rank, roster)
        new_content = render_post(post)
        parts = chunk_message(new_content)
        if len(parts) != 1:
            new_content = parts[0]
        try:
            await ref_msg.edit(content=new_content)
        except Exception:
            logger.exception("ADDROSTER: failed to edit war post %s", ref_msg.id)

    # React to confirm
    try:
        await ref_msg.add_reaction("➕")
    except Exception:
        pass

    # Delete user's command message
    try:
        await message.delete()
        logger.info("ADDROSTER: deleted user msg %s (added=%s unremoved=%s)", message.id, added, unremoved)
    except Exception:
        logger.warning("ADDROSTER: could not delete user msg %s (missing permissions?)", message.id)
        pass
    return True


def _parse_removeroster(text: str) -> List[str]:
    """Parse: REMOVEROSTER <name1> [, name2 ...]."""
    if not text:
        return []
    t = text.strip()
    if not t.upper().startswith("REMOVEROSTER"):
        return []
    rest = t[len("REMOVEROSTER"):].strip()
    if not rest:
        return []
    parts = [p.strip() for p in re.split(r"[,\n]+", rest) if p.strip()]
    out: List[str] = []
    for p in parts:
        if len(p) > 64:
            p = p[:64]
        out.append(p)
    return out


async def try_apply_removeroster_command(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """Handle reply command: REMOVEROSTER <name> [...].

    - Updates roster_removed.json (persistent)
    - If the replied-to message is a known war post in this runtime, re-renders it.
    - Deletes the user's command message (best effort).
    """
    if not message.reference or not message.reference.message_id:
        return False

    names = _parse_removeroster(message.content or "")
    if not names:
        return False

    # Fetch referenced message
    ref_msg: Optional[discord.Message] = None
    if isinstance(message.reference.resolved, discord.Message):
        ref_msg = message.reference.resolved
    else:
        try:
            ref_msg = await message.channel.fetch_message(message.reference.message_id)
        except Exception:
            ref_msg = None
    if not ref_msg:
        return False
    if not discord_client.user:
        return False
    if ref_msg.author.id != discord_client.user.id:
        return False

    removed = await remove_from_roster(names)
    if removed:
        try:
            await _persist_progress_to_discord(discord_client, what="roster")
        except Exception:
            pass
    if removed:
        try:
            await _persist_progress_to_discord(discord_client, what="roster")
        except Exception:
            pass
    if not removed:
        # Nothing changed -> treat as handled to avoid noise.
        try:
            await ref_msg.add_reaction("ℹ️")
        except Exception:
            pass
        try:
            await message.delete()
        except Exception:
            pass
        return True

    # If we can, re-render the referenced war post.
    post = WAR_POSTS.get(ref_msg.id)
    if post:
        roster = load_roster()
        apply_roster_mapping(post.lines_by_rank, roster)
        new_content = render_post(post)
        parts = chunk_message(new_content)
        if len(parts) != 1:
            new_content = parts[0]
        try:
            await ref_msg.edit(content=new_content)
        except Exception:
            logger.exception("REMOVEROSTER: failed to edit war post %s", ref_msg.id)

    # React to confirm
    try:
        await ref_msg.add_reaction("➖")
    except Exception:
        pass

    # Delete user's command message
    try:
        await message.delete()
        logger.info("REMOVEROSTER: deleted user msg %s (removed=%s)", message.id, removed)
    except Exception:
        logger.warning("REMOVEROSTER: could not delete user msg %s (missing permissions?)", message.id)
        pass
    return True


async def try_apply_addroster_channel_command(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """Handle channel command (not reply): ADDROSTER <name> [, ...].

    - Updates roster_overrides.json (persistent)
    - Best-effort deletes the user's command message
    """
    if message.reference and message.reference.message_id:
        return False

    names = _parse_addroster(message.content or "")
    if not names:
        return False

    # If a name was previously removed via REMOVEROSTER, allow ADDROSTER to bring it back.
    unremoved = await unremove_from_roster(names)
    added = await add_to_roster_overrides(names)
    if added or unremoved:
        try:
            await _persist_progress_to_discord(discord_client, what="roster")
        except Exception:
            pass

    if added or unremoved:
        parts = []
        if added:
            parts.append(f"➕ Dodano do rosteru: {', '.join(added)}")
        if unremoved:
            parts.append(f"♻️ Przywrócono (usunięte wcześniej): {', '.join(unremoved)}")
        resp = "\n".join(parts)
    else:
        resp = "ℹ️ Nic nie zmieniono (nicki już były w rosterze)."

    try:
        await message.channel.send(resp)
    except Exception:
        logger.exception("ADDROSTER(channel): failed to send confirmation")

    try:
        await message.delete()
    except Exception:
        pass

    return True


async def try_apply_removeroster_channel_command(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """Handle channel command (not reply): REMOVEROSTER <name> [, ...].

    - Updates roster_removed.json (persistent)
    - Best-effort deletes the user's command message
    """
    if message.reference and message.reference.message_id:
        return False

    names = _parse_removeroster(message.content or "")
    if not names:
        return False

    removed = await remove_from_roster(names)
    if removed:
        try:
            await _persist_progress_to_discord(discord_client, what="roster")
        except Exception:
            pass
        resp = f"➖ Usunięto z rosteru: {', '.join(removed)}"
    else:
        resp = "ℹ️ Nic nie zmieniono (nicki już były usunięte lub nie znaleziono)."

    try:
        await message.channel.send(resp)
    except Exception:
        logger.exception("REMOVEROSTER(channel): failed to send confirmation")

    try:
        await message.delete()
    except Exception:
        pass

    return True


async def try_apply_currentroster_channel_command(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """Handle channel command: CURRENTROSTER.

    Sends the current effective roster (base + overrides - removed).
    """
    if message.reference and message.reference.message_id:
        return False

    text = (message.content or "").strip()
    if not re.match(r"^CURRENTROSTER\b", text, flags=re.IGNORECASE):
        return False

    base = _load_roster_file(ROSTER_PATH)
    overrides = _load_roster_file(ROSTER_OVERRIDES_PATH)
    removed = _load_roster_file(ROSTER_REMOVED_PATH)
    roster = load_roster()

    header = f"📋 Aktualny roster ({len(roster)}): base={len(base)}, overrides={len(overrides)}, removed={len(removed)}"
    body = "\n".join(roster) if roster else "(pusto)"
    out = header + "\n```\n" + body + "\n```"

    try:
        for part in chunk_message(out):
            await message.channel.send(part)
    except Exception:
        logger.exception("CURRENTROSTER: failed to send")

    try:
        await message.delete()
    except Exception:
        pass

    return True


def _extract_war_id_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\bID:\s*`([^`]+)`", text)
    if m:
        return m.group(1).strip()
    # Allow suffixes like WAR-ABCDEF-2
    m2 = re.search(r"\b(WAR-[A-Z0-9-]+)\b", text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


async def try_apply_help_channel_command(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """Handle channel command: HELP.

    Prints all available commands and short usage.
    """
    if message.reference and message.reference.message_id:
        return False

    text = (message.content or "").strip()
    if not re.match(r"^HELP\b", text, flags=re.IGNORECASE):
        return False

    auto_del = max(5, int(HELP_AUTO_DELETE_SEC))
    help_text = (
        "🆘 **Dostępne komendy**\n"
        f"\n⏳ _Ta wiadomość zostanie usunięta za {auto_del} sekund._\n"
        "\n"
        "**Wojny**\n"
        "• `LISTWAR` — *(reply)* na wiadomość z **2 screenami** z wojny. Bot wyśle listę wyników (DRAFT).\n"
        "• `23 Nick 250` — *(reply)* na wiadomość bota z listą: popraw nick i punkty na pozycji 23.\n"
        "• `23 Nick` — *(reply)* popraw tylko nick (punkty zostają).\n"
        "• `ADDWAR` — *(reply)* na poprawną listę bota: **zatwierdza** i dodaje wojnę do strony (CONFIRMED).\n"
        "• `UNLISTWAR <ID>` — usuwa wojnę w statusie DRAFT (żeby zrobić LISTWAR ponownie i dostać nowe ID).\n"
        "• `REMOVEWAR <ID>` — usuwa wojnę ze strony (ID z nagłówka listy).\n"
        "\n"
        "**Roster**\n"
        "• `ADDROSTER <nick1, nick2>` — dodaje do rosteru (stałe).\n"
        "• `REMOVEROSTER <nick1, nick2>` — usuwa z rosteru (rotacja).\n"
        "• `CURRENTROSTER` — wyświetla aktualny roster (base + overrides - removed).\n"
        "\n"
        "**Info**\n"
        "• `HELP` — ta wiadomość.\n"
        "\n"
        "(Bot może usuwać Twoje wiadomości-komendy, żeby kanał był czysty.)"
    )

    sent_msgs: List[discord.Message] = []
    try:
        for part in chunk_message(help_text):
            sent = await message.channel.send(part)
            sent_msgs.append(sent)
    except Exception:
        logger.exception("HELP: failed to send")

    # Auto-delete help messages to keep channel clean.
    for sm in sent_msgs:
        try:
            asyncio.create_task(delete_message_later(sm, auto_del))
        except Exception:
            pass

    try:
        await message.delete()
    except Exception:
        pass
    return True


async def try_apply_removewar_channel_command(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """Handle channel command: REMOVEWAR <WAR-ID>."""
    if message.reference and message.reference.message_id:
        return False

    text = (message.content or "").strip()
    m = re.match(r"^REMOVEWAR\s+(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return False
    war_id = m.group(1).strip()
    if not war_id:
        return False
    # Normalize common formats
    war_id = war_id.replace("`", "").strip()

    try:
        async with WAR_STORE_LOCK:
            existed = await asyncio.to_thread(WAR_STORE.delete_war, war_id)
        if existed:
            try:
                await _persist_progress_to_discord(discord_client, what="wars")
            except Exception:
                pass
        if existed:
            # Prefer updating the original LISTWAR message with a final, current status.
            updated = await _try_update_listwar_status_message(
                discord_client,
                war_id=war_id,
                status_line="🗑️ Usunięto ze strony",
                ref_msg=None,
            )
            if not updated:
                await message.channel.send(f"🗑️ Usunięto wojnę z listy: `{war_id}`")
        else:
            await message.channel.send(f"ℹ️ Nie znaleziono wojny: `{war_id}`")
    except Exception:
        logger.exception("REMOVEWAR: failed")

    try:
        await message.delete()
    except Exception:
        pass
    return True


async def try_apply_unlistwar_channel_command(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """Handle channel command: UNLISTWAR <WAR-ID>.

    Removes a war in status DRAFT only. This allows running LISTWAR again on the
    same screenshots and receiving a new war ID.
    """
    if message.reference and message.reference.message_id:
        return False

    text = (message.content or "").strip()
    m = re.match(r"^UNLISTWAR\s+(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return False
    war_id = (m.group(1) or "").replace("`", "").strip()
    if not war_id:
        return False

    rec: Optional[Dict[str, object]] = None
    try:
        async with WAR_STORE_LOCK:
            rec = await asyncio.to_thread(WAR_STORE.get_war, war_id)

        if not isinstance(rec, dict):
            await message.channel.send(f"ℹ️ Nie znaleziono wojny: `{war_id}`")
            return True

        status = str(rec.get("status") or "").lower()
        if status != "draft":
            if status == "confirmed":
                await message.channel.send(
                    f"ℹ️ `{war_id}` jest już zatwierdzona (CONFIRMED). Użyj `REMOVEWAR {war_id}` jeśli chcesz ją usunąć ze strony."
                )
            else:
                await message.channel.send(
                    f"ℹ️ `UNLISTWAR` działa tylko dla wojen w statusie DRAFT. Status `{war_id}`: `{status or 'unknown'}`"
                )
            return True

        # Best-effort delete the LISTWAR bot message (if we have metadata).
        try:
            bot_msg_id = int(rec.get("bot_message_id") or 0)
            ch_id = int(rec.get("channel_id") or 0)
            if bot_msg_id and ch_id:
                ch = discord_client.get_channel(ch_id)
                if ch is None:
                    try:
                        ch = await discord_client.fetch_channel(ch_id)
                    except Exception:
                        ch = None
                if isinstance(ch, (discord.TextChannel, discord.Thread)):
                    try:
                        bm = await ch.fetch_message(bot_msg_id)
                        await bm.delete()
                    except Exception:
                        pass
        except Exception:
            pass

        # Clear parse cache for this source screenshot message (so next LISTWAR re-parses if desired).
        try:
            ref_mid = int(rec.get("ref_message_id") or 0)
            if ref_mid:
                WAR_PARSE_CACHE.pop(ref_mid, None)
        except Exception:
            pass

        # Remove draft from store.
        async with WAR_STORE_LOCK:
            existed = await asyncio.to_thread(WAR_STORE.delete_war, war_id)
        if existed:
            try:
                await _persist_progress_to_discord(discord_client, what="wars")
            except Exception:
                pass
            await message.channel.send(f"🧹 Unlisted DRAFT: `{war_id}` (możesz zrobić LISTWAR ponownie, dostaniesz nowe ID)")
        else:
            await message.channel.send(f"ℹ️ Nie znaleziono wojny: `{war_id}`")

    except Exception:
        logger.exception("UNLISTWAR: failed")

    try:
        await message.delete()
    except Exception:
        pass
    return True


async def try_apply_addwar_command(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """Handle reply command: ADDWAR.

    User replies to the bot's war list message to confirm it and add it to the website.
    """
    if not message.reference or not message.reference.message_id:
        return False
    text = (message.content or "").strip()
    if not re.match(r"^ADDWAR\b", text, flags=re.IGNORECASE):
        return False

    # Fetch referenced message (bot's war list)
    ref_msg: Optional[discord.Message] = None
    if isinstance(message.reference.resolved, discord.Message):
        ref_msg = message.reference.resolved
    else:
        try:
            ref_msg = await message.channel.fetch_message(message.reference.message_id)
        except Exception:
            ref_msg = None
    if not ref_msg:
        return False
    if not discord_client.user or ref_msg.author.id != discord_client.user.id:
        return False

    post = WAR_POSTS.get(ref_msg.id)
    war_id = post.war_id if post else _extract_war_id_from_text(ref_msg.content or "")
    if not war_id:
        await message.channel.send("Nie mogę znaleźć ID wojny w tej wiadomości. Zrób LISTWAR ponownie.")
        return True

    # Load existing store record (draft or confirmed)
    async with WAR_STORE_LOCK:
        existing = await asyncio.to_thread(WAR_STORE.get_war, war_id)
    existing_status = str((existing or {}).get("status") or "").lower()
    if existing_status == "confirmed":
        try:
            await ref_msg.add_reaction("ℹ️")
        except Exception:
            pass
        try:
            await _try_update_listwar_status_message(
                discord_client,
                war_id=war_id,
                status_line="✅ Dodano do strony",
                ref_msg=ref_msg,
            )
        except Exception:
            pass
        try:
            await message.delete()
        except Exception:
            pass
        return True

    # Validate: no missing ranks and no UNKNOWN entries.
    if post is not None:
        missing = post.missing_ranks()
        unknown = post.unknown_ranks()
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"brakujące pozycje: {', '.join(map(str, missing))}")
            if unknown:
                parts.append(f"UNKNOWN na pozycjach: {', '.join(map(str, sorted(unknown)))}")
            await message.channel.send(
                "❗ Nie mogę dodać tej wojny do strony, bo lista wciąż wymaga poprawek: " + "; ".join(parts)
            )
            try:
                await ref_msg.add_reaction("⚠️")
            except Exception:
                pass
            try:
                await message.delete()
            except Exception:
                pass
            return True

    now_ts = int(time.time())
    if post is not None:
        rec = _post_to_store_record(
            post,
            ref_msg=None,
            bot_msg=ref_msg,
            store_status="confirmed",
            confirmed_at_ts=now_ts,
            confirmed_by_user_id=message.author.id,
            confirmed_by_user_tag=str(message.author),
        )
        # Preserve immutable fields from any existing draft
        if isinstance(existing, dict):
            for k in (
                "ref_message_id",
                "ref_jump_url",
                "created_at_ts",
                "created_at_iso",
                "guild_id",
                "channel_id",
            ):
                if rec.get(k) is None and existing.get(k) is not None:
                    rec[k] = existing[k]
    else:
        # Fallback: confirm whatever is already in the store (draft record).
        if not isinstance(existing, dict):
            await message.channel.send("Nie mogę znaleźć danych tej wojny (brak w pamięci i w store). Zrób LISTWAR ponownie.")
            return True
        rec = dict(existing)
        rec["status"] = "confirmed"
        rec["confirmed_at_ts"] = now_ts
        rec["confirmed_by_user_id"] = message.author.id
        rec["confirmed_by_user_tag"] = str(message.author)
        rec["updated_at_ts"] = now_ts

    try:
        async with WAR_STORE_LOCK:
            await asyncio.to_thread(WAR_STORE.upsert_war, war_id, rec)
        try:
            await _persist_progress_to_discord(discord_client, what="wars")
        except Exception:
            pass
        try:
            await ref_msg.add_reaction("🟢")
        except Exception:
            pass
        # Avoid spamming the channel on repeated edits/toggles: update the LISTWAR message with current status.
        updated = await _try_update_listwar_status_message(
            discord_client,
            war_id=war_id,
            status_line="✅ Dodano do strony",
            ref_msg=ref_msg,
        )
        if not updated:
            await message.channel.send(f"✅ Dodano do strony: `{war_id}`")
    except Exception:
        logger.exception("ADDWAR: failed to persist confirm war_id=%s", war_id)
        await message.channel.send("❌ Nie udało się dodać wojny do strony (błąd zapisu).")

    try:
        await message.delete()
    except Exception:
        pass
    return True


async def try_apply_manual_corrections(
    discord_client: discord.Client,
    message: discord.Message,
) -> bool:
    """If message is a reply to a bot war-post, apply corrections and delete the user's message."""
    logger.info("Manual correction check: msg_id=%s author=%s", message.id, message.author.id)
    if not message.reference or not message.reference.message_id:
        return False

    # Fetch referenced message
    ref_msg: Optional[discord.Message] = None
    if isinstance(message.reference.resolved, discord.Message):
        ref_msg = message.reference.resolved
    else:
        try:
            ref_msg = await message.channel.fetch_message(message.reference.message_id)
        except Exception:
            return False

    if not ref_msg:
        return False
    if not discord_client.user:
        return False
    if ref_msg.author.id != discord_client.user.id:
        return False

    post = WAR_POSTS.get(ref_msg.id)
    if not post:
        # we only support edits for messages we created in this runtime
        logger.warning("Manual correction: referenced msg %s not found in WAR_POSTS", ref_msg.id)
        return False

    corrections = _parse_manual_corrections(message.content)
    if not corrections:
        return False

    logger.info("Manual correction parsed: %s", corrections)

    roster = load_roster()

    for rank, nick, points_opt in corrections:
        # If points not provided, keep the existing points for that rank.
        if points_opt is None:
            existing = post.lines_by_rank.get(rank)
            if not existing:
                logger.warning(
                    "Manual correction skipped: rank %d has no existing line and no points were provided",
                    rank,
                )
                continue
            points = int(existing.points)
        else:
            points = int(points_opt)

        roster_name = resolve_to_roster(nick, None, roster)
        logger.debug(
            "Manual correction line: rank=%d raw_nick=%r resolved=%r points=%d (provided=%s)",
            rank,
            nick,
            roster_name,
            points,
            points_opt is not None,
        )
        if roster_name:
            post.lines_by_rank[rank] = WarLine(
                rank=rank,
                points=points,
                name_raw=nick,
                name_display=roster_name,
                unknown_raw=None,
            )
        else:
            if is_clean_display(nick):
                cleaned = normalize_display(nick)
                post.lines_by_rank[rank] = WarLine(
                    rank=rank,
                    points=points,
                    name_raw=nick,
                    name_display=cleaned,
                    unknown_raw=None,
                    out_of_roster_raw=cleaned,
                )
            else:
                post.lines_by_rank[rank] = WarLine(
                    rank=rank,
                    points=points,
                    name_raw=nick,
                    name_display="UNKNOWN",
                    unknown_raw=normalize_display(nick) or nick,
                    out_of_roster_raw=None,
                )

        if rank > post.expected_max_rank:
            post.expected_max_rank = rank

    # Re-render and edit
    new_content = render_post(post)
    # If it's too long (unlikely), truncate warnings last.
    parts = chunk_message(new_content)
    if len(parts) != 1:
        # Keep only the first chunk to avoid multi-message edit complexity.
        new_content = parts[0]

    try:
        await ref_msg.edit(content=new_content)
    except Exception:
        logger.exception("Manual correction: failed to edit message %s", ref_msg.id)
        return False

    # Signal success
    try:
        await ref_msg.add_reaction("✅")
    except Exception:
        pass

    # Persist updated war to the store (keeps draft/confirmed status)
    try:
        async with WAR_STORE_LOCK:
            existing = await asyncio.to_thread(WAR_STORE.get_war, post.war_id)

        existing_status = str((existing or {}).get("status") or "draft").lower()
        rec = _post_to_store_record(
            post,
            ref_msg=None,
            bot_msg=ref_msg,
            store_status=existing_status,
            confirmed_at_ts=(existing or {}).get("confirmed_at_ts"),
            confirmed_by_user_id=(existing or {}).get("confirmed_by_user_id"),
            confirmed_by_user_tag=(existing or {}).get("confirmed_by_user_tag"),
        )

        if isinstance(existing, dict):
            for k in (
                "ref_message_id",
                "ref_jump_url",
                "created_at_ts",
                "created_at_iso",
                "guild_id",
                "channel_id",
            ):
                if rec.get(k) is None and existing.get(k) is not None:
                    rec[k] = existing[k]

        async with WAR_STORE_LOCK:
            await asyncio.to_thread(WAR_STORE.upsert_war, post.war_id, rec)
        try:
            await _persist_progress_to_discord(discord_client, what="wars")
        except Exception:
            pass
        logger.info("Manual correction: persisted war store update war_id=%s status=%s", post.war_id, existing_status)
    except Exception:
        logger.exception("Manual correction: failed to persist war store update war_id=%s", post.war_id)

    # Delete user's correction message
    try:
        await message.delete()
        logger.info("Manual correction: deleted user msg %s", message.id)
    except Exception:
        # requires Manage Messages; ignore if not available
        logger.warning("Manual correction: could not delete user msg %s (missing permissions?)", message.id)
        pass

    return True


def main():
    if not DISCORD_TOKEN:
        raise SystemExit("Brak DISCORD_TOKEN w .env / env vars")
    if WATCH_CHANNEL_ID == 0:
        raise SystemExit("Brak WATCH_CHANNEL_ID w .env / env vars")

    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        try:
            await _restore_progress_from_discord(client)
        except Exception:
            logger.exception("Failed to restore progress from Discord")
        asyncio.create_task(start_keepalive_server())
        logger.info("Zalogowano jako: %s | obserwuję kanał: %s", client.user, WATCH_CHANNEL_ID)

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        if message.channel.id != WATCH_CHANNEL_ID:
            return

        text_raw = message.content or ""
        text = text_raw.strip()
        is_reply = bool(message.reference and message.reference.message_id)

        # Fast pre-filter: we only react to explicit commands (or manual corrections).
        if is_reply:
            if not text:
                return
            tok0 = text.split()[0]
            tok0_clean = tok0.strip("[](){}")
            tok0_up = tok0_clean.upper()
            if not (tok0_up in {"LISTWAR", "ADDWAR", "ADDROSTER", "REMOVEROSTER"} or tok0_clean.isdigit()):
                return
        else:
            if not text:
                return
            tok0_up = text.split()[0].upper()
            if tok0_up not in {"ADDROSTER", "REMOVEROSTER", "CURRENTROSTER", "REMOVEWAR", "UNLISTWAR", "HELP"}:
                return

        # Try to "claim" this command message to prevent duplicate responses
        # in case two instances are connected at the same time.
        claimed = await _try_claim_command_message(message)
        if not claimed:
            return

        trace_id = _new_trace_id("discord")
        token = set_trace_id(trace_id)
        capture_handler: Optional[logging.Handler] = None
        capture_stream: Optional[io.StringIO] = None
        processed_images = False
        had_warning = False
        had_unknown = False
        had_out_of_roster = False
        had_missing = False
        had_exception = False
        bot_post_msg: Optional[discord.Message] = None
        listwar_progress_msg: Optional[discord.Message] = None

        # Capture a per-run DEBUG trace so it can be uploaded to Discord (useful on Render Free).
        try:
            capture_handler, capture_stream = _attach_per_trace_logger(trace_id)
        except Exception:
            capture_handler = None
            capture_stream = None

        logger.info(
            "===== START on_message id=%s author=%s reply=%s attachments=%d content_len=%d =====",
            message.id,
            message.author.id,
            bool(is_reply),
            len(message.attachments),
            len(text_raw or ""),
        )

        try:
            # ----------------------------
            # 1) Channel commands (no reply)
            # ----------------------------
            if not is_reply:
                try:
                    handled = await try_apply_help_channel_command(client, message)
                    if handled:
                        logger.info("HELP handled -> done")
                        return
                except Exception:
                    logger.exception("Błąd podczas HELP")
                    had_exception = True

                try:
                    handled = await try_apply_currentroster_channel_command(client, message)
                    if handled:
                        logger.info("CURRENTROSTER handled -> done")
                        return
                except Exception:
                    logger.exception("Błąd podczas CURRENTROSTER")
                    had_exception = True

                try:
                    handled = await try_apply_addroster_channel_command(client, message)
                    if handled:
                        logger.info("ADDROSTER(channel) handled -> done")
                        return
                except Exception:
                    logger.exception("Błąd podczas ADDROSTER(channel)")
                    had_exception = True

                try:
                    handled = await try_apply_removeroster_channel_command(client, message)
                    if handled:
                        logger.info("REMOVEROSTER(channel) handled -> done")
                        return
                except Exception:
                    logger.exception("Błąd podczas REMOVEROSTER(channel)")
                    had_exception = True

                try:
                    handled = await try_apply_unlistwar_channel_command(client, message)
                    if handled:
                        logger.info("UNLISTWAR handled -> done")
                        return
                except Exception:
                    logger.exception("Błąd podczas UNLISTWAR")
                    had_exception = True

                try:
                    handled = await try_apply_removewar_channel_command(client, message)
                    if handled:
                        logger.info("REMOVEWAR handled -> done")
                        return
                except Exception:
                    logger.exception("Błąd podczas REMOVEWAR")
                    had_exception = True

                return

            # ----------------------------
            # 2) Reply commands
            # ----------------------------
            # LISTWAR: reply to a message containing the two screenshots.
            if re.match(r"^\s*LISTWAR\b", text_raw, flags=re.IGNORECASE):
                # Fetch referenced message (the one with screenshots)
                ref_msg: Optional[discord.Message] = None
                if isinstance(message.reference.resolved, discord.Message):
                    ref_msg = message.reference.resolved
                else:
                    try:
                        ref_msg = await message.channel.fetch_message(message.reference.message_id)
                    except Exception:
                        ref_msg = None

                if not ref_msg:
                    await message.channel.send("Nie mogę znaleźć wiadomości, do której odpowiadasz (LISTWAR).")
                    return

                # Prefer reusing an existing active war_id for the same screenshot message.
                # If the prior DRAFT was removed via UNLISTWAR, we will generate a new ID
                # using a per-ref-message sequence counter (WAR-XXXX-2, ...).
                war_id = None
                try:
                    async with WAR_STORE_LOCK:
                        wars_list = await asyncio.to_thread(WAR_STORE.get_wars, True)
                    for r in wars_list:
                        try:
                            if int(r.get("ref_message_id") or 0) != int(ref_msg.id):
                                continue
                        except Exception:
                            continue
                        st = str(r.get("status") or "").lower()
                        if st in {"draft", "confirmed"}:
                            wid = str(r.get("war_id") or "").strip()
                            if wid:
                                war_id = wid
                                break
                except Exception:
                    war_id = None

                if not war_id:
                    async with WAR_STORE_LOCK:
                        seq = await asyncio.to_thread(WAR_STORE.bump_ref_sequence, ref_msg.id)
                    war_id = make_war_id_with_seq(ref_msg.id, seq)

                atts = [a for a in ref_msg.attachments if is_image(a)]
                if len(atts) < 2:
                    await message.channel.send(
                        "LISTWAR musi być reply na wiadomość z **dwoma** screenami (lista + podsumowanie wojny)."
                    )
                    return

                # Two-phase for cost: cache by the referenced screenshot message id.
                # If users run LISTWAR multiple times for the same screenshots, we reuse the parse.
                cached = WAR_PARSE_CACHE.get(ref_msg.id)

                # Inform the user immediately that processing has started (so they don't think the bot is stuck).
                if LISTWAR_PROGRESS_ENABLED:
                    eta = "kilka sekund" if cached is not None else "30–90 sekund"
                    try:
                        listwar_progress_msg = await message.channel.send(
                            f"⏳ {message.author.mention} Rozpoczynam analizę screenów (LISTWAR) dla `{war_id}`. "
                            f"To może potrwać {eta}…"
                        )
                    except Exception:
                        listwar_progress_msg = None

                try:
                    logger.info(
                        "LISTWAR using referenced attachments: #1=%s (%d bytes), #2=%s (%d bytes)",
                        atts[0].filename,
                        int(getattr(atts[0], "size", 0) or 0),
                        atts[1].filename,
                        int(getattr(atts[1], "size", 0) or 0),
                    )

                    if cached is not None:
                        summary, players, expected_max_rank = cached
                        logger.info(
                            "LISTWAR cache hit: ref_msg.id=%s war_id=%s players=%d",
                            ref_msg.id,
                            war_id,
                            len(players),
                        )
                        processed_images = True
                    else:
                        images = [await atts[0].read(), await atts[1].read()]
                        logger.debug("LISTWAR downloaded attachments: sizes=%s", [len(b) for b in images])
                        processed_images = True

                        try:
                            summary, players, expected_max_rank, _debug = await parse_images_in_thread(
                                images, trace_id=trace_id
                            )
                        except Exception as e:
                            logger.exception("Błąd parsowania OpenAI (LISTWAR): %s", e)
                            had_exception = True
                            await message.channel.send(
                                "Nie udało się odczytać screenów (błąd po stronie parsera)."
                            )
                            return

                        # Store parse result for this screenshot message
                        try:
                            if summary and players and expected_max_rank:
                                WAR_PARSE_CACHE[ref_msg.id] = (summary, players, int(expected_max_rank))
                                logger.info(
                                    "LISTWAR cached parse: ref_msg.id=%s war_id=%s players=%d max_rank=%s",
                                    ref_msg.id,
                                    war_id,
                                    len(players),
                                    expected_max_rank,
                                )
                        except Exception:
                            logger.exception("LISTWAR: failed to cache parse result")

                    if not summary or not players:
                        logger.warning(
                            "Parser returned incomplete data: summary=%s players=%s",
                            bool(summary),
                            bool(players),
                        )
                        await message.channel.send(
                            "Nie udało się jednoznacznie odczytać obu screenów (brak summary albo listy). "
                            "Upewnij się, że widać cały panel z listą i paski wyniku."
                        )
                        return

                    post = build_post(summary, players, expected_max_rank)
                    post.war_id = war_id
                    # Attach source metadata for traceability and later ADDWAR confirmation.
                    post.ref_message_id = ref_msg.id
                    try:
                        post.ref_jump_url = ref_msg.jump_url
                    except Exception:
                        post.ref_jump_url = None
                    try:
                        post.guild_id = int(ref_msg.guild.id) if getattr(ref_msg, "guild", None) else None
                    except Exception:
                        post.guild_id = None
                    try:
                        post.channel_id = int(ref_msg.channel.id)
                    except Exception:
                        post.channel_id = None
                    try:
                        post.created_at_ts = int(ref_msg.created_at.timestamp())
                    except Exception:
                        post.created_at_ts = int(time.time())
                    had_unknown = bool(post.unknown_ranks())
                    had_out_of_roster = bool(post.out_of_roster_ranks())
                    had_missing = bool(post.missing_ranks())
                    had_warning = bool(had_missing or had_unknown or had_out_of_roster)

                    out = render_post(post)
                    parts = chunk_message(out)
                    if len(parts) != 1:
                        out = parts[0]
                        logger.warning("Output message exceeded Discord limit; truncated to 1 chunk")

                    # Reply to the original screenshot message for context.
                    bot_post_msg = await ref_msg.reply(out, mention_author=False)
                    WAR_POSTS[bot_post_msg.id] = post
                    logger.info("LISTWAR sent war post msg_id=%s stored in WAR_POSTS", bot_post_msg.id)

                    # Persist as a DRAFT (not shown on the website until ADDWAR).
                    try:
                        async with WAR_STORE_LOCK:
                            existing = await asyncio.to_thread(WAR_STORE.get_war, post.war_id)
                            existing_status = str((existing or {}).get("status") or "").lower()
                        if existing_status != "confirmed":
                            rec = _post_to_store_record(
                                post, ref_msg=ref_msg, bot_msg=bot_post_msg, store_status="draft"
                            )
                            # Preserve immutable timestamps if we already have a draft.
                            if isinstance(existing, dict):
                                for k in ("created_at_ts", "created_at_iso", "ref_message_id", "ref_jump_url"):
                                    if existing.get(k) is not None:
                                        rec[k] = existing[k]
                            async with WAR_STORE_LOCK:
                                await asyncio.to_thread(WAR_STORE.upsert_war, post.war_id, rec)
                            try:
                                await _persist_progress_to_discord(client, what="wars")
                            except Exception:
                                pass
                            logger.info("LISTWAR persisted DRAFT war_id=%s to store", post.war_id)
                        else:
                            logger.info(
                                "LISTWAR: war_id=%s already confirmed -> not overwriting store",
                                post.war_id,
                            )
                    except Exception:
                        logger.exception("LISTWAR: failed to persist draft war to store")

                    # Delete LISTWAR command message to keep the channel clean (best effort).
                    try:
                        await message.delete()
                    except Exception:
                        pass

                    return
                finally:
                    # Clean up the progress message if we managed to send one.
                    if listwar_progress_msg is not None:
                        try:
                            await listwar_progress_msg.delete()
                        except Exception:
                            pass

            # ADDWAR: reply to the bot's war list to confirm and add it to the website.
            try:
                handled = await try_apply_addwar_command(client, message)
                if handled:
                    logger.info("ADDWAR handled -> done")
                    return
            except Exception:
                logger.exception("Błąd podczas ADDWAR")
                had_exception = True

            # Reply: ADDROSTER / REMOVEROSTER / manual corrections for bot posts.
            try:
                handled = await try_apply_addroster_command(client, message)
                if handled:
                    logger.info("ADDROSTER(reply) handled -> done")
                    return
            except Exception:
                logger.exception("Błąd podczas ADDROSTER")
                had_exception = True

            try:
                handled = await try_apply_removeroster_command(client, message)
                if handled:
                    logger.info("REMOVEROSTER(reply) handled -> done")
                    return
            except Exception:
                logger.exception("Błąd podczas REMOVEROSTER")
                had_exception = True

            try:
                handled = await try_apply_manual_corrections(client, message)
                if handled:
                    logger.info("Manual correction handled -> done")
                    return
            except Exception:
                logger.exception("Błąd podczas manual correction")
                had_exception = True

        finally:
            logger.info("===== END on_message id=%s =====", message.id)
            _flush_logs()

            # Detach capture handler before we send anything else (avoid capturing its own upload logs).
            if capture_handler is not None:
                try:
                    logging.getLogger().removeHandler(capture_handler)
                    capture_handler.flush()
                    capture_handler.close()
                except Exception:
                    pass

            # Optionally upload the per-run debug log to Discord.
            try:
                if SEND_LOG_TO_DISCORD and processed_images and capture_stream is not None:
                    if SEND_LOG_ONLY_ON_UNKNOWN:
                        should_send = had_unknown or had_out_of_roster or had_missing or had_exception
                    elif SEND_LOG_ONLY_ON_WARN:
                        should_send = had_warning or had_exception
                    else:
                        should_send = True

                    if should_send:
                        raw_text = capture_stream.getvalue()
                        raw_text = _redact_secrets(raw_text)
                        data = raw_text.encode("utf-8", errors="replace")

                        truncated = False
                        if DISCORD_LOG_MAX_BYTES and len(data) > int(DISCORD_LOG_MAX_BYTES):
                            truncated = True
                            tail = data[-int(DISCORD_LOG_MAX_BYTES):]
                            prefix = (
                                f"[TRUNCATED] Log exceeded {DISCORD_LOG_MAX_BYTES} bytes; showing last chunk only.\n"
                            ).encode("utf-8")
                            data = prefix + tail

                        bio = io.BytesIO(data)
                        filename = f"log-{trace_id}.txt"
                        file = discord.File(fp=bio, filename=filename)

                        # IMPORTANT: send debug logs to the private storage channel (e.g. #warbot-storage)
                        # instead of spamming the public results channel (#wyniki-wojenne).
                        target_ch: discord.abc.Messageable = message.channel
                        try:
                            storage = await _ensure_storage_channel(client)
                            if storage is not None:
                                target_ch = storage
                        except Exception:
                            pass

                        src = None
                        try:
                            src = (bot_post_msg.jump_url if bot_post_msg is not None else message.jump_url)
                        except Exception:
                            src = None

                        note = f"📝 Debug log: `{trace_id}`" + (" (truncated)" if truncated else "")
                        if src:
                            note += f"\n🔗 Source: {src}"

                        await target_ch.send(note, file=file)
            except Exception:
                logger.exception("Failed to upload debug log to Discord")

            reset_trace_id(token)


    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
