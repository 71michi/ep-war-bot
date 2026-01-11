import os
import re
import asyncio
import logging
import time
import uuid
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aiohttp import web
import discord

from .config import env_str, env_int
from .logging_setup import setup_logging, set_trace_id, reset_trace_id, get_trace_id
from .openai_parser import parse_war_from_images, WarSummary, PlayerScore
from .nicknames import normalize_with_aliases, normalize_display, roster_match, canonical_key
from rapidfuzz import fuzz, process

setup_logging()
logger = logging.getLogger("warbot")


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
OPENAI_MODEL = env_str("OPENAI_MODEL", "gpt-4o-mini")
ALIASES_PATH = env_str("ALIASES_PATH", "aliases.json")
ROSTER_PATH = env_str("ROSTER_PATH", "roster.json")

# When running on Render Free (no shell / no FS access), you can still get the full debug trace.
# By default we attach per-run debug logs as a text file on Discord.
SEND_LOG_TO_DISCORD = env_int("SEND_LOG_TO_DISCORD", 1) == 1
# If enabled, we only send logs when warnings were emitted (missing/unknown/etc.) or on exceptions.
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
    global _keepalive_started
    if _keepalive_started:
        return
    _keepalive_started = True

    app = web.Application()

    async def health(_request):
        return web.Response(text="ok")

    app.router.add_get("/health", health)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.getenv("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logger.info("Keepalive HTTP listening on :%s/health", port)


# ----------------------------
# Roster helpers
# ----------------------------

_roster_cache: Tuple[float, List[str]] = (0.0, [])


def load_roster() -> List[str]:
    """Load roster.json with a tiny mtime cache."""
    import json
    global _roster_cache

    try:
        st = os.stat(ROSTER_PATH)
        if _roster_cache[0] == st.st_mtime and _roster_cache[1]:
            return _roster_cache[1]
        with open(ROSTER_PATH, "r", encoding="utf-8") as f:
            roster = json.load(f).get("roster", [])
        roster = [str(x) for x in roster if str(x).strip()]
        _roster_cache = (st.st_mtime, roster)
        return roster
    except FileNotFoundError:
        return []
    except Exception:
        return []


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

@dataclass
class WarPost:
    summary: WarSummary
    expected_max_rank: int
    lines_by_rank: Dict[int, WarLine] = field(default_factory=dict)

    def missing_ranks(self) -> List[int]:
        return [r for r in range(1, self.expected_max_rank + 1) if r not in self.lines_by_rank]

    def unknown_ranks(self) -> List[int]:
        return [r for r, ln in self.lines_by_rank.items() if ln.name_display == "UNKNOWN"]


# message_id -> WarPost
WAR_POSTS: Dict[int, WarPost] = {}


def _generate_key_variants(key: str) -> List[str]:
    """Generate a small set of OCR-robust variants for a canonical key.

    IMPORTANT: We generate *combinations* of a few transforms (2 rounds) so we can
    recover cases like stylized "ɭαяσ" -> canonical "lars" -> variants "laro" -> "jaro".
    """
    if not key:
        return []

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

        # First letter confusions (Jaro case)
        if v.startswith("r"):
            out.add("j" + v[1:])
        if v.startswith("j"):
            out.add("r" + v[1:])
        if v.startswith("l"):
            out.add("j" + v[1:])

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

        return {x for x in out if x}

    # Two rounds to allow chaining (e.g. lars -> laro -> jaro)
    for _ in range(2):
        cur = list(variants)
        for v in cur:
            variants.update(_step(v))

        # Cap the set to avoid blowups (roster is small; we don't need many variants)
        if len(variants) > 40:
            variants = set(sorted(variants)[:40])

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
            logger.debug("Final roster mapping: rank %s raw=%r -> %s", r, ln.name_raw, ln.name_display)
        else:
            ln.name_display = "UNKNOWN"
            ln.unknown_raw = ln.name_raw
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
    if missing or unknown:
        logger.warning("Post validation: missing=%s unknown=%s", missing, unknown)
    else:
        logger.info("Post validation: OK")

    return post


def render_post(post: WarPost) -> str:
    s = post.summary

    header = (
        f"**Wojna zakończona: {s.result}**\n"
        f"**{s.our_alliance}** {s.our_score} — {s.opponent_score} **{s.opponent_alliance}**\n"
    )
    if s.war_mode:
        header += f"Tryb: **{s.war_mode}**" + (" (BETA)\n" if s.beta_badge else "\n")
    header += "\n"

    lines_out: List[str] = []
    for r in sorted(post.lines_by_rank.keys()):
        ln = post.lines_by_rank[r]
        lines_out.append(f"[{ln.rank:02d}] {ln.name_display} — {ln.points}")

    msg = header + "\n".join(lines_out)

    missing = post.missing_ranks()
    unknown = post.unknown_ranks()

    if missing or unknown:
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
            warn_lines.append("• Nieznane nicki: " + ", ".join(parts))

        warn_lines.append("")
        warn_lines.append("Reply na tę wiadomość w formacie: `23 ropuch13 250` (może być wiele linii).")
        warn_lines.append("Po przetworzeniu poprawki bot usunie Twoją wiadomość.")
        msg += "\n" + "\n".join(warn_lines)

    return msg


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


def is_image(att: discord.Attachment) -> bool:
    if att.content_type:
        return att.content_type.startswith("image/")
    return att.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))


async def parse_images_in_thread(images: List[bytes], trace_id: str):
    # Pass trace_id explicitly so the OpenAI thread gets the same id.
    return await asyncio.to_thread(parse_war_from_images, images, OPENAI_MODEL, trace_id=trace_id)


def _parse_manual_corrections(text: str) -> List[Tuple[int, str, int]]:
    """Parse lines like: 23 ropuch13 250 (rank, nick, points)."""
    out: List[Tuple[int, str, int]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Normalize separators
        line = line.replace("—", " ").replace("-", " ").replace(":", " ")
        toks = [t for t in line.split() if t]
        if len(toks) < 3:
            continue

        # Rank
        t0 = toks[0].strip("[](){}")
        try:
            rank = int(t0)
        except ValueError:
            continue

        # Points
        t_last = toks[-1].strip("[](){}")
        try:
            points = int(t_last)
        except ValueError:
            continue

        nick = " ".join(toks[1:-1]).strip()
        if not nick:
            continue

        if rank <= 0 or rank > 200:
            continue
        if points < 0 or points > 9999:
            continue

        out.append((rank, nick, points))
    return out


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

    for rank, nick, points in corrections:
        roster_name = resolve_to_roster(nick, None, roster)
        logger.debug("Manual correction line: rank=%d raw_nick=%r resolved=%r points=%d", rank, nick, roster_name, points)
        if roster_name:
            post.lines_by_rank[rank] = WarLine(
                rank=rank,
                points=points,
                name_raw=nick,
                name_display=roster_name,
                unknown_raw=None,
            )
        else:
            post.lines_by_rank[rank] = WarLine(
                rank=rank,
                points=points,
                name_raw=nick,
                name_display="UNKNOWN",
                unknown_raw=nick,
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
        asyncio.create_task(start_keepalive_server())
        logger.info("Zalogowano jako: %s | obserwuję kanał: %s", client.user, WATCH_CHANNEL_ID)

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        if message.channel.id != WATCH_CHANNEL_ID:
            return

        trace_id = _new_trace_id("discord")
        token = set_trace_id(trace_id)
        capture_handler: Optional[logging.Handler] = None
        capture_stream: Optional[io.StringIO] = None
        processed_images = False
        had_warning = False
        had_exception = False
        bot_post_msg: Optional[discord.Message] = None

        # Capture a per-run DEBUG trace so it can be uploaded to Discord (useful on Render Free).
        try:
            capture_handler, capture_stream = _attach_per_trace_logger(trace_id)
        except Exception:
            capture_handler = None
            capture_stream = None
        logger.info(
            "===== START on_message id=%s author=%s attachments=%d content_len=%d =====",
            message.id,
            message.author.id,
            len(message.attachments),
            len(message.content or ""),
        )

        try:
            # 1) Manual corrections via reply
            try:
                handled = await try_apply_manual_corrections(client, message)
                if handled:
                    logger.info("Manual correction handled -> done")
                    return
            except Exception:
                logger.exception("Błąd podczas manual correction")
                had_exception = True

            # 2) Regular flow: 2 images -> parse -> post
            atts = [a for a in message.attachments if is_image(a)]
            if len(atts) < 2:
                logger.info("Not enough image attachments (%d). Ignoring.", len(atts))
                return

            logger.info(
                "Using attachments: #1=%s (%d bytes, ct=%s), #2=%s (%d bytes, ct=%s)",
                atts[0].filename,
                int(getattr(atts[0], "size", 0) or 0),
                atts[0].content_type,
                atts[1].filename,
                int(getattr(atts[1], "size", 0) or 0),
                atts[1].content_type,
            )

            images = [await atts[0].read(), await atts[1].read()]
            logger.debug("Downloaded attachments: sizes=%s", [len(b) for b in images])
            processed_images = True

            try:
                summary, players, expected_max_rank, _debug = await parse_images_in_thread(images, trace_id=trace_id)
            except Exception as e:
                logger.exception("Błąd parsowania OpenAI: %s", e)
                had_exception = True
                await message.channel.send("Nie udało się odczytać screenów (błąd po stronie parsera).")
                return

            if not summary or not players:
                logger.warning("Parser returned incomplete data: summary=%s players=%s", bool(summary), bool(players))
                await message.channel.send(
                    "Nie udało się jednoznacznie odczytać obu screenów (brak summary albo listy). "
                    "Upewnij się, że widać cały panel z listą i paski wyniku."
                )
                return

            post = build_post(summary, players, expected_max_rank)
            had_warning = bool(post.missing_ranks() or post.unknown_ranks())
            out = render_post(post)

            parts = chunk_message(out)
            if len(parts) != 1:
                # Send only first chunk; if this happens, we still want consistent reply behavior.
                out = parts[0]
                logger.warning("Output message exceeded Discord limit; truncated to 1 chunk")

            sent = await message.channel.send(out)
            bot_post_msg = sent
            WAR_POSTS[sent.id] = post
            logger.info("Sent war post msg_id=%s stored in WAR_POSTS", sent.id)
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
                    if (not SEND_LOG_ONLY_ON_WARN) or had_warning or had_exception:
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

                        note = f"📝 Debug log: `{trace_id}`" + (" (truncated)" if truncated else "")
                        if bot_post_msg is not None:
                            await bot_post_msg.reply(note, file=file, mention_author=False)
                        else:
                            await message.channel.send(note, file=file)
            except Exception:
                # Never fail the whole handler due to log upload problems.
                logger.exception("Failed to upload debug log to Discord")

            reset_trace_id(token)

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
