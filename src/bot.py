import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aiohttp import web
import discord

from .config import env_str, env_int
from .openai_parser import parse_war_from_images, WarSummary, PlayerScore
from .nicknames import normalize_with_aliases, normalize_display, roster_match

LOG_LEVEL = env_str("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)

DISCORD_TOKEN = env_str("DISCORD_TOKEN", "")
WATCH_CHANNEL_ID = env_int("WATCH_CHANNEL_ID", 0)
OPENAI_MODEL = env_str("OPENAI_MODEL", "gpt-4o-mini")
ALIASES_PATH = env_str("ALIASES_PATH", "aliases.json")
ROSTER_PATH = env_str("ROSTER_PATH", "roster.json")


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

    logging.info("Keepalive HTTP listening on :%s/health", port)


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
    """Return a roster name or None (strict roster-only)."""
    if not roster:
        return None

    roster_lower = {r.lower(): r for r in roster}

    # 1) If model provided name_norm, trust it only if it's in roster.
    if name_norm_from_model:
        nn = name_norm_from_model.strip()
        if nn and nn.lower() in roster_lower:
            return roster_lower[nn.lower()]

    # 2) Aliases
    mapped = normalize_with_aliases(name_raw, ALIASES_PATH)
    if mapped and mapped.lower() in roster_lower:
        return roster_lower[mapped.lower()]

    # 3) Exact match after minimal cleanup
    cleaned = normalize_display(name_raw)
    if cleaned.lower() in roster_lower:
        return roster_lower[cleaned.lower()]

    # 4) Fuzzy
    hit = roster_match(cleaned, roster, min_score=88)
    return hit


# ----------------------------
# War post model + rendering
# ----------------------------


@dataclass
class WarLine:
    rank: int
    points: int
    name_raw: str
    name_display: str
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


def build_post(summary: WarSummary, players: List[PlayerScore], expected_max_rank: Optional[int]) -> WarPost:
    roster = load_roster()

    by_rank: Dict[int, WarLine] = {}

    def better(a: WarLine, b: WarLine) -> WarLine:
        # prefer non-UNKNOWN, then longer raw, then higher points
        a_score = (0 if a.name_display == "UNKNOWN" else 100) + min(len(a.name_raw or ""), 40) + int(a.points)
        b_score = (0 if b.name_display == "UNKNOWN" else 100) + min(len(b.name_raw or ""), 40) + int(b.points)
        return a if a_score >= b_score else b

    for p in players:
        if not isinstance(p.rank, int) or p.rank <= 0 or p.rank > 200:
            continue
        if not isinstance(p.points, int) or p.points < 0 or p.points > 9999:
            continue

        roster_name = resolve_to_roster(p.name_raw, p.name_norm, roster)
        if roster_name:
            ln = WarLine(rank=p.rank, points=p.points, name_raw=p.name_raw, name_display=roster_name, unknown_raw=None)
        else:
            ln = WarLine(rank=p.rank, points=p.points, name_raw=p.name_raw, name_display="UNKNOWN", unknown_raw=p.name_raw)

        if p.rank in by_rank:
            by_rank[p.rank] = better(by_rank[p.rank], ln)
        else:
            by_rank[p.rank] = ln

    mx = expected_max_rank if expected_max_rank and expected_max_rank > 0 else None
    if not mx:
        mx = max(by_rank.keys()) if by_rank else 0
    mx = int(mx) if mx else 0

    return WarPost(summary=summary, expected_max_rank=mx, lines_by_rank=by_rank)


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


async def parse_images_in_thread(images: List[bytes]):
    return await asyncio.to_thread(parse_war_from_images, images, OPENAI_MODEL)


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
        return False

    corrections = _parse_manual_corrections(message.content)
    if not corrections:
        return False

    roster = load_roster()

    for rank, nick, points in corrections:
        roster_name = resolve_to_roster(nick, None, roster)
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
        return False

    # Signal success
    try:
        await ref_msg.add_reaction("✅")
    except Exception:
        pass

    # Delete user's correction message
    try:
        await message.delete()
    except Exception:
        # requires Manage Messages; ignore if not available
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
        logging.info("Zalogowano jako: %s | obserwuję kanał: %s", client.user, WATCH_CHANNEL_ID)

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        if message.channel.id != WATCH_CHANNEL_ID:
            return

        # 1) Manual corrections via reply
        try:
            handled = await try_apply_manual_corrections(client, message)
            if handled:
                return
        except Exception:
            logging.exception("Błąd podczas manual correction")

        # 2) Regular flow: 2 images -> parse -> post
        atts = [a for a in message.attachments if is_image(a)]
        if len(atts) < 2:
            return

        images = [await atts[0].read(), await atts[1].read()]

        try:
            summary, players, expected_max_rank, _debug = await parse_images_in_thread(images)
        except Exception as e:
            logging.exception("Błąd parsowania OpenAI: %s", e)
            await message.channel.send("Nie udało się odczytać screenów (błąd po stronie parsera).")
            return

        if not summary or not players:
            await message.channel.send(
                "Nie udało się jednoznacznie odczytać obu screenów (brak summary albo listy). "
                "Upewnij się, że widać cały panel z listą i paski wyniku."
            )
            return

        post = build_post(summary, players, expected_max_rank)
        out = render_post(post)

        parts = chunk_message(out)
        if len(parts) != 1:
            # Send only first chunk; if this happens, we still want consistent reply behavior.
            out = parts[0]

        sent = await message.channel.send(out)
        WAR_POSTS[sent.id] = post

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
