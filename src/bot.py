import os
import asyncio
import logging
import re
from typing import List, Tuple, Dict, Any, Optional

import discord
from aiohttp import web

from .config import env_str, env_int
from .openai_parser import parse_war_from_images, WarSummary, PlayerScore
from .nicknames import normalize_display, resolve_players_to_roster_unique

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

# Optional: lock corrections behind a role.
# If ADMIN_ROLE_ID=0 or missing -> anyone can correct (not recommended for public channels).
ADMIN_ROLE_ID = env_int("ADMIN_ROLE_ID", 0)

# Expected output line format:
# [01] Nick — 307
_RESULT_LINE_RE = re.compile(r"^\[(\d{2})\]\s+(.+?)\s+—\s+(\d+)\s*$")

# Correction line format (when replying to the bot message):
# 28 Jaro 250
# 28 Jaro +250
# 28 +10
# 28 240
_CORR_LINE_RE = re.compile(r"^\s*\[?\s*(\d{1,2})\s*\]?\s*(.*)$")


async def start_keepalive_server():
    """Small HTTP server so Render can healthcheck and keep the service alive."""
    app = web.Application()

    async def health(_request):
        return web.Response(text="ok")

    app.router.add_get("/", health)
    app.router.add_get("/health", health)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.getenv("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info("Keepalive HTTP listening on :%s/health", port)


def load_roster() -> list[str]:
    import json
    try:
        with open(ROSTER_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("roster", [])
    except FileNotFoundError:
        return []


def format_message(summary: WarSummary, players: List[PlayerScore]) -> str:
    header = (
        f"**Wojna zakończona: {summary.result}**\n"
        f"**{summary.our_alliance}** {summary.our_score} — {summary.opponent_score} **{summary.opponent_alliance}**\n"
    )
    if summary.war_mode:
        header += f"Tryb: **{summary.war_mode}**" + (" (BETA)\n" if summary.beta_badge else "\n")
    header += "\n"

    roster = load_roster()

    # Always map displayed names to roster (unique matching).
    rank_raw = [(p.rank, p.name_raw) for p in players]
    rank_to_roster = resolve_players_to_roster_unique(rank_raw, roster, ALIASES_PATH)

    lines = []
    for p in sorted(players, key=lambda x: x.rank):
        nick = rank_to_roster.get(p.rank)
        if not nick:
            nick = normalize_display(p.name_raw)
        lines.append(f"[{p.rank:02d}] {nick} — {p.points}")

    return header + "\n".join(lines)


def chunk_message(msg: str, limit: int = 1900) -> list[str]:
    chunks = []
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


def _has_admin_rights(message: discord.Message) -> bool:
    if ADMIN_ROLE_ID == 0:
        return True
    if not isinstance(message.author, discord.Member):
        return False
    return any(r.id == ADMIN_ROLE_ID for r in message.author.roles)


def _parse_bot_message(content: str) -> Tuple[List[str], Dict[int, Tuple[str, int]], List[str]]:
    """
    Splits the bot output into:
    - header_lines: before the list
    - entries: rank -> (name, points)
    - tail_lines: after the list (if any)
    """
    lines = content.splitlines()

    start = None
    for i, ln in enumerate(lines):
        if _RESULT_LINE_RE.match(ln.strip()):
            start = i
            break

    if start is None:
        return lines, {}, []

    end = start
    while end < len(lines) and _RESULT_LINE_RE.match(lines[end].strip()):
        end += 1

    header_lines = lines[:start]
    list_lines = lines[start:end]
    tail_lines = lines[end:]

    entries: Dict[int, Tuple[str, int]] = {}
    for ln in list_lines:
        m = _RESULT_LINE_RE.match(ln.strip())
        if not m:
            continue
        rank = int(m.group(1))
        name = m.group(2).strip()
        points = int(m.group(3))
        entries[rank] = (name, points)

    return header_lines, entries, tail_lines


def _canonical_roster_name(name: str, roster: list[str]) -> Optional[str]:
    if not name:
        return None
    if name in roster:
        return name
    low = name.lower()
    for r in roster:
        if r.lower() == low:
            return r
    return None


def _parse_corrections(text: str) -> List[Dict[str, Any]]:
    """
    Parses corrections from the reply message.
    Supports:
      28 Jaro 250
      28 Jaro +250
      28 +10
      28 240
      28 Jaro
    Returns list of dicts: {rank, name?, points?, is_delta}
    """
    out: List[Dict[str, Any]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _CORR_LINE_RE.match(line)
        if not m:
            continue

        rank = int(m.group(1))
        rest = (m.group(2) or "").strip()
        if not rest:
            continue

        tokens = rest.split()
        last = tokens[-1]

        points = None
        is_delta = False

        if re.fullmatch(r"[+-]?\d+", last):
            points = int(last)
            is_delta = last.startswith(("+", "-"))
            name = " ".join(tokens[:-1]).strip() if len(tokens) > 1 else None
        else:
            name = rest
            points = None
            is_delta = False

        out.append({
            "rank": rank,
            "name": name if name else None,
            "points": points,
            "is_delta": is_delta,
        })
    return out


def _apply_corrections(
    entries: Dict[int, Tuple[str, int]],
    corrections: List[Dict[str, Any]],
    roster: list[str],
) -> Tuple[Dict[int, Tuple[str, int]], List[str]]:
    warnings: List[str] = []
    new_entries = dict(entries)

    for c in corrections:
        rank = int(c["rank"])
        raw_name = c.get("name")
        pts = c.get("points")
        is_delta = bool(c.get("is_delta"))

        if rank <= 0 or rank > 60:
            warnings.append(f"⚠️ Zła pozycja: {rank}")
            continue

        cur_name, cur_pts = new_entries.get(rank, ("", 0))

        # name
        if raw_name:
            canon = _canonical_roster_name(raw_name, roster)
            if not canon:
                warnings.append(f"⚠️ Nick '{raw_name}' nie jest w rosterze – pomijam zmianę nicku na pozycji {rank}.")
                canon = None

            if canon:
                # Preserve uniqueness: if canon already used on another rank -> swap names.
                other_rank = None
                for r, (nm, _p) in new_entries.items():
                    if nm == canon and r != rank:
                        other_rank = r
                        break

                if other_rank is not None:
                    other_name, other_pts = new_entries[other_rank]
                    new_entries[other_rank] = (cur_name, other_pts)
                    cur_name = canon
                else:
                    cur_name = canon

        # points
        if pts is not None:
            if is_delta:
                cur_pts = max(0, int(cur_pts) + int(pts))
            else:
                cur_pts = max(0, int(pts))

        new_entries[rank] = (cur_name, cur_pts)

    return new_entries, warnings


def _rebuild_message(header_lines: List[str], entries: Dict[int, Tuple[str, int]], tail_lines: List[str]) -> str:
    ranks = sorted(entries.keys())
    list_lines = [f"[{r:02d}] {entries[r][0]} — {entries[r][1]}" for r in ranks]

    out_lines = list(header_lines)

    if out_lines and out_lines[-1].strip() != "":
        out_lines.append("")
    elif not out_lines:
        out_lines.append("")

    out_lines.extend(list_lines)

    if tail_lines:
        if tail_lines[0].strip() != "":
            out_lines.append("")
        out_lines.extend(tail_lines)

    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    return "\n".join(out_lines)


def main():
    if not DISCORD_TOKEN:
        raise SystemExit("Brak DISCORD_TOKEN w .env")
    if WATCH_CHANNEL_ID == 0:
        raise SystemExit("Brak WATCH_CHANNEL_ID w .env")

    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)
    keepalive_started = False

    @client.event
    async def on_ready():
        nonlocal keepalive_started
        logging.info("Zalogowano jako: %s | obserwuję kanał: %s", client.user, WATCH_CHANNEL_ID)
        if not keepalive_started:
            keepalive_started = True
            client.loop.create_task(start_keepalive_server())

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        if message.channel.id != WATCH_CHANNEL_ID:
            return

        # --- corrections: reply to the bot's message ---
        if message.reference and message.reference.message_id:
            try:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)
            except Exception:
                ref_msg = None

            if ref_msg and ref_msg.author and client.user and ref_msg.author.id == client.user.id:
                if not _has_admin_rights(message):
                    await message.channel.send("Brak uprawnień do korekt (ADMIN_ROLE_ID).")
                    return

                roster = load_roster()
                header_lines, entries, tail_lines = _parse_bot_message(ref_msg.content)
                if entries:
                    corrections = _parse_corrections(message.content)
                    if corrections:
                        new_entries, warnings = _apply_corrections(entries, corrections, roster)
                        new_content = _rebuild_message(header_lines, new_entries, tail_lines)

                        try:
                            await ref_msg.edit(content=new_content)
                            if warnings:
                                await message.channel.send("\n".join(warnings[:5]))
                        except Exception as e:
                            logging.exception("Nie udało się edytować wiadomości: %s", e)
                            await message.channel.send("Nie udało się edytować wiadomości bota (brak uprawnień?).")
                        return

        # --- standard: parse 2 images ---
        atts = [a for a in message.attachments if is_image(a)]
        if len(atts) < 2:
            return

        images = [await atts[0].read(), await atts[1].read()]

        try:
            summary, players, _debug = await parse_images_in_thread(images)
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

        out = format_message(summary, players)
        for part in chunk_message(out):
            await message.channel.send(part)

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
