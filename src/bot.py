import os
import re
import asyncio
import logging
from typing import List, Optional, Tuple, Dict, Any

from aiohttp import web
import discord

from .config import env_str, env_int
from .openai_parser import parse_war_from_images, WarSummary, PlayerScore
from .nicknames import normalize_with_aliases, normalize_display, roster_autocorrect


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

    port = int(os.getenv("PORT", "8080"))  # Render ustawia PORT
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logging.info("Keepalive HTTP listening on :%s/health", port)


# ----------------------------
# Bot logic
# ----------------------------


def load_roster() -> list[str]:
    import json
    try:
        with open(ROSTER_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("roster", [])
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _normalize_output_nick(name_raw: str, roster: list[str]) -> str:
    """Normalizacja nicku (aliasy -> cleanup -> fuzzy do roster)."""
    nick = (name_raw or "").strip()
    if not nick:
        return "UNKNOWN"

    mapped = normalize_with_aliases(nick, ALIASES_PATH)
    if mapped:
        nick = mapped

    cleaned = normalize_display(nick)
    fixed = roster_autocorrect(cleaned, roster, min_score=88)
    return fixed or "UNKNOWN"


# ----------------------------
# Warnings + manual corrections
# ----------------------------

_PLAYER_LINE_RE = re.compile(r"^\[(\d{1,3})\]\s+(.+?)\s+(?:—|-)\s+(\d+)\s*$")
_MANUAL_CORR_RE = re.compile(r"^\s*\[?\s*(\d{1,3})\s*\]?\s+(.+?)\s+(?:—|-|:)?\s*(\d{1,4})\s*$")
_EXPECTED_MAX_RANK_RE = re.compile(r"Brak odczytu pozycji\s*\(\s*1\s*[–-]\s*(\d{1,3})\s*\)", re.IGNORECASE)


def _analyze_ranks(players: List[PlayerScore], expected_max_rank: Optional[int]) -> Tuple[List[int], List[int], int]:
    """Return (missing_ranks, duplicate_ranks, max_rank_used_for_check)."""
    ranks = [p.rank for p in players if isinstance(p.rank, int) and p.rank > 0]
    if not ranks:
        return ([], [], int(expected_max_rank or 0))

    seen = set()
    dups: List[int] = []
    for r in ranks:
        if r in seen:
            dups.append(r)
        seen.add(r)

    max_rank = int(expected_max_rank or max(ranks))
    missing = [r for r in range(1, max_rank + 1) if r not in seen]
    return (missing, sorted(set(dups)), max_rank)


def _format_warning(players: List[PlayerScore], expected_max_rank: Optional[int], validation_errors: Optional[List[str]]) -> Optional[str]:
    missing, dups, max_rank = _analyze_ranks(players, expected_max_rank)
    parts: List[str] = []

    if missing:
        miss_txt = ", ".join(f"{m:02d}" for m in missing[:15])
        if len(missing) > 15:
            miss_txt += f" …(+{len(missing)-15})"
        parts.append(f"⚠️ Brak odczytu pozycji (1–{max_rank}): **{miss_txt}**.")
    if dups:
        dup_txt = ", ".join(f"{d:02d}" for d in dups[:15])
        if len(dups) > 15:
            dup_txt += f" …(+{len(dups)-15})"
        parts.append(f"⚠️ Duplikaty pozycji: **{dup_txt}**.")

    # Jeśli walidacja parsera zgłasza problem, a nie ma oczywistego missing/dup, pokaż krótki hint.
    if validation_errors and not (missing or dups):
        parts.append("⚠️ Wynik wygląda podejrzanie (walidacja parsera nie przeszła).")

    if not parts:
        return None

    parts.append(
        "Aby uzupełnić / poprawić, odpowiedz (reply) na tę wiadomość np.: `23 ropuch13 250` (możesz podać kilka linii)."
    )
    return "\n".join(parts)


def _parse_manual_corrections(text: str) -> List[Tuple[int, str, int]]:
    """Parse user reply lines like: '23 ropuch13 250' into (rank, nick, points)."""
    out: List[Tuple[int, str, int]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _MANUAL_CORR_RE.match(line)
        if not m:
            continue
        rank = int(m.group(1))
        nick = m.group(2).strip()
        points = int(m.group(3))
        if not nick or rank <= 0:
            continue
        out.append((rank, nick, points))
    return out


def _parse_players_from_bot_message(content: str) -> Tuple[Optional[str], Dict[int, Tuple[str, int]]]:
    """Return (header, players_by_rank). Header includes everything before the first player line."""
    if not content:
        return (None, {})

    lines = content.splitlines()
    first_player_idx = None
    for i, l in enumerate(lines):
        if _PLAYER_LINE_RE.match(l.strip()):
            first_player_idx = i
            break
    if first_player_idx is None:
        return (None, {})

    header = "\n".join(lines[:first_player_idx]).rstrip()
    players: Dict[int, Tuple[str, int]] = {}

    for l in lines[first_player_idx:]:
        mm = _PLAYER_LINE_RE.match(l.strip())
        if not mm:
            continue
        r = int(mm.group(1))
        name = mm.group(2).strip()
        pts = int(mm.group(3))
        players[r] = (name, pts)

    return (header, players)


def _extract_expected_max_rank_from_message(content: str) -> Optional[int]:
    """Spróbuj wyciągnąć expected max rank z warningu w wiadomości bota.

    Przykład warningu:
      ⚠️ Brak odczytu pozycji (1–30): ...
    """
    if not content:
        return None
    m = _EXPECTED_MAX_RANK_RE.search(content)
    if not m:
        return None
    try:
        v = int(m.group(1))
        return v if v > 0 else None
    except Exception:
        return None


def _apply_corrections_to_bot_message(content: str, corrections: List[Tuple[int, str, int]]) -> Optional[str]:
    header, players_by_rank = _parse_players_from_bot_message(content)
    if header is None or not corrections:
        return None

    expected_max_rank = _extract_expected_max_rank_from_message(content)

    roster = load_roster()

    # Apply updates
    for r, nick, pts in corrections:
        players_by_rank[int(r)] = (_normalize_output_nick(nick, roster), int(pts))

    # Rebuild players list
    player_lines = []
    for r in sorted(players_by_rank.keys()):
        name, pts = players_by_rank[r]
        player_lines.append(f"[{r:02d}] {name} — {pts}")

    # Warning based on rebuilt list (jeśli znaliśmy expected_max_rank, użyj go z poprzedniej wiadomości)
    rebuilt_players = [
        PlayerScore(rank=r, name_raw=name, points=pts, name_norm=None)
        for r, (name, pts) in players_by_rank.items()
    ]
    warning = _format_warning(rebuilt_players, expected_max_rank=expected_max_rank, validation_errors=None)

    new_content = header + "\n\n" + "\n".join(player_lines)
    if warning:
        new_content += "\n\n" + warning
    return new_content


# ----------------------------
# Formatting
# ----------------------------


def format_message(summary: WarSummary, players: List[PlayerScore], meta: Optional[Dict[str, Any]] = None) -> str:
    header = (
        f"**Wojna zakończona: {summary.result}**\n"
        f"**{summary.our_alliance}** {summary.our_score} — {summary.opponent_score} **{summary.opponent_alliance}**\n"
    )
    if summary.war_mode:
        header += f"Tryb: **{summary.war_mode}**" + (" (BETA)\n" if summary.beta_badge else "\n")
    header += "\n"

    roster = load_roster()
    lines = []
    for p in sorted(players, key=lambda x: x.rank):
        nick = _normalize_output_nick(p.name_raw, roster)
        lines.append(f"[{p.rank:02d}] {nick} — {p.points}")

    body = "\n".join(lines)

    expected_max_rank = None
    validation_errors: Optional[List[str]] = None
    if meta:
        expected_max_rank = meta.get("chat_expected_max_rank")
        validation_errors = meta.get("chat_validation_errors")

    warning = _format_warning(players, expected_max_rank, validation_errors)
    if warning:
        return header + body + "\n\n" + warning
    return header + body


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
    # uruchamiamy parser (sync) w wątku, żeby nie blokować event loop discord.py
    return await asyncio.to_thread(parse_war_from_images, images, OPENAI_MODEL)


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

        # --- Manual correction mode ---
        if message.reference and (message.content or "").strip():
            corrections = _parse_manual_corrections(message.content)
            if corrections:
                ref_msg = message.reference.resolved
                if ref_msg is None and message.reference.message_id:
                    try:
                        ref_msg = await message.channel.fetch_message(message.reference.message_id)
                    except Exception:
                        ref_msg = None

                if ref_msg and ref_msg.author and client.user and ref_msg.author.id == client.user.id:
                    new_content = _apply_corrections_to_bot_message(ref_msg.content, corrections)
                    if new_content and new_content != ref_msg.content:
                        try:
                            await ref_msg.edit(content=new_content)
                            try:
                                await message.add_reaction("✅")
                            except Exception:
                                pass
                        except Exception as e:
                            logging.exception("Nie udało się zaktualizować wiadomości bota: %s", e)
                            await message.channel.send("Nie udało się zaktualizować wiadomości (błąd edycji).")
                    return

        atts = [a for a in message.attachments if is_image(a)]
        if len(atts) < 2:
            return

        images = [await atts[0].read(), await atts[1].read()]

        try:
            summary, players, _debug, meta = await parse_images_in_thread(images)
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

        out = format_message(summary, players, meta=meta)
        for part in chunk_message(out):
            await message.channel.send(part)

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
