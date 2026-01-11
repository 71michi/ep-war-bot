# -*- coding: utf-8 -*-
import os
import re
import asyncio
import logging
from typing import List, Optional, Tuple, Dict

from aiohttp import web
import discord

from .config import env_str, env_int
from .openai_parser import parse_war_from_images, WarSummary, PlayerScore
from .nicknames import (
    normalize_with_aliases,
    normalize_display,
    roster_autocorrect,
    assign_unique_roster_names,
)

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

# How many times we re-run the parser and take the "majority" result.
# This helps when the vision model randomly drops/merges a line.
PARSER_RETRIES = env_int("PARSER_RETRIES", 3)

# Minimum similarity for roster_autocorrect when user types correction name
CORRECTION_MIN_SCORE = env_int("CORRECTION_MIN_SCORE", 80)


# ----------------- Keepalive (Render free) -----------------
async def start_keepalive_server():
    app = web.Application()

    async def health(_request):
        return web.Response(text="ok")

    app.router.add_get("/", health)
    app.router.add_get("/health", health)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.getenv("PORT", "8080"))  # Render provides PORT
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info("Keepalive HTTP listening on :%s/health", port)


# ----------------- Roster / formatting -----------------
def load_roster() -> list[str]:
    import json
    try:
        with open(ROSTER_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("roster", [])
    except FileNotFoundError:
        return []
    except Exception:
        logging.exception("Nie mogę odczytać roster.json")
        return []


def _points_diff(summary: WarSummary) -> int:
    # our - opponent (positive => win for us)
    return int(summary.our_score) - int(summary.opponent_score)


def format_message(summary: WarSummary, players: List[PlayerScore], roster_warnings: Optional[List[str]] = None) -> str:
    header = (
        f"**Wojna zakończona: {summary.result}**\n"
        f"**{summary.our_alliance}** {summary.our_score} — {summary.opponent_score} **{summary.opponent_alliance}**\n"
    )
    if summary.war_mode:
        header += f"Tryb: **{summary.war_mode}**" + (" (BETA)\n" if summary.beta_badge else "\n")
    header += "\n"

    lines = []
    for p in sorted(players, key=lambda x: x.rank):
        nick = (p.name_norm or "").strip()
        # name_norm should already be a roster name. If for any reason it's empty:
        if not nick:
            nick = normalize_display(p.name_raw)
        lines.append(f"[{p.rank:02d}] {nick} — {p.points}")

    msg = header + "\n".join(lines)

    if roster_warnings:
        msg += "\n\n" + "\n".join(f"⚠️ {w}" for w in roster_warnings)

    return msg


def chunk_message(msg: str, limit: int = 1900) -> list[str]:
    # Usually unnecessary (30 lines fits), but keep safe.
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


# ----------------- HARD parsing: retry + majority vote -----------------
def _summary_key(s: WarSummary) -> tuple:
    return (
        s.our_alliance,
        s.opponent_alliance,
        s.result,
        int(s.our_score),
        int(s.opponent_score),
        (s.war_mode or "").strip(),
    )


def _players_signature(players: List[PlayerScore]) -> tuple:
    # Signature used to vote across retries (rank, name_norm, points)
    return tuple((p.rank, (p.name_norm or ""), int(p.points)) for p in sorted(players, key=lambda x: x.rank))


def _map_players_to_roster(players: List[PlayerScore], roster: list[str]) -> Tuple[List[PlayerScore], List[str]]:
    """
    Force roster names and uniqueness. Returns (updated players, warnings).
    """
    warnings: List[str] = []
    if not roster:
        warnings.append("Brak roster.json – nie mogę wymusić nicków z roster.")
        return players, warnings

    # Use model-proposed name_norm if it exactly matches roster; otherwise fallback to raw.
    names_for_matching = []
    for p in players:
        if p.name_norm and p.name_norm in roster:
            names_for_matching.append(p.name_norm)
        else:
            # try aliases first (cheap + strong)
            a = normalize_with_aliases(p.name_raw, ALIASES_PATH)
            names_for_matching.append(a if a else p.name_raw)

    assigned, scores = assign_unique_roster_names(names_for_matching, roster, aliases_path=ALIASES_PATH)

    # Build updated list
    updated: List[PlayerScore] = []
    for p, rname, score in zip(players, assigned, scores):
        p.name_norm = rname  # always roster name (or best-effort)
        updated.append(p)
        if score < 80:
            warnings.append(f"Niska pewność dopasowania: '{p.name_raw}' -> '{rname}' (score={score}).")

    # If duplicates somehow remain (shouldn't), warn.
    seen = set()
    dupes = []
    for p in updated:
        if p.name_norm in seen:
            dupes.append(p.name_norm)
        seen.add(p.name_norm)
    if dupes:
        warnings.append("Wykryto duplikaty po dopasowaniu do roster (to nie powinno się zdarzyć): " + ", ".join(sorted(set(dupes))))

    return updated, warnings


def parse_war_hard(images: List[bytes], model: str) -> Tuple[Optional[WarSummary], Optional[List[PlayerScore]], List[str]]:
    """
    Run the OpenAI parser multiple times and pick the most consistent result.
    Returns (summary, players, warnings).
    """
    roster = load_roster()
    attempts = max(1, PARSER_RETRIES)

    candidates: List[Tuple[WarSummary, List[PlayerScore], List[str]]] = []

    for i in range(attempts):
        try:
            summary, players, _debug = parse_war_from_images(images, model=model)
        except Exception as e:
            logging.exception("Parser attempt %s failed: %s", i + 1, e)
            continue

        if not summary or not players:
            continue

        mapped_players, warnings = _map_players_to_roster(players, roster)

        # basic sanity: we expect 29 or 30 (your roster is 30, but sometimes 29 participates)
        if len(mapped_players) < 20:
            warnings.append(f"Parser zwrócił podejrzanie mało graczy ({len(mapped_players)}).")
        candidates.append((summary, mapped_players, warnings))

    if not candidates:
        return None, None, ["Parser nie zwrócił poprawnych danych w żadnej próbie."]

    # 1) Pick most common summary
    from collections import Counter
    summary_counts = Counter(_summary_key(s) for (s, _p, _w) in candidates)
    best_summary_key, _ = summary_counts.most_common(1)[0]
    filtered = [(s, p, w) for (s, p, w) in candidates if _summary_key(s) == best_summary_key]

    # 2) Prefer the most common player-count (29 vs 30), then majority signature inside that
    len_counts = Counter(len(p) for (_s, p, _w) in filtered)
    best_len, best_len_count = len_counts.most_common(1)[0]

    # In a tie, choose larger length
    tied_lens = [l for l, c in len_counts.items() if c == best_len_count]
    best_len = max(tied_lens)

    filtered2 = [(s, p, w) for (s, p, w) in filtered if len(p) == best_len]
    sig_counts = Counter(_players_signature(p) for (_s, p, _w) in filtered2)
    best_sig, _ = sig_counts.most_common(1)[0]

    # Choose the first candidate that matches the best signature
    for s, p, w in filtered2:
        if _players_signature(p) == best_sig:
            return s, p, w

    # Fallback
    s, p, w = filtered2[0]
    return s, p, w


async def parse_images_in_thread(images: List[bytes]):
    return await asyncio.to_thread(parse_war_hard, images, OPENAI_MODEL)


# ----------------- Corrections (reply to bot message) -----------------
_CORR_RE = re.compile(r"^\s*(\d{1,2})\s+(.+?)\s+([+-]?\d+)\s*$")
_LINE_RE = re.compile(r"^\[(\d{2})\]\s+(.+?)\s+[—\-]\s+(\d+)\s*$")


def _apply_correction_to_content(content: str, rank: int, new_name: str, new_points_str: str) -> Tuple[Optional[str], str]:
    """
    Returns (new_content or None, error_message).
    """
    lines = content.splitlines()
    header_lines: List[str] = []
    list_lines: List[str] = []

    started = False
    for ln in lines:
        if _LINE_RE.match(ln):
            started = True
            list_lines.append(ln)
        else:
            if not started:
                header_lines.append(ln)
            else:
                # trailing text after list (warnings)
                list_lines.append(ln)

    # Extract entries
    entries: Dict[int, Tuple[str, int]] = {}
    other_lines: List[str] = []
    for ln in list_lines:
        m = _LINE_RE.match(ln)
        if not m:
            other_lines.append(ln)
            continue
        r = int(m.group(1))
        nm = m.group(2).strip()
        pts = int(m.group(3))
        entries[r] = (nm, pts)

    if not entries:
        return None, "Nie widzę listy [01] ... w tej wiadomości bota."

    if rank not in entries:
        return None, f"Nie ma pozycji [{rank:02d}] w tej wiadomości."

    # Validate points (set or delta)
    cur_name, cur_pts = entries[rank]
    if new_points_str.startswith(("+", "-")):
        new_pts = cur_pts + int(new_points_str)
    else:
        new_pts = int(new_points_str)

    # Enforce unique roster names: refuse if name already used at other rank
    for r, (nm, _pts) in entries.items():
        if r != rank and nm == new_name:
            return None, f"Nick '{new_name}' już jest na pozycji [{r:02d}]. Najpierw popraw tamtą pozycję."

    entries[rank] = (new_name, new_pts)

    # Rebuild list in ascending rank order, but keep extra lines (warnings) after list
    rebuilt_list = []
    for r in sorted(entries.keys()):
        nm, pts = entries[r]
        rebuilt_list.append(f"[{r:02d}] {nm} — {pts}")

    # Preserve any trailing non-list lines after list (warnings)
    # Keep them only if they are not empty.
    trailing = [ln for ln in other_lines if ln.strip()]
    new_content = "\n".join(header_lines + rebuilt_list + ([""] + trailing if trailing else []))
    return new_content, ""


# ----------------- Discord bot -----------------
def main():
    if not DISCORD_TOKEN:
        raise SystemExit("Brak DISCORD_TOKEN w .env / ENV")
    if WATCH_CHANNEL_ID == 0:
        raise SystemExit("Brak WATCH_CHANNEL_ID w .env / ENV")

    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        logging.info("Zalogowano jako: %s | obserwuję kanał: %s", client.user, WATCH_CHANNEL_ID)
        # Start keepalive server (Render)
        asyncio.create_task(start_keepalive_server())

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        if message.channel.id != WATCH_CHANNEL_ID:
            return

        # --- 1) Correction command (reply to a bot message) ---
        m = _CORR_RE.match(message.content or "")
        if m and message.reference:
            try:
                rank = int(m.group(1))
                name_typed = m.group(2).strip()
                pts_str = m.group(3).strip()
            except Exception:
                return

            if not (1 <= rank <= 30):
                await message.reply("Pozycja musi być w zakresie 1..30.")
                return

            # Resolve roster name (must be from roster)
            roster = load_roster()
            if not roster:
                await message.reply("Brak roster.json – nie mogę zastosować korekty nicku.")
                return

            # Exact match or fuzzy to roster
            name_norm = ""
            for r in roster:
                if name_typed.lower() == r.lower():
                    name_norm = r
                    break
            if not name_norm:
                name_norm = roster_autocorrect(name_typed, roster, min_score=CORRECTION_MIN_SCORE)

            if not name_norm:
                await message.reply(f"Nie znam takiego nicku w roster: '{name_typed}'.")
                return

            # Fetch referenced message
            try:
                ref = message.reference.resolved
                if ref is None:
                    ref = await message.channel.fetch_message(message.reference.message_id)  # type: ignore[arg-type]
            except Exception:
                await message.reply("Nie mogę pobrać wiadomości, do której odpowiadasz.")
                return

            if not isinstance(ref, discord.Message) or ref.author != client.user:
                await message.reply("Ta komenda działa tylko jako odpowiedź na wiadomość bota z wynikami.")
                return

            new_content, err = _apply_correction_to_content(ref.content or "", rank, name_norm, pts_str)
            if not new_content:
                await message.reply(err or "Nie udało się zastosować korekty.")
                return

            try:
                await ref.edit(content=new_content)
                try:
                    await message.add_reaction("✅")
                except Exception:
                    pass
            except Exception:
                logging.exception("Nie udało się zedytować wiadomości bota.")
                await message.reply("Nie udało się zedytować wiadomości bota (sprawdź uprawnienia).")
            return

        # --- 2) Normal flow: parse two screenshots ---
        atts = [a for a in message.attachments if is_image(a)]
        if len(atts) < 2:
            return

        images = [await atts[0].read(), await atts[1].read()]

        try:
            summary, players, warnings = await parse_images_in_thread(images)
        except Exception as e:
            logging.exception("Błąd parsowania: %s", e)
            await message.channel.send("Nie udało się odczytać screenów (błąd po stronie parsera).")
            return

        if not summary or not players:
            await message.channel.send(
                "Nie udało się jednoznacznie odczytać obu screenów (brak summary albo listy). "
                "Upewnij się, że widać cały panel z listą i paski wyniku."
            )
            return

        out = format_message(summary, players, roster_warnings=warnings if warnings else None)
        for part in chunk_message(out):
            await message.channel.send(part)

    client.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()
