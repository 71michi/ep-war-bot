import os
import asyncio
import logging
from typing import List

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


def format_message(summary: WarSummary, players: List[PlayerScore]) -> str:
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
        nick = (p.name_norm or "").strip()

        if not nick:
            mapped = normalize_with_aliases(p.name_raw, ALIASES_PATH)
            if mapped:
                nick = mapped

        if not nick:
            cleaned = normalize_display(p.name_raw)
            nick = roster_autocorrect(cleaned, roster, min_score=88)

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
        # start keepalive server (Render/UptimeRobot)
        asyncio.create_task(start_keepalive_server())
        logging.info("Zalogowano jako: %s | obserwuję kanał: %s", client.user, WATCH_CHANNEL_ID)

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        if message.channel.id != WATCH_CHANNEL_ID:
            return

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
