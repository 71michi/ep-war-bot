from __future__ import annotations
from typing import List
from .schemas import WarSummary, PlayerScore
from .nicknames import normalize_nick

def build_discord_message(summary: WarSummary, players: List[PlayerScore]) -> str:
    lines = []
    for p in sorted(players, key=lambda x: x.rank):
        nick = normalize_nick(p.name_raw)
        # Format bez markdown-list (Discord lubi "1." przerabiać na listę)
        lines.append(f"[{p.rank:02d}] {nick} — {p.points}")

    header = (
        f"**Wojna zakończona: {summary.result}**\n"
        f"**{summary.our_alliance}** {summary.our_score} — {summary.opponent_score} **{summary.opponent_alliance}**\n"
    )
    if summary.war_mode:
        header += f"Tryb: **{summary.war_mode}**" + (" (BETA)\n" if summary.beta_badge else "\n")
    else:
        header += "\n"

    return header + "\n" + "\n".join(lines)

def chunk_message(msg: str, limit: int = 1900) -> list[str]:
    chunks = []
    msg = msg.strip()
    while len(msg) > limit:
        cut = msg.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(msg[:cut])
        msg = msg[cut:].lstrip("\n")
    if msg:
        chunks.append(msg)
    return chunks
