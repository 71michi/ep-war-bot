# -*- coding: utf-8 -*-
"""
Nick normalization utilities for Empires & Puzzles war screenshots.

Goals:
- Normalize stylized / homoglyph nicknames into a canonical form for matching.
- Map nicknames to your alliance roster (roster.json) so the bot always outputs roster names.
- Support aliases.json (exact + canonical maps).
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Optional, Tuple, List

from rapidfuzz import fuzz


# --- Character map (homoglyphs, OCR substitutions, stylistic variants) ---
# Note: Values can be multiple characters (e.g. "ч" -> "ch").
_CHAR_MAP: Dict[str, str] = {
    # common substitutions / OCR
    "@": "a", "$": "s",
    "0": "o", "1": "i", "!": "i", "¡": "i", "|": "i",
    "£": "l", "ð": "d", "ɭ": "l",
    "√": "w",

    # extra common OCR / symbols
    "€": "e", "¢": "c", "¥": "y",
    "§": "s",
    "ß": "b",
    "™": "", "®": "", "©": "c",
    "°": "", "•": "", "·": "", "⋅": "",
    "—": "", "–": "", "-": "",
    "’": "", "'": "", "`": "",
    "“": "", "”": "", '"': "",

    # greek (lower)
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z",
    "η": "n", "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "v",
    "ο": "o", "ρ": "p", "σ": "s", "ς": "s", "τ": "t", "υ": "u", "ω": "w",

    # greek (upper) – often appear as latin-like
    "Α": "a", "Β": "b", "Ε": "e", "Ζ": "z", "Η": "h", "Ι": "i",
    "Κ": "k", "Μ": "m", "Ν": "n", "Ο": "o", "Ρ": "p", "Τ": "t",
    "Υ": "u", "Χ": "x",

    # cyrillic (subset + extra) – homoglyphs
    "А": "a", "а": "a",
    "В": "b", "в": "b",
    "Б": "b", "б": "b",
    "Е": "e", "е": "e",
    "Є": "e", "є": "e",
    "Ё": "e", "ё": "e",

    "Г": "g", "г": "g",
    "Д": "d", "д": "d",
    "Л": "l", "л": "l",
    "П": "p", "п": "p",
    "З": "z", "з": "z",
    "Ч": "ch", "ч": "ch",
    "Ф": "f", "ф": "f",

    "И": "i", "и": "i",
    "І": "i", "і": "i",
    "К": "k", "к": "k",
    "М": "m", "м": "m",
    "Н": "h", "н": "h",
    "О": "o", "о": "o",
    "Р": "p", "р": "p",
    "С": "c", "с": "c",
    "Т": "t", "т": "t",
    "У": "u", "у": "u",
    "Й": "y", "й": "y",
    "Я": "r", "я": "r",
    "Ш": "sh", "ш": "sh",
    "Ѕ": "s", "ѕ": "s",

    "Х": "x", "х": "x",
    "Ы": "y", "ы": "y",
    "Ж": "zh", "ж": "zh",
    "Ц": "c", "ц": "c",

    # latin special letters (seen in your cases)
    "ø": "o", "Ø": "o",

    # extra Cyrillic variants seen in your cases
    "ѵ": "w",   # izhitsa often used as "w"
    "Ѡ": "w", "ѡ": "w",

    # small caps / stylistic latin
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ꜰ": "f",
    "ɢ": "g", "ʜ": "h", "ɪ": "i", "ᴊ": "j", "ᴋ": "k", "ʟ": "l",
    "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ʀ": "r", "ᴛ": "t",
    "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "ʏ": "y", "ᴢ": "z",
}


_G3W_PREFIX_RE = re.compile(r"^\s*\[\s*g3w\s*\]\s*", re.IGNORECASE)


def strip_g3w_prefix(name: str) -> str:
    return _G3W_PREFIX_RE.sub("", (name or "").strip())


def _strip_diacritics(s: str) -> str:
    # Keep base letters, remove combining marks
    nkfd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nkfd if not unicodedata.combining(ch))


def normalize_display(name: str) -> str:
    """
    A "cleaned" human-readable version of the raw nickname (for debugging / fallback).
    We do NOT force roster formatting here.
    """
    s = strip_g3w_prefix(name)
    s = s.strip()

    # Replace mapped chars
    out = []
    for ch in s:
        if ch in _CHAR_MAP:
            out.append(_CHAR_MAP[ch])
        else:
            out.append(ch)
    s = "".join(out)

    # Trim obvious decorations around
    s = re.sub(r"[⟪⟫《》◊◇◆★☆\u2605\u2606]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize(name: str) -> str:
    """
    Canonical form for matching: lowercase ascii-ish, alnum only.
    """
    s = normalize_display(name).lower()
    s = _strip_diacritics(s)

    # Replace any leftover whitespace/punctuation with nothing
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


@lru_cache(maxsize=64)
def _load_aliases_cached(path: str) -> Dict[str, Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        exact = {str(k): str(v) for k, v in (data.get("exact") or {}).items()}
        canonical = {canonicalize(k): str(v) for k, v in (data.get("canonical") or {}).items()}
        return {"exact": exact, "canonical": canonical}
    except FileNotFoundError:
        return {"exact": {}, "canonical": {}}
    except Exception:
        # If aliases file is broken, fail "soft"
        return {"exact": {}, "canonical": {}}


def normalize_with_aliases(name_raw: str, aliases_path: str = "aliases.json") -> str:
    """
    Returns mapped nickname based on aliases.json or empty string if not matched.
    """
    aliases = _load_aliases_cached(os.path.abspath(aliases_path))
    raw = strip_g3w_prefix(name_raw)

    if raw in aliases["exact"]:
        return aliases["exact"][raw].strip()

    c = canonicalize(raw)
    if c in aliases["canonical"]:
        return aliases["canonical"][c].strip()

    return ""


def roster_autocorrect(name: str, roster: List[str], min_score: int = 88) -> str:
    """
    Fuzzy-match 'name' to one of roster names. Returns roster name or empty string.
    """
    if not roster:
        return ""

    cn = canonicalize(name)
    if not cn:
        return ""

    best_name = ""
    best_score = -1

    for r in roster:
        score = fuzz.WRatio(cn, canonicalize(r))
        if score > best_score:
            best_score = score
            best_name = r

    if best_score >= min_score:
        return best_name
    return ""


def map_name_to_roster(name_raw: str, roster: List[str], aliases_path: str = "aliases.json", min_score: int = 88) -> Tuple[str, int]:
    """
    Map a raw nickname to a roster name.

    Returns (roster_name, confidence_score_0_100).
    """
    if not roster:
        return ("", 0)

    # 1) alias exact/canonical
    a = normalize_with_aliases(name_raw, aliases_path)
    if a:
        # Ensure it's actually in roster; if not, try match it to roster
        if a in roster:
            return (a, 100)
        m = roster_autocorrect(a, roster, min_score=70)
        if m:
            return (m, 95)

    # 2) direct exact (case-insensitive)
    raw = strip_g3w_prefix(name_raw).strip()
    for r in roster:
        if raw.lower() == r.lower():
            return (r, 100)

    # 3) fuzzy to roster
    cn = canonicalize(name_raw)
    best_name = ""
    best_score = -1
    for r in roster:
        score = fuzz.WRatio(cn, canonicalize(r))
        if score > best_score:
            best_score = score
            best_name = r

    if best_name:
        return (best_name, int(best_score))
    return ("", 0)


def assign_unique_roster_names(
    raw_names: List[str],
    roster: List[str],
    aliases_path: str = "aliases.json",
) -> Tuple[List[str], List[int]]:
    """
    Assign each raw name to a UNIQUE roster name (greedy max-score matching).
    Returns (assigned_roster_names, confidence_scores).
    """
    n_players = len(raw_names)
    if n_players == 0:
        return ([], [])

    # Precompute scores for all (player, roster) pairs.
    pairs = []
    raw_canons = [canonicalize(x) for x in raw_names]
    roster_canons = [canonicalize(r) for r in roster]

    # Alias boosting: if alias maps to a roster name, give it a strong bonus.
    alias_map = _load_aliases_cached(os.path.abspath(aliases_path))
    alias_to_roster = {}
    # exact alias can include decorations - keep raw exact keys
    for k, v in alias_map.get("exact", {}).items():
        alias_to_roster[k] = v
    for k, v in alias_map.get("canonical", {}).items():
        alias_to_roster[k] = v

    for i in range(n_players):
        raw = strip_g3w_prefix(raw_names[i]).strip()
        raw_c = raw_canons[i]
        for j in range(len(roster)):
            base = fuzz.WRatio(raw_c, roster_canons[j])

            bonus = 0
            # Exact alias hit
            if raw in alias_map["exact"] and alias_map["exact"][raw] == roster[j]:
                bonus = 30
            # Canonical alias hit
            if raw_c in alias_map["canonical"] and alias_map["canonical"][raw_c] == roster[j]:
                bonus = max(bonus, 25)

            score = min(100, int(base + bonus))
            pairs.append((score, i, j))

    pairs.sort(reverse=True, key=lambda x: x[0])

    assigned_player = [False] * n_players
    assigned_roster = [False] * len(roster)
    out_names = [""] * n_players
    out_scores = [0] * n_players

    for score, i, j in pairs:
        if assigned_player[i] or assigned_roster[j]:
            continue
        assigned_player[i] = True
        assigned_roster[j] = True
        out_names[i] = roster[j]
        out_scores[i] = score

    # Any unassigned players: assign remaining roster entries (if any), best available score.
    remaining_roster = [idx for idx, used in enumerate(assigned_roster) if not used]
    for i in range(n_players):
        if out_names[i]:
            continue
        if remaining_roster:
            j = remaining_roster.pop(0)
            out_names[i] = roster[j]
            out_scores[i] = int(fuzz.WRatio(raw_canons[i], roster_canons[j]))
        else:
            # Shouldn't happen (players > roster), but keep safe
            out_names[i] = raw_names[i]
            out_scores[i] = 0

    return out_names, out_scores
