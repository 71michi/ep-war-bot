import json
import os
import re
from typing import Optional, Dict, Any, List, Tuple

from rapidfuzz import fuzz


# -----------------------------
# Character normalization map
# -----------------------------
# NOTE:
# - This map is used to build a comparison-friendly form for OCR / stylized nicknames.
# - We keep Polish diacritics intact here; folding is done separately in _key().
_CHAR_MAP: dict[str, str] = {
    # common substitutions / OCR
    "@": "a",
    "$": "s",
    "0": "o",
    "1": "i",
    "!": "i",
    "¡": "i",
    "|": "i",
    "£": "l",
    "ð": "d",
    "ɭ": "l",
    "√": "w",

    # extra common OCR / symbols
    "€": "e",
    "¢": "c",
    "¥": "y",
    "§": "s",
    "ß": "b",

    # trademark-ish / punctuation we usually want to drop
    "™": "",
    "®": "",
    "©": "c",
    "°": "",
    "•": "",
    "·": "",
    "⋅": "",
    "—": "",
    "–": "",
    "’": "",
    "'": "",
    "`": "",
    "“": "",
    "”": "",
    '"': "",

    # greek (lower)
    "α": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "ζ": "z",
    "η": "n",
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "v",
    "ο": "o",
    "ρ": "p",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "υ": "u",
    "ω": "w",

    # greek (upper) – often confused with latin
    "Α": "a",
    "Β": "b",
    "Ε": "e",
    "Ζ": "z",
    "Η": "h",
    "Ι": "i",
    "Κ": "k",
    "Μ": "m",
    "Ν": "n",
    "Ο": "o",
    "Ρ": "p",
    "Τ": "t",
    "Υ": "u",
    "Χ": "x",

    # cyrillic (subset + homoglyphs)
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

    # latin-ish
    "ø": "o",
    "Ø": "o",

    # extra Cyrillic variants seen in your cases
    "ѵ": "w",   # izhitsa often used as "w"
    "Ѡ": "w",
    "ѡ": "w",

    # small caps / stylistic latin
    "ᴀ": "a",
    "ʙ": "b",
    "ᴄ": "c",
    "ᴅ": "d",
    "ᴇ": "e",
    "ꜰ": "f",
    "ɢ": "g",
    "ʜ": "h",
    "ɪ": "i",
    "ᴊ": "j",
    "ᴋ": "k",
    "ʟ": "l",
    "ᴍ": "m",
    "ɴ": "n",
    "ᴏ": "o",
    "ᴘ": "p",
    "ʀ": "r",
    "ᴛ": "t",
    "ᴜ": "u",
    "ᴠ": "v",
    "ᴡ": "w",
    "ʏ": "y",
    "ᴢ": "z",
}

# Polish folding used only for matching keys
_PL_FOLD = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ż": "z", "ź": "z",
    "Ą": "a", "Ć": "c", "Ę": "e", "Ł": "l", "Ń": "n", "Ó": "o", "Ś": "s", "Ż": "z", "Ź": "z",
})


# -----------------------------
# Basic normalization utilities
# -----------------------------

_PREFIX_RE = re.compile(r"^\s*\[[^\]]*\]\s*")  # leading [tag] like [g3w]
_BRACKET_TAG_RE = re.compile(r"(\[[^\]]*\])", flags=re.UNICODE)

# decorative brackets/stars etc. We'll trim from both ends repeatedly
_TRIM_CHARS = " \t\n\r\u200b•·⋅—–-:|<>[](){}⟪⟫《》◊◇♦★☆✦✧✩✪✫✬✭✮✯❖⌂™®"

def strip_prefix_tags(name: str) -> str:
    """
    Removes leading bracket tags like [g3w] (one or more).
    """
    s = (name or "").strip()
    while True:
        m = _PREFIX_RE.match(s)
        if not m:
            break
        s = s[m.end():].lstrip()
    return s

def normalize_display(name: str) -> str:
    """
    Cleans a raw nickname from OCR/stylized symbols into a readable form.
    This function is used for matching keys and as a display fallback.
    """
    s = strip_prefix_tags(name)

    # map characters
    out = []
    for ch in s:
        out.append(_CHAR_MAP.get(ch, ch))
    s = "".join(out)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # trim decorative chars from ends
    s = s.strip(_TRIM_CHARS)

    # remove duplicate spaces again
    s = re.sub(r"\s+", " ", s).strip()

    return s

def _key(s: str) -> str:
    """
    Comparison key used for fuzzy matching.
    - normalize_display (removes stylings)
    - fold Polish diacritics
    - lower
    - keep only [a-z0-9 ] and collapse spaces
    """
    s = normalize_display(s)
    s = s.translate(_PL_FOLD)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


# -----------------------------
# Aliases file support
# -----------------------------

def _load_aliases(aliases_path: str) -> Dict[str, Any]:
    try:
        with open(aliases_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def normalize_with_aliases(name_raw: str, aliases_path: str) -> Optional[str]:
    """
    Uses aliases.json with structure like:
    {
      "exact": { "яга": "Jaro", ... },
      "canonical": { "bush22": "Bush22", "raga": "Jaro", ... }
    }
    """
    data = _load_aliases(aliases_path)
    exact = data.get("exact", {}) or {}
    canonical = data.get("canonical", {}) or {}

    raw = (name_raw or "").strip()

    # exact match (raw, or raw after simple strip of leading tags)
    if raw in exact:
        return exact[raw]
    stripped = strip_prefix_tags(raw)
    if stripped in exact:
        return exact[stripped]

    # canonical match by key
    k = _key(raw)
    if k in canonical:
        return canonical[k]

    # also try key of stripped (helps if raw has [g3w])
    ks = _key(stripped)
    if ks in canonical:
        return canonical[ks]

    return None


# -----------------------------
# Roster unique resolver (nicki zawsze jak roster)
# -----------------------------

def resolve_players_to_roster_unique(
    name_raw_by_rank: List[Tuple[int, str]],
    roster: List[str],
    aliases_path: str,
    min_score_warn: int = 78,
) -> Dict[int, str]:
    """
    Returns mapping: rank -> EXACT nickname from roster.

    Properties:
    - Always outputs only names from roster (if roster is non-empty).
    - Unique assignment: each roster name used at most once.
    - If there are fewer players (e.g. 29) than roster (e.g. 30), one roster member will remain unused (OK).
    """
    if not roster:
        return {rank: normalize_display(raw) for rank, raw in name_raw_by_rank}

    # 1) pre-assign via aliases if alias points to an exact roster name
    assigned_rank_to_roster: Dict[int, str] = {}
    used_roster: set[str] = set()

    pending: List[Tuple[int, str]] = []
    for rank, raw in name_raw_by_rank:
        ali = normalize_with_aliases(raw, aliases_path)
        if ali and ali in roster and ali not in used_roster:
            assigned_rank_to_roster[rank] = ali
            used_roster.add(ali)
        else:
            pending.append((rank, raw))

    if not pending:
        return assigned_rank_to_roster

    # 2) build all edges (score) for remaining
    roster_keys = [_key(r) for r in roster]
    pending_keys = [_key(raw) for _rank, raw in pending]

    edges: List[Tuple[int, int, int]] = []  # (score, pending_i, roster_j)
    for i, pk in enumerate(pending_keys):
        for j, rk in enumerate(roster_keys):
            if roster[j] in used_roster:
                continue
            score = fuzz.WRatio(pk, rk)
            edges.append((score, i, j))

    edges.sort(reverse=True, key=lambda x: x[0])

    assigned_pending: set[int] = set()
    assigned_roster_idx: set[int] = set()

    # 3) greedy unique matching
    for score, i, j in edges:
        if i in assigned_pending or j in assigned_roster_idx:
            continue
        rname = roster[j]
        if rname in used_roster:
            continue

        rank, _raw = pending[i]
        assigned_rank_to_roster[rank] = rname
        used_roster.add(rname)
        assigned_pending.add(i)
        assigned_roster_idx.add(j)

    # 4) fallback: assign remaining ranks to remaining roster (best remaining match)
    remaining_roster = [r for r in roster if r not in used_roster]
    for i, (rank, raw) in enumerate(pending):
        if i in assigned_pending:
            continue

        if remaining_roster:
            pk = _key(raw)
            best = max(remaining_roster, key=lambda r: fuzz.WRatio(pk, _key(r)))
            assigned_rank_to_roster[rank] = best
            remaining_roster.remove(best)
        else:
            assigned_rank_to_roster[rank] = roster[0]

    return assigned_rank_to_roster
