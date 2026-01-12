import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Optional

from rapidfuzz import fuzz, process

_G3W_PREFIX_RE = re.compile(r"^\s*\[g3w\]\s*", re.IGNORECASE)

# Broad homoglyph map (Latin-like)
_CHAR_MAP = {
    # common substitutions / OCR
    "@": "a", "$": "s",
    "0": "o", "1": "i", "!": "i", "¡": "i", "|": "i",
    "£": "l", "ð": "d", "ɭ": "l",
    # IPA / special latin letters that OCR sometimes emits
    "ɾ": "r", "ɼ": "r", "ɽ": "r", "ɹ": "r",
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

    # greek (upper) – często wpadają jako łacińskie
    "Α": "a", "Β": "b", "Ε": "e", "Ζ": "z", "Η": "h", "Ι": "i",
    "Κ": "k", "Μ": "m", "Ν": "n", "Ο": "o", "Ρ": "p", "Τ": "t",
    "Υ": "u", "Χ": "x",

    # cyrillic (subset + extra) – homoglify
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

    # latin variants
    "ø": "o", "Ø": "o",

    # extra Cyrillic variants seen in your cases
    "ѵ": "w",   # izhitsa often used as w
    "Ѡ": "w", "ѡ": "w",

    # small caps / stylistic latin
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ꜰ": "f",
    "ɢ": "g", "ʜ": "h", "ɪ": "i", "ᴊ": "j", "ᴋ": "k", "ʟ": "l",
    "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ʀ": "r", "ᴛ": "t",
    "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "ʏ": "y", "ᴢ": "z",
}

def _strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def canonical_key(raw: str) -> str:
    'Convert nick into a-z0-9 canonical key for matching.'
    s = (raw or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = _G3W_PREFIX_RE.sub("", s)
    s = s.replace("≡", "三")

    out = []
    for ch in s:
        out.append(_CHAR_MAP.get(ch, ch))
    s = "".join(out)

    s = _strip_diacritics(s).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    if s.endswith("tm"):
        s = s[:-2]
    return s

@dataclass
class AliasConfig:
    exact: Dict[str, str]
    canonical: Dict[str, str]
    mtime: float

def _load_aliases(path: str) -> AliasConfig:
    try:
        st = os.stat(path)
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return AliasConfig(
            exact=data.get("exact", {}) or {},
            canonical=data.get("canonical", {}) or {},
            mtime=st.st_mtime,
        )
    except FileNotFoundError:
        return AliasConfig(exact={}, canonical={}, mtime=0.0)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # If aliases.json is saved with unexpected encoding/BOM or is invalid JSON, ignore it.
        return AliasConfig(exact={}, canonical={}, mtime=0.0)

_alias_cache: Optional[AliasConfig] = None

def get_aliases(path: str) -> AliasConfig:
    global _alias_cache
    cur = _load_aliases(path) if _alias_cache is None else _alias_cache
    try:
        st = os.stat(path)
        if st.st_mtime != cur.mtime:
            cur = _load_aliases(path)
    except FileNotFoundError:
        cur = AliasConfig(exact={}, canonical={}, mtime=0.0)
    _alias_cache = cur
    return cur

def normalize_with_aliases(raw: str, aliases_path: str) -> Optional[str]:
    s = (raw or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = _G3W_PREFIX_RE.sub("", s)
    s = " ".join(s.split())
    s = s.replace("≡", "三")

    aliases = get_aliases(aliases_path)

    if s in aliases.exact:
        return aliases.exact[s]

    ck = canonical_key(s)
    if ck in aliases.canonical:
        return aliases.canonical[ck]
    return None

def normalize_display(raw: str) -> str:
    'Minimal cleanup for display when we cannot map.'
    s = (raw or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = _G3W_PREFIX_RE.sub("", s)
    s = " ".join(s.split())
    s = s.replace("≡", "三")
    return s

def roster_autocorrect(name: str, roster: list[str], min_score: int = 88) -> str:
    'Match name to closest roster entry (by canonical key) if similarity is high.'
    if not roster:
        return name

    target_key = canonical_key(name)
    roster_keys = [canonical_key(r) for r in roster]

    match = process.extractOne(target_key, roster_keys, scorer=fuzz.ratio)
    if not match:
        return name

    _best_key, score, idx = match
    if score >= min_score:
        return roster[idx]
    return name


def roster_match(name: str, roster: list[str], min_score: int = 88) -> Optional[str]:
    """Return the best roster entry for *name* if similarity is high enough, else None.

    This is a stricter variant used when we want roster-only output.
    """
    if not roster:
        return None

    target_key = canonical_key(name)
    if not target_key:
        return None

    roster_keys = [canonical_key(r) for r in roster]
    match = process.extractOne(target_key, roster_keys, scorer=fuzz.ratio)
    if not match:
        return None

    _best_key, score, idx = match
    if score >= min_score:
        return roster[idx]
    return None
