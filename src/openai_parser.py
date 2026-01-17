import base64
import io
import json
import re
import logging
from typing import List, Optional, Literal, Tuple, Dict

from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image
from PIL import ImageEnhance, ImageFilter
from rapidfuzz import fuzz as rf_fuzz

from .config import env_str, env_int, env_bool, env_float
from .nicknames import canonical_key
from .logging_setup import setup_logging, set_trace_id, reset_trace_id

setup_logging()
logger = logging.getLogger("warbot.openai_parser")

client = OpenAI()


# ----------------------------
# Models (Structured Outputs)
# ----------------------------

class WarModeOnly(BaseModel):
    war_mode: Optional[str] = None


class ExpectedMaxRankOnly(BaseModel):
    expected_max_rank: Optional[int] = None


class PlayerScore(BaseModel):
    rank: int
    name_raw: str
    points: int
    name_norm: Optional[str] = None  # chosen from roster when confident


class ChatResults(BaseModel):
    title: str = Field(default="Najlepsi atakujący na wojnach")
    players: List[PlayerScore]
    # Największy numer pozycji widoczny na screenie (np. 30, 25, 20...).
    # Używane do twardej walidacji kompletności.
    expected_max_rank: Optional[int] = None


# Wersja "slice" do robust OCR: działa na wycinkach listy,
# więc nie wymaga tytułu.
class ChatSliceResults(BaseModel):
    players: List[PlayerScore]


class WarSummary(BaseModel):
    our_alliance: str
    opponent_alliance: str
    result: Literal["Zwycięstwo", "Porażka"]
    our_score: int
    opponent_score: int
    war_mode: Optional[str] = None
    beta_badge: Optional[bool] = None


class ParsedImage(BaseModel):
    kind: Literal["chat_results", "war_summary", "unknown"]
    chat_results: Optional[ChatResults] = None
    war_summary: Optional[WarSummary] = None
    confidence: float = Field(ge=0, le=1, default=0.7)
    notes: Optional[str] = None


class ExpectedMaxRankOnly(BaseModel):
    expected_max_rank: Optional[int] = None


# ----------------------------
# Helpers
# ----------------------------

def _ensure_png_bytes(image_bytes: bytes) -> bytes:
    """Always convert input bytes to PNG.

    Some phones/Discord uploads send JPG/WEBP. In older versions we declared
    mime=image/png but passed non-PNG bytes, which can lower vision accuracy.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = (img.format or "").upper()
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out = buf.getvalue()
        if fmt and fmt != "PNG":
            logger.debug("Converted input image %s -> PNG (%d bytes -> %d bytes)", fmt, len(image_bytes), len(out))
        else:
            logger.debug("Input image already decodable; normalized to PNG (%d bytes -> %d bytes)", len(image_bytes), len(out))
        return out
    except Exception:
        # If PIL can't decode it, fall back to original bytes.
        logger.debug("PIL decode failed; using original bytes (%d bytes)", len(image_bytes))
        return image_bytes

def _to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _load_roster(roster_path: str) -> list[str]:
    try:
        with open(roster_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        r = data.get("roster", [])
        return [str(x) for x in r if str(x).strip()]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def _preprocess_chat_image(image_bytes: bytes) -> Image.Image:
    """Lekki preprocessing pod OCR listy rankingowej."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    # Utnij górę (nagłówek CHAT / zakładki) i dół (pole pisania)
    # To są proporcje, które dobrze działają na typowych screenach z telefonu.
    y1 = int(h * 0.23)
    y2 = int(h * 0.86)
    x1 = int(w * 0.02)
    x2 = int(w * 0.98)

    logger.debug("Preprocess chat ROI crop: (x1=%d, y1=%d, x2=%d, y2=%d) from (w=%d, h=%d)", x1, y1, x2, y2, w, h)
    roi = img.crop((x1, y1, x2, y2))

    # Powiększ, zwiększ kontrast i delikatnie wyostrz.
    roi = roi.resize((roi.width * 2, roi.height * 2), Image.LANCZOS)
    roi = ImageEnhance.Contrast(roi).enhance(1.35)
    roi = roi.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    logger.debug("Preprocess chat ROI output size: %s", roi.size)
    return roi


def _make_overlapping_slices(img: Image.Image, parts: int = 3, overlap: float = 0.12) -> List[bytes]:
    """Dzieli obraz pionowo na części z nakładką (overlap) i zwraca PNG bytes."""
    w, h = img.size
    parts = max(1, int(parts))
    overlap = max(0.0, min(float(overlap), 0.35))

    base = h / parts
    ov = int(base * overlap)

    slices: List[bytes] = []
    for i in range(parts):
        y1 = int(i * base) - (ov if i > 0 else 0)
        y2 = int((i + 1) * base) + (ov if i < parts - 1 else 0)
        y1 = max(0, y1)
        y2 = min(h, y2)
        logger.debug("Slice %d/%d: y1=%d y2=%d (overlap=%d)", i + 1, parts, y1, y2, ov)
        crop = img.crop((0, y1, w, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        slices.append(buf.getvalue())
    return slices


_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def _is_suspicious_name(name_raw: str) -> bool:
    """Heuristic: detect OCR garbage that often indicates misread lines.

    We deliberately keep it conservative: Polish diacritics are fine; Cyrillic
    letters or heavy symbol noise are not.
    """
    s = (name_raw or "").strip()
    if not s:
        return True
    if _CYRILLIC_RE.search(s):
        return True
    if any(ch in s for ch in ("†", "§", "€")):
        return True
    ck = canonical_key(s)
    if len(ck) < 3:
        return True
    return False


def _detect_expected_max_rank(image_bytes: bytes, model: str) -> Optional[int]:
    """Try to read the largest visible rank number from the bottom of the list."""
    try:
        pre = _preprocess_chat_image(image_bytes)
        w, h = pre.size
        # Bottom part of the list, left side where [n] appears
        y1 = int(h * 0.76)
        x2 = int(w * 0.42)
        crop = pre.crop((0, y1, x2, h))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        crop_bytes = buf.getvalue()
        data_url = _to_data_url(crop_bytes)
    except Exception:
        return None

    prompt = (
        "Na tym fragmencie jest dół listy rankingowej z pozycjami w formacie [n]. "
        "Podaj TYLKO największy widoczny numer pozycji jako expected_max_rank. "
        "Jeśli nie widać żadnych [n], zwróć null."
    )

    timeout_s = env_int("OPENAI_TIMEOUT_SECONDS", 90)
    logger.debug("Detecting expected_max_rank via bottom-crop OCR")
    logger.debug("expected_max_rank OCR request: bytes=%d model=%s", len(crop_bytes), model)

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Jesteś OCR. Zwróć tylko expected_max_rank."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=ExpectedMaxRankOnly,
        temperature=0,
        timeout=timeout_s,
    )

    out = resp.output_parsed
    if out and out.expected_max_rank and out.expected_max_rank > 0:
        emr = int(out.expected_max_rank)
        logger.debug("Detected expected_max_rank=%d", emr)
        return emr
    logger.debug("Failed to detect expected_max_rank")
    return None



def _parse_chat_slice(slice_bytes: bytes, model: str, roster: list[str]) -> List[PlayerScore]:
    roster_text = "\n".join(f"- {n}" for n in roster) if roster else "(brak)"
    system_instructions = f"""Jesteś OCR-em listy rankingowej z gry Empires & Puzzles.
To jest WYCIĘTY FRAGMENT listy „Najlepsi atakujący w wojnach”.

Masz odczytać WSZYSTKIE widoczne wiersze rankingu.

Każdy wiersz ma format zbliżony do:
  [23] [g3w] ropuch13 192

Zasady krytyczne:
- NIE POMIJAJ żadnego wiersza.
- Jeśli rank i punkty są czytelne, ale nick jest trudny: przepisz możliwie najbliżej; w ostateczności ustaw name_raw na '???' (ale NIE pomijaj wiersza).
- Nie duplikuj rang; każdy widoczny [n] ma trafić do outputu dokładnie raz.
- Nick może zawierać małe litery i cyfry (np. ropuch13) oraz ozdobniki. Przepisz je do name_raw.
- points to ostatnia liczba w wierszu.
- rank to liczba w nawiasie kwadratowym [n].
- name_norm: jeżeli jesteś PEWNY, wybierz DOKŁADNIE jeden z nicków z ROSTER, inaczej null.

ROSTER:
{roster_text}
""".strip()

    data_url = _to_data_url(slice_bytes)
    timeout_s = env_int("OPENAI_TIMEOUT_SECONDS", 90)

    logger.debug("Parsing chat slice (%d bytes) with model=%s", len(slice_bytes), model)
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Odczytaj ranking z widocznych wierszy."},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=ChatSliceResults,
        temperature=0,
        timeout=timeout_s,
    )

    out = resp.output_parsed
    players = out.players if out and out.players else []
    # sanitize: some models return literal strings like "null" for optional fields
    for p in players:
        if isinstance(p.name_norm, str) and p.name_norm.strip().lower() in {"null", "none", "nil", "n/a"}:
            p.name_norm = None
    logger.debug("Chat slice parsed: %d players", len(players))
    return players


def _merge_players(lists: List[List[PlayerScore]]) -> List[PlayerScore]:
    """Scala wyniki z kilku slice'ów po rank i wybiera "lepszy" rekord."""
    by_rank: Dict[int, PlayerScore] = {}

    def _norm_name_norm(nn: Optional[str]) -> Optional[str]:
        if nn is None:
            return None
        if not isinstance(nn, str):
            return None
        t = nn.strip()
        if not t:
            return None
        if t.lower() in {"null", "none", "nil", "n/a"}:
            return None
        return t

    def score(p: PlayerScore) -> int:
        # Heurystyka: preferuj dłuższe name_raw (mniej obcięte) i niezerowe points
        s = 0
        if p.name_raw:
            s += min(len(p.name_raw), 40)
        if p.points and p.points > 0:
            s += 20
        # Jeśli model podał name_norm z roster, to zwykle oznacza "pewny" odczyt.
        if _norm_name_norm(p.name_norm):
            s += 120
        return s

    for lst in lists:
        for p in lst:
            if not isinstance(p.rank, int):
                continue
            if p.rank <= 0 or p.rank > 200:
                continue
            cur = by_rank.get(p.rank)
            if cur is None or score(p) > score(cur):
                # normalizuj name_norm (czasem modele zwracają literal "null")
                p.name_norm = _norm_name_norm(p.name_norm)
                by_rank[p.rank] = p

    merged = list(by_rank.values())
    merged.sort(key=lambda x: x.rank)
    return merged


def _chat_needs_repair(players: Optional[List[PlayerScore]], expected_max_rank: Optional[int] = None) -> bool:
    """Decide whether we should re-run OCR in robust mode."""
    if not players:
        logger.debug("Chat repair: players empty")
        return True

    ranks = [p.rank for p in players if isinstance(p.rank, int)]
    if not ranks:
        logger.debug("Chat repair: no valid ranks")
        return True

    if len(set(ranks)) != len(ranks):
        logger.debug("Chat repair: duplicate ranks detected")
        return True

    mx = int(expected_max_rank) if expected_max_rank and expected_max_rank > 0 else max(ranks)
    if mx <= 0:
        logger.debug("Chat repair: mx<=0")
        return True

    # Hard validation: do we have a full 1..mx set?
    if set(ranks) != set(range(1, mx + 1)):
        missing = sorted(set(range(1, mx + 1)) - set(ranks))
        extra = sorted(set(ranks) - set(range(1, mx + 1)))
        logger.debug("Chat repair: rank set mismatch (mx=%d) missing=%s extra=%s", mx, missing, extra)
        return True

    # Extra: if many names look like OCR garbage, robust mode often fixes it.
    suspicious = sum(1 for p in players if _is_suspicious_name(p.name_raw))
    if suspicious >= max(2, mx // 10):
        logger.debug("Chat repair: too many suspicious names (%d out of mx=%d)", suspicious, mx)
        return True

    # Extra: duplicate names (by canonical key) are almost always a sign that
    # one line got misread and copied into another. Prefer a cheap repair first.
    canons: List[str] = []
    for p in players:
        try:
            canons.append(canonical_key((p.name_norm or p.name_raw or "").strip()))
        except Exception:
            continue
    canon_counts: Dict[str, int] = {}
    for ck in canons:
        if not ck:
            continue
        canon_counts[ck] = canon_counts.get(ck, 0) + 1
    dups = [ck for ck, n in canon_counts.items() if n > 1]
    if dups:
        logger.debug("Chat repair: duplicate names detected (canons=%s)", dups[:6])
        return True

    return False

# --- Cheap repair for suspicious names (cost optimization) ---

def _chat_repair_details(players: Optional[List[PlayerScore]], expected_max_rank: Optional[int] = None) -> dict:
    """Return details used to decide repair strategy."""
    details = {
        "empty": False,
        "duplicate_ranks": False,
        "duplicate_names": False,
        "duplicate_name_ranks": [],
        "mx": None,
        "missing_ranks": [],
        "extra_ranks": [],
        "suspicious_ranks": [],
        "suspicious_count": 0,
    }
    if not players:
        details["empty"] = True
        return details

    ranks = [p.rank for p in players if isinstance(p.rank, int) and p.rank > 0]
    if not ranks:
        details["empty"] = True
        return details

    if len(set(ranks)) != len(ranks):
        details["duplicate_ranks"] = True

    mx = int(expected_max_rank) if expected_max_rank and expected_max_rank > 0 else max(ranks)
    details["mx"] = mx

    full = set(range(1, mx + 1))
    rset = set(ranks)
    if rset != full:
        details["missing_ranks"] = sorted(full - rset)
        details["extra_ranks"] = sorted(rset - full)

    # Track suspicious ranks from name heuristics
    suspicious: List[int] = []
    for p in players:
        try:
            if _is_suspicious_name(p.name_raw):
                suspicious.append(int(p.rank))
        except Exception:
            continue

    # Track duplicate names (canonical key) and mark involved ranks as suspicious.
    canon_to_ranks: Dict[str, List[int]] = {}
    for p in players:
        try:
            r = int(p.rank)
            if r <= 0:
                continue
            ck = canonical_key((p.name_norm or p.name_raw or "").strip())
            if not ck:
                continue
            canon_to_ranks.setdefault(ck, []).append(r)
        except Exception:
            continue

    dup_ranks: List[int] = []
    for ck, rs in canon_to_ranks.items():
        if len(rs) > 1:
            dup_ranks.extend(rs)
    if dup_ranks:
        details["duplicate_names"] = True
        details["duplicate_name_ranks"] = sorted(set(dup_ranks))
        suspicious.extend(details["duplicate_name_ranks"])

    details["suspicious_ranks"] = sorted(set(suspicious))
    details["suspicious_count"] = len(details["suspicious_ranks"])
    return details


def _cluster_ranks(ranks: List[int], gap: int = 2) -> List[Tuple[int, int]]:
    if not ranks:
        return []
    r = sorted(set(int(x) for x in ranks if isinstance(x, int)))
    clusters: List[Tuple[int, int]] = []
    start = prev = r[0]
    for x in r[1:]:
        if x - prev <= gap:
            prev = x
        else:
            clusters.append((start, prev))
            start = prev = x
    clusters.append((start, prev))
    return clusters


def _roster_mismatch_ranks(
    players: Optional[List[PlayerScore]],
    roster: List[str],
    min_similarity: float = 80.0,
) -> List[int]:
    """Find ranks whose name_raw looks unlike anything in roster (and model didn't provide name_norm).

    This catches cases where the model read a completely different word (e.g. "Mądrasyn") instead of
    a stylized roster nick. We keep it conservative to avoid extra calls.
    """
    if not players or not roster:
        return []
    roster_canons = [canonical_key(r) for r in roster]
    out: List[int] = []

    for p in players:
        if not isinstance(p.rank, int) or p.rank <= 0:
            continue

        # If the model already confidently picked a roster nick, don't touch it.
        if p.name_norm and isinstance(p.name_norm, str) and p.name_norm.strip() and p.name_norm.strip().lower() not in {"null", "none"}:
            continue

        canon = canonical_key(p.name_raw or "")
        if len(canon) < 3:
            out.append(int(p.rank))
            continue

        best = 0.0
        for rc in roster_canons:
            if not rc:
                continue
            best = max(best, float(rf_fuzz.ratio(canon, rc)))

        if best < float(min_similarity):
            out.append(int(p.rank))

    return sorted(set(out))


def _repair_suspicious_rows(image_bytes: bytes, model: str, expected_max_rank: int, suspicious_ranks: List[int]) -> Optional[List[PlayerScore]]:
    """Attempt a cheaper repair by re-OCR'ing only regions that contain suspicious ranks.

    This is much cheaper than full slicing. If it doesn't help, we fall back to robust mode.
    """
    if not suspicious_ranks or not expected_max_rank or expected_max_rank <= 0:
        return None

    max_ranks = env_int("CHAT_ROW_REPAIR_MAX_RANKS", 12)
    uniq = sorted(set(int(r) for r in suspicious_ranks if isinstance(r, int) and r > 0))
    # If there are many suspicious ranks, downselect to keep cost under control while still covering the list.
    if len(uniq) > max_ranks:
        picks = set()
        # Cover local problem areas
        for (a, b) in _cluster_ranks(uniq, gap=2):
            picks.add(a); picks.add((a + b) // 2); picks.add(b)
        # Add a few "sentinel" ranks to catch silent shifts / missing rows in the middle.
        for rr in {max(1, expected_max_rank // 3), max(1, expected_max_rank // 2), max(1, (2 * expected_max_rank) // 3)}:
            picks.add(rr)
        uniq2 = sorted(r for r in picks if 1 <= r <= expected_max_rank)
        if len(uniq2) > max_ranks and max_ranks > 1:
            # Evenly sample to max_ranks (keep coverage).
            idxs = {round(i * (len(uniq2) - 1) / (max_ranks - 1)) for i in range(max_ranks)}
            uniq2 = [uniq2[i] for i in sorted(idxs)]
        logger.debug("Row-repair downselected suspicious ranks: %d -> %d (%s)", len(uniq), len(uniq2), uniq2)
        uniq = uniq2
    suspicious_ranks = uniq

    max_clusters = env_int("CHAT_ROW_REPAIR_MAX_CLUSTERS", 5)
    header_ratio = env_float("CHAT_ROW_REPAIR_HEADER_RATIO", 0.14)
    pad_lines = env_float("CHAT_ROW_REPAIR_PAD_LINES", 1.6)

    roster_path = env_str("ROSTER_PATH", "roster.json")
    roster = _load_roster(roster_path)

    pre = _preprocess_chat_image(image_bytes)
    w, h = pre.size
    header = int(h * header_ratio)
    header = max(0, min(header, h - 1))
    list_h = max(1, h - header)
    line_h = list_h / float(max(1, expected_max_rank))

    clusters = _cluster_ranks(suspicious_ranks, gap=2)[:max_clusters]
    logger.info("Cheap row-repair: suspicious_ranks=%s clusters=%s (max_rank=%d)", sorted(set(suspicious_ranks)), clusters, expected_max_rank)

    all_lists: List[List[PlayerScore]] = []
    for (a, b) in clusters:
        # Expand cluster by a few lines so OCR has context
        a2 = max(1, int(a - pad_lines))
        b2 = min(expected_max_rank, int(b + pad_lines))
        y1 = int(header + (a2 - 1) * line_h - line_h * 0.2)
        y2 = int(header + (b2) * line_h + line_h * 0.2)
        y1 = max(0, y1)
        y2 = min(h, y2)
        if y2 - y1 < int(line_h * 1.2):
            continue

        crop = pre.crop((0, y1, w, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        crop_bytes = buf.getvalue()
        logger.debug("Row-repair crop for ranks %d..%d -> roi=(0,%d,%d) bytes=%d", a2, b2, y1, y2, len(crop_bytes))
        try:
            lst = _parse_chat_slice(crop_bytes, model=model, roster=roster)
            # keep only ranks in the expanded window
            keep = [p for p in lst if isinstance(p.rank, int) and a2 <= p.rank <= b2]
            logger.debug("Row-repair parsed %d players (kept %d) for window %d..%d", len(lst), len(keep), a2, b2)
            all_lists.append(keep)
        except Exception as e:
            logger.debug("Row-repair crop parse failed for window %d..%d: %s", a2, b2, e)
            continue

    if not all_lists:
        return None

    merged = _merge_players(all_lists)
    logger.info("Cheap row-repair produced %d updated rows", len(merged))
    return merged



def _parse_chat_results_robust(image_bytes: bytes, model: str, parts: int) -> List[PlayerScore]:
    """Robust parsing: preprocessing + slicing with overlap + merge."""
    roster_path = env_str("ROSTER_PATH", "roster.json")
    roster = _load_roster(roster_path)

    logger.debug("Robust chat parse: parts=%d", int(parts))
    pre = _preprocess_chat_image(image_bytes)
    slices = _make_overlapping_slices(pre, parts=max(1, int(parts)), overlap=0.16)

    all_lists: List[List[PlayerScore]] = []
    for s in slices:
        try:
            all_lists.append(_parse_chat_slice(s, model=model, roster=roster))
        except Exception:
            # jeśli jeden slice się wywali, próbuj pozostałe
            continue

    merged = _merge_players(all_lists)
    logger.debug("Robust chat parse merged players=%d", len(merged))
    return merged


def _extract_war_mode_fallback(image_bytes: bytes, model: str) -> Optional[str]:
    """
    Fallback OCR:
    Tryb wojenny jest ZAWSZE pod ikonką trybu i nad przyciskiem 'POLE BITWY'
    (prawa strona panelu wojny). Robimy crop tej okolicy i prosimy model o odczyt
    WYŁĄCZNIE nazwy trybu.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    # ROI: prawa część panelu wojny, okolice ikonki + napis trybu (bez "POLE BITWY")
    # Proporcje dobrane tak, by złapać: ikonkę + nazwę trybu (np. HORDA NIEUMARŁYCH)
    x1 = int(w * 0.55)
    y1 = int(h * 0.45)
    x2 = int(w * 0.98)
    y2 = int(h * 0.57)

    crop = img.crop((x1, y1, x2, y2))

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    crop_bytes = buf.getvalue()

    data_url = _to_data_url(crop_bytes)
    timeout_s = env_int("OPENAI_TIMEOUT_SECONDS", 90)

    prompt = (
        "Odczytaj WYŁĄCZNIE nazwę trybu wojennego z tego fragmentu ekranu. "
        "Nazwa jest bezpośrednio POD okrągłą ikonką trybu i NAD przyciskiem 'POLE BITWY'. "
        "Zwróć samą nazwę (np. 'HORDA NIEUMARŁYCH', 'GRAD STRZAŁ', 'ŻAR Z NIEBA', 'STAROŻYTNY UPIÓR'). "
        "Nie zwracaj 'POLE BITWY' ani innych napisów. Jeśli nie widać – zwróć null."
    )

    logger.debug("War mode fallback OCR crop (w=%d,h=%d) roi=(%d,%d,%d,%d)", w, h, x1, y1, x2, y2)
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Jesteś OCR. Zwróć tylko pole war_mode."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=WarModeOnly,
        temperature=0,
        timeout=timeout_s,
    )

    out = resp.output_parsed
    if out and out.war_mode:
        wm = out.war_mode
        if isinstance(wm, str) and wm.strip().lower() in {"null", "none", "nil", "n/a"}:
            return None
        wm2 = str(wm).strip().upper()
        return wm2 if wm2 else None
    return None


# ----------------------------
# Main parsing
# ----------------------------

def parse_single_image(image_bytes: bytes, model: str) -> ParsedImage:
    logger.debug("parse_single_image: input bytes=%d model=%s", len(image_bytes), model)
    image_bytes = _ensure_png_bytes(image_bytes)
    roster_path = env_str("ROSTER_PATH", "roster.json")
    roster = _load_roster(roster_path)
    roster_text = "\n".join(f"- {n}" for n in roster) if roster else "(brak)"

    system_instructions = f"""Jesteś parserem screenów z gry Empires & Puzzles (Alliance War).
Zwracasz TYLKO dane zgodne ze schematem (Structured Output).

1) Rozpoznaj typ obrazka:
   - CHAT z listą „Najlepsi atakujący na wojnach:” (niebieski panel z rankingiem)
   - SOJUSZ/WOJNA z podsumowaniem (paski wyniku, tekst Zwycięstwo/Porażka, tryb)

2) Wyciągnij dane:
   A) Jeśli CHAT:
      - players: lista pozycji [1]..[X]
      - expected_max_rank: największy numer pozycji widoczny na screenie
      - rank (int), points (int)
      - name_raw: dokładnie jak widać (z ozdobnikami)
      - name_norm: jeżeli jesteś PEWNY, wybierz DOKŁADNIE jeden z nicków z ROSTER, inaczej null
      - NIE POMIJAJ żadnych wierszy (nicki mogą zawierać małe litery i cyfry, np. 'ropuch13')
      - Jeśli widzisz rank i punkty, ale nick jest trudny: przepisz możliwie najbliżej. Jeśli nadal niepewne, ustaw name_raw na '???' (ale ZAWSZE zwróć rekord dla tej rangi).
      - Upewnij się, że rangi są unikalne i mieszczą się w 1..expected_max_rank; nie duplikuj rang.

   B) Jeśli SOJUSZ/WOJNA:
      - our_alliance, opponent_alliance
      - result: Zwycięstwo albo Porażka
      - our_score, opponent_score (z pasków)
      - war_mode: NAZWA TRYBU WOJENNEGO
        UWAGA: war_mode jest ZAWSZE pod ikonką trybu i nad napisem/przyciskiem „POLE BITWY”.
        Jeśli widać, ZAWSZE uzupełnij war_mode.
      - beta_badge: true jeśli na panelu widać „BETA”, inaczej null/false.

ROSTER:
{roster_text}

Zasady:
- Ignoruj UI poza panelem z wynikami / paskami.
- points to liczba (int).
""".strip()

    data_url = _to_data_url(image_bytes)
    timeout_s = env_int("OPENAI_TIMEOUT_SECONDS", 90)

    logger.debug("OpenAI structured parse: sending image (%d bytes)", len(image_bytes))
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Rozpoznaj typ screena i wyciągnij dane."},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=ParsedImage,
        temperature=0,
        timeout=timeout_s,
    )

    out = resp.output_parsed
    if out:
        # sanitize optional fields (sometimes returned as literal strings)
        if out.kind == "chat_results" and out.chat_results and out.chat_results.players:
            for p in out.chat_results.players:
                if isinstance(p.name_norm, str) and p.name_norm.strip().lower() in {"null", "none", "nil", "n/a"}:
                    p.name_norm = None
        if out.kind == "war_summary" and out.war_summary:
            wm = out.war_summary.war_mode
            if isinstance(wm, str) and wm.strip().lower() in {"null", "none", "nil", "n/a"}:
                out.war_summary.war_mode = None
            elif isinstance(wm, str):
                out.war_summary.war_mode = wm.strip().upper()
        logger.debug("parse_single_image result: kind=%s conf=%.2f notes=%s", out.kind, out.confidence, (out.notes or ""))
        if out.kind == "chat_results" and out.chat_results:
            logger.debug("chat_results players=%d expected_max_rank=%s", len(out.chat_results.players or []), out.chat_results.expected_max_rank)
        if out.kind == "war_summary" and out.war_summary:
            ws = out.war_summary
            logger.debug("war_summary %s %d-%d mode=%s", ws.result, ws.our_score, ws.opponent_score, ws.war_mode)
    else:
        logger.debug("parse_single_image: OpenAI returned no parsed output")
    return out


def _merge_players_row_repair(
    base: List[PlayerScore],
    updates: List[PlayerScore],
    *,
    log_prefix: str = "Row-repair merge",
) -> List[PlayerScore]:
    """Safer merge used ONLY for targeted row-repair.

    Goals:
    - Preserve existing points when row-repair tries to change them.
    - Avoid overwriting a rank with an update that duplicates another rank (same name + points).
    - Prefer updates only when they materially improve the name (e.g. set name_norm).
    """

    def _name_key(p: PlayerScore) -> str:
        if p.name_norm:
            return p.name_norm.strip().lower()
        return canonical_key(p.name_raw or "")

    def _points(p: PlayerScore) -> Optional[int]:
        try:
            return int(p.points) if p.points is not None else None
        except Exception:
            return None

    by_rank: Dict[int, PlayerScore] = {p.rank: p for p in base if isinstance(p.rank, int)}

    sig_to_rank: Dict[Tuple[str, int], int] = {}
    for r, p in by_rank.items():
        nk = _name_key(p)
        pts = _points(p)
        if nk and pts is not None:
            sig_to_rank[(nk, pts)] = r

    for u in updates:
        if not isinstance(u.rank, int):
            continue
        r = u.rank
        cur = by_rank.get(r)

        # Preserve points from the base when possible.
        cur_pts = _points(cur) if cur else None
        upd_pts = _points(u)
        if cur_pts is not None:
            if upd_pts is None:
                u.points = cur_pts
            elif upd_pts != cur_pts:
                logger.debug(
                    "%s: preserving points for rank=%s (%s -> %s)",
                    log_prefix,
                    r,
                    upd_pts,
                    cur_pts,
                )
                u.points = cur_pts

        # Duplicate protection: do not allow (name,points) to appear on two ranks.
        nk = _name_key(u) or (_name_key(cur) if cur else "")
        pts = _points(u) if _points(u) is not None else cur_pts
        if nk and pts is not None:
            other = sig_to_rank.get((nk, pts))
            if other is not None and other != r:
                logger.debug(
                    "%s: rejecting update for rank=%s -> duplicates rank=%s (name=%r points=%s)",
                    log_prefix,
                    r,
                    other,
                    nk,
                    pts,
                )
                continue

        # Decide whether the update is actually better.
        def _is_better(cur_p: PlayerScore, upd_p: PlayerScore) -> bool:
            if (cur_p.name_raw or "").strip().upper() == "UNKNOWN":
                return True
            if upd_p.name_norm and not cur_p.name_norm:
                return True
            if cur_p.name_norm and not upd_p.name_norm:
                return False
            if cur_p.name_norm and upd_p.name_norm and cur_p.name_norm.strip().lower() != upd_p.name_norm.strip().lower():
                # Row-repair suggesting a different normalized name than we already have is risky.
                return False
            # Otherwise, prefer the one with a longer canonical key (more information).
            cur_len = len(canonical_key(cur_p.name_raw or ""))
            upd_len = len(canonical_key(upd_p.name_raw or ""))
            return upd_len >= (cur_len + 1)

        if cur is None:
            by_rank[r] = u
        else:
            if _is_better(cur, u):
                by_rank[r] = u
            else:
                continue

        # Update sig map with the accepted value.
        nk2 = _name_key(by_rank[r])
        pts2 = _points(by_rank[r])
        if nk2 and pts2 is not None:
            sig_to_rank[(nk2, pts2)] = r

    out = [by_rank[r] for r in sorted(by_rank.keys())]
    return out


def parse_war_from_images(
    images: List[bytes],
    model: str,
    trace_id: Optional[str] = None,
) -> Tuple[Optional[WarSummary], Optional[List[PlayerScore]], Optional[int], List[ParsedImage]]:
    """
    images: lista obrazków (typowo 2 szt.: chat + war summary)
    Zwraca: summary, players, parsed_debug
    """
    token = None
    if trace_id:
        token = set_trace_id(trace_id)

    logger.info("START parse_war_from_images: images=%d model=%s", len(images), model)

    parsed_debug: List[ParsedImage] = []
    parsed_with_bytes: List[tuple[bytes, ParsedImage]] = []

    # Ensure consistent PNG bytes for all inputs.
    images_png = [_ensure_png_bytes(b) for b in images]
    logger.debug("Normalized images to PNG: sizes=%s", [len(b) for b in images_png])

    for idx, b in enumerate(images_png):
        logger.debug("Parsing image #%d/%d", idx + 1, len(images_png))
        p = parse_single_image(b, model=model)
        parsed_debug.append(p)
        parsed_with_bytes.append((b, p))

    summary: Optional[WarSummary] = None
    players: Optional[List[PlayerScore]] = None
    expected_max_rank: Optional[int] = None
    summary_image_bytes: Optional[bytes] = None
    chat_image_bytes: Optional[bytes] = None

    for b, p in parsed_with_bytes:
        if p.kind == "war_summary" and p.war_summary:
            summary = p.war_summary
            summary_image_bytes = b
        if p.kind == "chat_results" and p.chat_results:
            players = p.chat_results.players
            expected_max_rank = p.chat_results.expected_max_rank
            chat_image_bytes = b

    logger.debug("Selected summary=%s chat=%s", bool(summary_image_bytes), bool(chat_image_bytes))
    logger.debug("Initial expected_max_rank=%s players=%s", expected_max_rank, (len(players) if players else None))

    if chat_image_bytes and (not expected_max_rank or expected_max_rank <= 0):
        expected_max_rank = _detect_expected_max_rank(chat_image_bytes, model=model)
    if players and (not expected_max_rank or expected_max_rank <= 0):
        ranks = [p.rank for p in players if isinstance(p.rank, int) and p.rank > 0]
        expected_max_rank = max(ranks) if ranks else None

    logger.debug("After expected_max_rank inference: %s", expected_max_rank)

    # Jeśli chat wyszedł podejrzany (braki/duplikaty/śmieciowe nicki), spróbuj naprawy.
    if chat_image_bytes and _chat_needs_repair(players, expected_max_rank):
        details = _chat_repair_details(players, expected_max_rank)

        # Optymalizacja kosztów: jeżeli ranki są kompletne, a problem to głównie "podejrzane" nicki,
        # spróbuj najpierw taniej naprawy (row-crops) zamiast pełnego slicing-u.
        if env_bool("CHAT_ROW_REPAIR", True):
            only_suspicious = (
                (not details.get("empty"))
                and (not details.get("duplicate_ranks"))
                and (not details.get("missing_ranks"))
                and (not details.get("extra_ranks"))
                and bool(details.get("suspicious_ranks"))
                and bool(details.get("mx"))
            )
            if only_suspicious and expected_max_rank:
                upd = _repair_suspicious_rows(
                    chat_image_bytes,
                    model=model,
                    expected_max_rank=int(expected_max_rank),
                    suspicious_ranks=details["suspicious_ranks"],
                )
                if upd:
                    players = _merge_players_row_repair(players or [], upd, log_prefix="Row-repair(cheap)")
                    if not _chat_needs_repair(players, expected_max_rank):
                        logger.info("Cheap row-repair fixed chat -> skipping robust slicing")

        # Jeśli nadal podejrzane (braki/duplikaty/śmieciowe nicki), przechodzimy do robust OCR (slicing).
        if _chat_needs_repair(players, expected_max_rank):
            primary_parts = env_int("OPENAI_CHAT_SLICES", 4)
            fallback_parts = env_int("OPENAI_CHAT_SLICES_FALLBACK", 6)

            logger.info("Chat flagged as suspicious -> robust mode (parts=%d, fallback=%d)", primary_parts, fallback_parts)

            cand = _parse_chat_results_robust(chat_image_bytes, model=model, parts=primary_parts)
            logger.debug("Robust(primary) produced %d players", len(cand or []))
            if cand and not _chat_needs_repair(cand, expected_max_rank):
                players = cand
                logger.info("Robust(primary) accepted")
            elif cand:
                # Merge with the original as a last resort (choose better lines per rank).
                players = _merge_players([players or [], cand])
                logger.info("Robust(primary) merged with initial -> %d players", len(players or []))

            if _chat_needs_repair(players, expected_max_rank) and fallback_parts != primary_parts:
                logger.info("Still suspicious -> robust fallback (parts=%d)", fallback_parts)
                cand2 = _parse_chat_results_robust(chat_image_bytes, model=model, parts=fallback_parts)
                logger.debug("Robust(fallback) produced %d players", len(cand2 or []))
                if cand2 and not _chat_needs_repair(cand2, expected_max_rank):
                    players = cand2
                    logger.info("Robust(fallback) accepted")
                elif cand2:
                    players = _merge_players([players or [], cand2])
                    logger.info("Robust(fallback) merged -> %d players", len(players or []))


    # Jeśli w ogóle nie wykryliśmy chatu (zła klasyfikacja), spróbuj robust na wszystkich obrazkach
    # i wybierz ten z największą liczbą pozycji.
    if players is None:
        logger.info("Chat not detected in any image -> trying robust parse on all images")
        best: List[PlayerScore] = []
        primary_parts = env_int("OPENAI_CHAT_SLICES", 4)
        for b in images_png:
            try:
                cand = _parse_chat_results_robust(b, model=model, parts=primary_parts)
                if len(cand) > len(best):
                    best = cand
            except Exception:
                continue
        if best:
            players = best
            ranks = [p.rank for p in players if isinstance(p.rank, int) and p.rank > 0]
            expected_max_rank = max(ranks) if ranks else expected_max_rank
            logger.info("Robust-all-images selected %d players (expected_max_rank=%s)", len(players), expected_max_rank)

    # Post-pass: if the structure looks OK but some rows still look unlike the roster,
    # try a cheap row-level re-OCR for only those ranks.
    if chat_image_bytes and players and expected_max_rank and env_bool("CHAT_ROW_REPAIR", True):
        details2 = _chat_repair_details(players, expected_max_rank)
        structure_ok = (
            (not details2.get("empty"))
            and (not details2.get("duplicate_ranks"))
            and (not details2.get("missing_ranks"))
            and (not details2.get("extra_ranks"))
            and bool(details2.get("mx"))
        )
        if structure_ok:
            roster_path = env_str("ROSTER_PATH", "roster.json")
            roster = _load_roster(roster_path)
            min_sim = env_float("CHAT_ROW_REPAIR_ROSTER_MIN_SIM", 80.0)
            mismatch = _roster_mismatch_ranks(players, roster, min_similarity=min_sim)
            if mismatch:
                logger.info("Roster-mismatch ranks detected -> row-repair: %s", mismatch)
                upd2 = _repair_suspicious_rows(
                    chat_image_bytes,
                    model=model,
                    expected_max_rank=int(expected_max_rank),
                    suspicious_ranks=mismatch,
                )
                if upd2:
                    players = _merge_players_row_repair(players or [], upd2, log_prefix="Row-repair(roster-mismatch)")
                    logger.info("Roster-mismatch row-repair merged -> %d players", len(players or []))

    # Fallback na war_mode: jeśli model pominął tryb na pełnym screenie, doczytaj z cropa
    if summary and (not summary.war_mode or not summary.war_mode.strip()) and summary_image_bytes:
        logger.info("war_mode missing -> fallback OCR")
        wm = _extract_war_mode_fallback(summary_image_bytes, model=model)
        if wm:
            summary.war_mode = wm
            logger.info("war_mode fallback success: %s", wm)

    # Final normalization: keep war_mode stable for downstream filtering/UI.
    if summary and summary.war_mode and isinstance(summary.war_mode, str):
        wm3 = summary.war_mode.strip().upper()
        summary.war_mode = wm3 if wm3 else None

    logger.info("END parse_war_from_images: summary=%s players=%s expected_max_rank=%s", bool(summary), (len(players) if players else None), expected_max_rank)

    if token is not None:
        reset_trace_id(token)

    return summary, players, expected_max_rank, parsed_debug
