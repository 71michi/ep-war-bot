import base64
import io
import json
import re
from typing import List, Optional, Literal, Tuple, Dict

from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image
from PIL import ImageEnhance, ImageFilter

from .config import env_str, env_int
from .nicknames import canonical_key

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
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # If PIL can't decode it, fall back to original bytes.
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

    roi = img.crop((x1, y1, x2, y2))

    # Powiększ, zwiększ kontrast i delikatnie wyostrz.
    roi = roi.resize((roi.width * 2, roi.height * 2), Image.LANCZOS)
    roi = ImageEnhance.Contrast(roi).enhance(1.35)
    roi = roi.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
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
        data_url = _to_data_url(buf.getvalue())
    except Exception:
        return None

    prompt = (
        "Na tym fragmencie jest dół listy rankingowej z pozycjami w formacie [n]. "
        "Podaj TYLKO największy widoczny numer pozycji jako expected_max_rank. "
        "Jeśli nie widać żadnych [n], zwróć null."
    )

    timeout_s = env_int("OPENAI_TIMEOUT_SECONDS", 90)
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
        return int(out.expected_max_rank)
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
- Nick może zawierać małe litery i cyfry (np. ropuch13) oraz ozdobniki. Przepisz je do name_raw.
- points to ostatnia liczba w wierszu.
- rank to liczba w nawiasie kwadratowym [n].
- name_norm: jeżeli jesteś PEWNY, wybierz DOKŁADNIE jeden z nicków z ROSTER, inaczej null.

ROSTER:
{roster_text}
""".strip()

    data_url = _to_data_url(slice_bytes)
    timeout_s = env_int("OPENAI_TIMEOUT_SECONDS", 90)

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
    return out.players if out and out.players else []


def _merge_players(lists: List[List[PlayerScore]]) -> List[PlayerScore]:
    """Scala wyniki z kilku slice'ów po rank i wybiera "lepszy" rekord."""
    by_rank: Dict[int, PlayerScore] = {}

    def score(p: PlayerScore) -> int:
        # Heurystyka: preferuj dłuższe name_raw (mniej obcięte) i niezerowe points
        s = 0
        if p.name_raw:
            s += min(len(p.name_raw), 40)
        if p.points and p.points > 0:
            s += 20
        if p.name_norm:
            s += 10
        return s

    for lst in lists:
        for p in lst:
            if not isinstance(p.rank, int):
                continue
            if p.rank <= 0 or p.rank > 200:
                continue
            cur = by_rank.get(p.rank)
            if cur is None or score(p) > score(cur):
                by_rank[p.rank] = p

    merged = list(by_rank.values())
    merged.sort(key=lambda x: x.rank)
    return merged


def _chat_needs_repair(players: Optional[List[PlayerScore]], expected_max_rank: Optional[int] = None) -> bool:
    """Decide whether we should re-run OCR in robust mode."""
    if not players:
        return True

    ranks = [p.rank for p in players if isinstance(p.rank, int)]
    if not ranks:
        return True

    if len(set(ranks)) != len(ranks):
        return True

    mx = int(expected_max_rank) if expected_max_rank and expected_max_rank > 0 else max(ranks)
    if mx <= 0:
        return True

    # Hard validation: do we have a full 1..mx set?
    if set(ranks) != set(range(1, mx + 1)):
        return True

    # Extra: if many names look like OCR garbage, robust mode often fixes it.
    suspicious = sum(1 for p in players if _is_suspicious_name(p.name_raw))
    if suspicious >= max(2, mx // 10):
        return True

    return False


def _parse_chat_results_robust(image_bytes: bytes, model: str, parts: int) -> List[PlayerScore]:
    """Robust parsing: preprocessing + slicing with overlap + merge."""
    roster_path = env_str("ROSTER_PATH", "roster.json")
    roster = _load_roster(roster_path)

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
        wm = out.war_mode.strip()
        return wm if wm else None
    return None


# ----------------------------
# Main parsing
# ----------------------------

def parse_single_image(image_bytes: bytes, model: str) -> ParsedImage:
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

    return resp.output_parsed


def parse_war_from_images(
    images: List[bytes],
    model: str
) -> Tuple[Optional[WarSummary], Optional[List[PlayerScore]], Optional[int], List[ParsedImage]]:
    """
    images: lista obrazków (typowo 2 szt.: chat + war summary)
    Zwraca: summary, players, parsed_debug
    """
    parsed_debug: List[ParsedImage] = []
    parsed_with_bytes: List[tuple[bytes, ParsedImage]] = []

    # Ensure consistent PNG bytes for all inputs.
    images_png = [_ensure_png_bytes(b) for b in images]

    for b in images_png:
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

    if chat_image_bytes and (not expected_max_rank or expected_max_rank <= 0):
        expected_max_rank = _detect_expected_max_rank(chat_image_bytes, model=model)
    if players and (not expected_max_rank or expected_max_rank <= 0):
        ranks = [p.rank for p in players if isinstance(p.rank, int) and p.rank > 0]
        expected_max_rank = max(ranks) if ranks else None

    # Jeśli chat wyszedł podejrzany (braki/duplikaty/śmieciowe nicki), spróbuj robust OCR.
    if chat_image_bytes and _chat_needs_repair(players, expected_max_rank):
        primary_parts = env_int("OPENAI_CHAT_SLICES", 4)
        fallback_parts = env_int("OPENAI_CHAT_SLICES_FALLBACK", 6)

        cand = _parse_chat_results_robust(chat_image_bytes, model=model, parts=primary_parts)
        if cand and not _chat_needs_repair(cand, expected_max_rank):
            players = cand
        elif cand:
            # Merge with the original as a last resort (choose better lines per rank).
            players = _merge_players([players or [], cand])

        if _chat_needs_repair(players, expected_max_rank) and fallback_parts != primary_parts:
            cand2 = _parse_chat_results_robust(chat_image_bytes, model=model, parts=fallback_parts)
            if cand2 and not _chat_needs_repair(cand2, expected_max_rank):
                players = cand2
            elif cand2:
                players = _merge_players([players or [], cand2])

    # Jeśli w ogóle nie wykryliśmy chatu (zła klasyfikacja), spróbuj robust na wszystkich obrazkach
    # i wybierz ten z największą liczbą pozycji.
    if players is None:
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

    # Fallback na war_mode: jeśli model pominął tryb na pełnym screenie, doczytaj z cropa
    if summary and (not summary.war_mode or not summary.war_mode.strip()) and summary_image_bytes:
        wm = _extract_war_mode_fallback(summary_image_bytes, model=model)
        if wm:
            summary.war_mode = wm

    return summary, players, expected_max_rank, parsed_debug
