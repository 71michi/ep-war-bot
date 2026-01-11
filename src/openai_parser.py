import base64
import io
import json
from typing import List, Optional, Literal, Tuple, Dict, Any

from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image
from PIL import ImageEnhance, ImageFilter

from .config import env_str, env_int

client = OpenAI()


# ----------------------------
# Models (Structured Outputs)
# ----------------------------

class WarModeOnly(BaseModel):
    war_mode: Optional[str] = None


class VisibleMaxRank(BaseModel):
    """Największy widoczny numer rankingu na screenie chatu."""
    max_rank: Optional[int] = None


class PlayerScore(BaseModel):
    rank: int
    name_raw: str
    points: int
    # (opcjonalne) jeśli model jest 100% pewny, może wskazać roster. W praktyce
    # NIE polegamy na tym polu do normalizacji nicków.
    name_norm: Optional[str] = None


class ChatResults(BaseModel):
    title: str = Field(default="Najlepsi atakujący na wojnach")
    players: List[PlayerScore]


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


# ----------------------------
# Helpers
# ----------------------------

def _ensure_png_bytes(image_bytes: bytes) -> bytes:
    """Zawsze konwertuj wejście do PNG (stabilniejsze dla vision/OCR)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


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
    y1 = int(h * 0.23)
    y2 = int(h * 0.86)
    x1 = int(w * 0.02)
    x2 = int(w * 0.98)
    roi = img.crop((x1, y1, x2, y2))

    # Powiększ + kontrast + wyostrzenie
    roi = roi.resize((roi.width * 2, roi.height * 2), Image.LANCZOS)
    roi = ImageEnhance.Contrast(roi).enhance(1.35)
    roi = roi.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    return roi


def _make_overlapping_slices(img: Image.Image, parts: int = 4, overlap: float = 0.16) -> List[bytes]:
    """Dzieli obraz pionowo na części z nakładką (overlap) i zwraca PNG bytes."""
    w, h = img.size
    parts = max(1, int(parts))
    overlap = max(0.0, min(float(overlap), 0.35))

    base = h / parts
    ov = int(base * overlap)

    out: List[bytes] = []
    for i in range(parts):
        y1 = int(i * base) - (ov if i > 0 else 0)
        y2 = int((i + 1) * base) + (ov if i < parts - 1 else 0)
        y1 = max(0, y1)
        y2 = min(h, y2)
        crop = img.crop((0, y1, w, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG", optimize=True)
        out.append(buf.getvalue())
    return out


def _parse_chat_slice(slice_png_bytes: bytes, model: str) -> List[PlayerScore]:
    system_instructions = """Jesteś OCR-em listy rankingowej z gry Empires & Puzzles.
To jest WYCIĘTY FRAGMENT listy „Najlepsi atakujący w wojnach”.

Masz odczytać WSZYSTKIE widoczne wiersze rankingu.

Każdy wiersz ma format zbliżony do:
  [23] [g3w] ropuch13 192

Zasady krytyczne:
- NIE POMIJAJ żadnego wiersza.
- Nick może zawierać małe litery i cyfry (np. ropuch13) oraz ozdobniki. Przepisz je do name_raw.
- points to ostatnia liczba w wierszu.
- rank to liczba w nawiasie kwadratowym [n].
- name_norm ZAWSZE ustaw na null (normalizacją zajmie się program).
""".strip()

    data_url = _to_data_url(slice_png_bytes)
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
    players = out.players if out and out.players else []
    # Wymuś name_norm = None (żeby nie wpływało na merge).
    for p in players:
        p.name_norm = None
    return players


def _merge_players(lists: List[List[PlayerScore]]) -> List[PlayerScore]:
    """Scala wyniki z kilku slice'ów po rank i wybiera "lepszy" rekord."""
    by_rank: Dict[int, PlayerScore] = {}

    def score(p: PlayerScore) -> int:
        # Heurystyka: preferuj dłuższe name_raw (mniej obcięte) i poprawne points.
        s = 0
        if p.name_raw:
            s += min(len(p.name_raw), 40)
        if isinstance(p.points, int) and p.points > 0:
            s += 20
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


def _validate_chat(players: Optional[List[PlayerScore]], expected_max_rank: Optional[int]) -> List[str]:
    """Zwraca listę błędów walidacji (pusta lista = OK)."""
    errors: List[str] = []
    if not players:
        return ["no_players"]

    ranks = [p.rank for p in players if isinstance(p.rank, int) and p.rank > 0]
    if not ranks:
        return ["no_ranks"]

    if len(set(ranks)) != len(ranks):
        errors.append("duplicate_ranks")

    # Oczekiwany max rank (wyczytany z dołu screena)
    if expected_max_rank and expected_max_rank > 0:
        exp = int(expected_max_rank)
        exp_set = set(range(1, exp + 1))
        got_set = set(ranks)
        if got_set != exp_set:
            missing = sorted(exp_set - got_set)
            extra = sorted(got_set - exp_set)
            if missing:
                errors.append(f"missing_ranks:{missing[:30]}")
            if extra:
                errors.append(f"extra_ranks:{extra[:30]}")
    else:
        # Fallback: przynajmniej brak "dziur" do max rank
        mx = max(ranks) if ranks else 0
        if mx >= 10:
            exp_set = set(range(1, mx + 1))
            got_set = set(ranks)
            if got_set != exp_set:
                missing = sorted(exp_set - got_set)
                if missing:
                    errors.append(f"missing_ranks:{missing[:30]}")

    return errors


def _extract_visible_max_rank(chat_image_bytes: bytes, model: str) -> Optional[int]:
    """Czyta z dolnej części chatu największy widoczny numer rankingu."""
    try:
        pre = _preprocess_chat_image(chat_image_bytes)
    except Exception:
        return None

    w, h = pre.size
    # Dolna część (ostatnie ~30% listy), lewa strona (tam są nawiasy z numerami)
    x1 = 0
    x2 = int(w * 0.40)
    y1 = int(h * 0.70)
    y2 = h
    crop = pre.crop((x1, y1, x2, y2))

    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    data_url = _to_data_url(buf.getvalue())
    timeout_s = env_int("OPENAI_TIMEOUT_SECONDS", 90)

    prompt = (
        "Na obrazie widać końcówkę listy rankingu (linie z [n]). "
        "Zwróć WYŁĄCZNIE największy numer n widoczny w nawiasach kwadratowych. "
        "Jeśli nie widzisz żadnego [n], zwróć null."
    )

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Jesteś OCR. Zwróć tylko pole max_rank."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=VisibleMaxRank,
        temperature=0,
        timeout=timeout_s,
    )

    out = resp.output_parsed
    if out and isinstance(out.max_rank, int) and out.max_rank > 0:
        return int(out.max_rank)
    return None


def _parse_chat_results_robust(chat_image_bytes: bytes, model: str, parts: int) -> List[PlayerScore]:
    """Robust parsing: preprocessing + cięcie na overlappujące fragmenty + merge."""
    pre = _preprocess_chat_image(chat_image_bytes)
    slices = _make_overlapping_slices(pre, parts=parts, overlap=0.16)
    all_lists: List[List[PlayerScore]] = []
    for s in slices:
        try:
            all_lists.append(_parse_chat_slice(s, model=model))
        except Exception:
            continue
    return _merge_players(all_lists)


def _extract_war_mode_fallback(image_bytes: bytes, model: str) -> Optional[str]:
    """Fallback OCR dla war_mode z ROI po prawej stronie panelu wojny."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    x1 = int(w * 0.55)
    y1 = int(h * 0.45)
    x2 = int(w * 0.98)
    y2 = int(h * 0.57)

    crop = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    data_url = _to_data_url(buf.getvalue())
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
    # MUST-HAVE: zawsze PNG
    png = _ensure_png_bytes(image_bytes)

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
      - rank (int), points (int)
      - name_raw: dokładnie jak widać (z ozdobnikami)
      - NIE POMIJAJ żadnych wierszy (nicki mogą zawierać małe litery i cyfry, np. 'ropuch13')
      - name_norm: ustaw null (normalizacja jest po stronie programu)

   B) Jeśli SOJUSZ/WOJNA:
      - our_alliance, opponent_alliance
      - result: Zwycięstwo albo Porażka
      - our_score, opponent_score (z pasków)
      - war_mode: NAZWA TRYBU WOJENNEGO
        UWAGA: war_mode jest ZAWSZE pod ikonką trybu i nad napisem/przyciskiem „POLE BITWY”.
        Jeśli widać, ZAWSZE uzupełnij war_mode.
      - beta_badge: true jeśli na panelu widać „BETA”, inaczej null/false.

ROSTER (tylko do kontekstu nazwy sojuszy, NIE mapuj nicków):
{roster_text}

Zasady:
- Ignoruj UI poza panelem z wynikami / paskami.
- points to liczba (int).
""".strip()

    data_url = _to_data_url(png)
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

    out = resp.output_parsed
    # Nie ufamy name_norm z modelu.
    try:
        if out and out.chat_results and out.chat_results.players:
            for p in out.chat_results.players:
                p.name_norm = None
    except Exception:
        pass
    return out


def parse_war_from_images(
    images: List[bytes],
    model: str
) -> Tuple[Optional[WarSummary], Optional[List[PlayerScore]], List[ParsedImage], Dict[str, Any]]:
    """Zwraca: summary, players, parsed_debug, meta."""
    # Konwertuj WSZYSTKO do PNG na wejściu (stabilność)
    images_png = [_ensure_png_bytes(b) for b in images]

    parsed_debug: List[ParsedImage] = []
    parsed_with_bytes: List[tuple[bytes, ParsedImage]] = []

    for b in images_png:
        p = parse_single_image(b, model=model)
        parsed_debug.append(p)
        parsed_with_bytes.append((b, p))

    summary: Optional[WarSummary] = None
    players: Optional[List[PlayerScore]] = None
    summary_image_bytes: Optional[bytes] = None
    chat_image_bytes: Optional[bytes] = None

    for b, p in parsed_with_bytes:
        if p and p.kind == "war_summary" and p.war_summary:
            summary = p.war_summary
            summary_image_bytes = b
        if p and p.kind == "chat_results" and p.chat_results:
            players = p.chat_results.players
            chat_image_bytes = b

    meta: Dict[str, Any] = {
        "chat_expected_max_rank": None,
        "chat_validation_errors": [],
        "chat_repaired": False,
        "chat_repair_strategy": None,
    }

    def _pick_best(candidates: List[List[PlayerScore]], expected: Optional[int]) -> tuple[List[PlayerScore], List[str]]:
        best: List[PlayerScore] = []
        best_err: List[str] = ["no_players"]

        for cand in candidates:
            err = _validate_chat(cand, expected)
            # 1) mniej błędów = lepiej
            # 2) więcej unikalnych ranków = lepiej
            # 3) wyższy max rank = lepiej
            def key(pl: List[PlayerScore], er: List[str]) -> tuple[int, int, int]:
                ranks = [p.rank for p in pl if isinstance(p.rank, int) and p.rank > 0]
                uniq = len(set(ranks))
                mx = max(ranks) if ranks else 0
                return (-len(er), uniq, mx)

            if key(cand, err) > key(best, best_err):
                best = cand
                best_err = err
        return best, best_err

    # --- CHAT: walidacja + auto-naprawa (2-fazowo) ---
    if chat_image_bytes:
        expected = _extract_visible_max_rank(chat_image_bytes, model=model)
        meta["chat_expected_max_rank"] = expected

        base_err = _validate_chat(players, expected)
        if base_err:
            # 1) robust z domyślną liczbą slice'ów (ekonomicznie)
            parts1 = env_int("OPENAI_CHAT_SLICES", 4)
            robust1 = _parse_chat_results_robust(chat_image_bytes, model=model, parts=parts1)

            # Czasem połączenie (normal + robust) daje najlepszy efekt
            merged1 = _merge_players([players or [], robust1])

            best1, best1_err = _pick_best([players or [], robust1, merged1], expected)

            # 2) jeśli nadal fail -> fallback na 6 slice'ów
            if best1_err:
                parts2 = max(6, parts1)
                robust2 = _parse_chat_results_robust(chat_image_bytes, model=model, parts=parts2)
                merged2 = _merge_players([best1, robust2])
                best2, best2_err = _pick_best([best1, robust2, merged2], expected)
                players = best2
                meta["chat_repaired"] = True
                meta["chat_repair_strategy"] = f"slices:{parts2}" if best2_err != base_err else f"attempted_slices:{parts2}"
                meta["chat_validation_errors"] = best2_err
            else:
                players = best1
                meta["chat_repaired"] = True
                meta["chat_repair_strategy"] = f"slices:{parts1}"
                meta["chat_validation_errors"] = best1_err
        else:
            meta["chat_validation_errors"] = []

    # Jeśli nie wykryliśmy chatu (zła klasyfikacja), spróbuj robust na wszystkich obrazkach i wybierz najlepszy.
    if players is None:
        best: List[PlayerScore] = []
        best_err: List[str] = ["no_players"]
        best_expected: Optional[int] = None
        for b in images_png:
            try:
                expected = _extract_visible_max_rank(b, model=model)
                parts = env_int("OPENAI_CHAT_SLICES", 4)
                cand = _parse_chat_results_robust(b, model=model, parts=parts)
                err = _validate_chat(cand, expected)
                ranks = [p.rank for p in cand if isinstance(p.rank, int) and p.rank > 0]
                if (-len(err), len(set(ranks))) > (-len(best_err), len(set([p.rank for p in best if isinstance(p.rank, int) and p.rank > 0]))):
                    best = cand
                    best_err = err
                    best_expected = expected
            except Exception:
                continue
        if best:
            players = best
            meta["chat_expected_max_rank"] = best_expected
            meta["chat_validation_errors"] = best_err
            meta["chat_repaired"] = True
            meta["chat_repair_strategy"] = "fallback_scan"

    # Fallback na war_mode: jeśli model pominął tryb na pełnym screenie, doczytaj z cropa
    if summary and (not summary.war_mode or not summary.war_mode.strip()) and summary_image_bytes:
        wm = _extract_war_mode_fallback(summary_image_bytes, model=model)
        if wm:
            summary.war_mode = wm

    return summary, players, parsed_debug, meta
