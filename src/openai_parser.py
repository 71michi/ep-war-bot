import base64
import io
import json
import re
from typing import Dict, List, Optional, Literal, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image

from .config import env_str, env_int

client = OpenAI()

# -----------------------------
# Pydantic schemas (Structured Output)
# -----------------------------

class WarModeOnly(BaseModel):
    war_mode: Optional[str] = None

class OcrText(BaseModel):
    text: Optional[str] = None

class PlayerScore(BaseModel):
    rank: int
    name_raw: str
    points: int
    name_norm: Optional[str] = None  # chosen from roster when confident

class ChatResults(BaseModel):
    title: str = Field(default="Najlepsi atakujący na wojnach")
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


# -----------------------------
# Helpers
# -----------------------------

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

def _img_bytes_from_pil(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _ocr_text(image_bytes: bytes, model: str, prompt: str, timeout_s: int) -> str:
    """
    OCR (vision) → zwraca surowy tekst (w ramach możliwości), bez interpretacji.
    """
    data_url = _to_data_url(image_bytes)
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Jesteś OCR. Zwróć wyłącznie pole JSON: {\"text\": \"...\"}."},
            {"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ]},
        ],
        text_format=OcrText,
        temperature=0,
        timeout=timeout_s,
    )
    out = resp.output_parsed
    return (out.text or "").strip() if out else ""

# Regex: [12] (opcjonalne [tag]) nazwa ... punkty
_CHAT_LINE_RE = re.compile(
    r"^\s*\[(\d{1,2})\]\s*(.+?)\s+(?:—\s*)?(\d{1,4})\s*$"
)

def _parse_chat_lines_to_rank_map(text: str) -> Dict[int, Tuple[str, int]]:
    """
    Z surowego OCR tekstu wyciąga linie rankingu.
    Zwraca dict: rank -> (name_raw, points)
    """
    rank_map: Dict[int, Tuple[str, int]] = {}
    if not text:
        return rank_map

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # pomijamy nagłówki itp.
        if "Najlepsi atakujący" in line:
            continue

        m = _CHAT_LINE_RE.match(line)
        if not m:
            continue

        r = int(m.group(1))
        name = m.group(2).strip()
        pts = int(m.group(3))

        # wybieramy "lepszy" wpis jeśli dubel:
        # preferuj dłuższe name (często pełniejsze), a przy remisie nic nie zmieniaj
        if r in rank_map:
            old_name, old_pts = rank_map[r]
            if (len(name) > len(old_name)) or (old_pts != pts and pts > old_pts):
                rank_map[r] = (name, pts)
        else:
            rank_map[r] = (name, pts)

    return rank_map

def _merge_rank_maps(maps: List[Dict[int, Tuple[str, int]]]) -> Dict[int, Tuple[str, int]]:
    merged: Dict[int, Tuple[str, int]] = {}
    for mp in maps:
        for r, (name, pts) in mp.items():
            if r not in merged:
                merged[r] = (name, pts)
            else:
                old_name, old_pts = merged[r]
                if (len(name) > len(old_name)) or (old_pts != pts and pts > old_pts):
                    merged[r] = (name, pts)
    return merged

def _guess_expected_size(roster_size: int) -> int:
    """
    Ile osób powinno być w rankingu.
    Priorytet: ENV EXPECTED_PLAYERS, potem roster_size (jeśli sensowny), w ostateczności 30.
    """
    env_expected = env_int("EXPECTED_PLAYERS", 0)
    if env_expected and env_expected > 0:
        return env_expected
    if 10 <= roster_size <= 30:
        return roster_size
    return 30

def _chat_panel_box(w: int, h: int) -> Tuple[int, int, int, int]:
    """
    Przybliżony bounding box na niebieski panel z listą rankingową w screenie CHAT.
    Działa na Twoich screenach (telefon pionowo).
    """
    x1 = int(w * 0.03)
    x2 = int(w * 0.97)
    y1 = int(h * 0.22)
    y2 = int(h * 0.92)
    return x1, y1, x2, y2

def _hard_recover_single_rank(
    img: Image.Image,
    panel_box: Tuple[int, int, int, int],
    rank: int,
    max_rank: int,
    model: str,
    timeout_s: int,
) -> Optional[Tuple[str, int]]:
    """
    HARD recovery: duży crop (3-6 linii) na oko w miejscu ranku i OCR.
    Zwraca (name, points) dla konkretnego ranku albo None.
    """
    x1, y1, x2, y2 = panel_box
    panel_h = (y2 - y1)
    if max_rank <= 0:
        return None

    # przybliżona wysokość jednej linii
    line_h = max(10, int(panel_h / max_rank))
    # środek linijki ranku
    y_center = int(y1 + (rank - 0.5) * line_h)

    # najpierw okno ~4 linie
    for lines in (4, 6):
        half = int((lines * line_h) / 2)
        cy1 = max(0, y_center - half)
        cy2 = min(img.size[1], y_center + half)

        crop = img.crop((x1, cy1, x2, cy2))
        crop_bytes = _img_bytes_from_pil(crop)

        prompt = (
            "Przepisz dokładnie linie rankingu widoczne na tym fragmencie. "
            "BARDZO WAŻNE: zachowaj numery w nawiasach kwadratowych [NN] jeśli są widoczne. "
            "Zwróć tekst 1:1, po jednej linii na wpis."
        )
        txt = _ocr_text(crop_bytes, model=model, prompt=prompt, timeout_s=timeout_s)
        mp = _parse_chat_lines_to_rank_map(txt)
        if rank in mp:
            return mp[rank]

    return None

def _hard_parse_chat_results(image_bytes: bytes, model: str, roster_size: int) -> Optional[List[PlayerScore]]:
    """
    HARD: wielocropowy OCR + parsowanie regexem po [NN].
    Dodatkowo recovery brakujących numerów (duże cropy).
    """
    timeout_s = env_int("OPENAI_TIMEOUT_SECONDS", 90)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    panel_box = _chat_panel_box(w, h)

    x1, y1, x2, y2 = panel_box
    panel = img.crop(panel_box)
    pw, ph = panel.size

    # 1) Full panel + 3 pionowe "slices" z overlapem
    crops: List[Image.Image] = [panel]

    # slices: top/mid/bottom z overlapem
    # (na tyle duże, żeby model nie gubił pojedynczej linijki)
    slice_h = int(ph * 0.46)
    overlaps = int(ph * 0.08)
    starts = [0, int(ph * 0.30), int(ph * 0.54)]
    for s in starts:
        sy1 = max(0, s - overlaps)
        sy2 = min(ph, s + slice_h + overlaps)
        crops.append(panel.crop((0, sy1, pw, sy2)))

    prompt = (
        "Przepisz ranking z panelu. "
        "Każdy wpis ma format: [NN] nazwa punkty. "
        "Zwróć tylko te linie (bez komentarzy)."
    )

    maps: List[Dict[int, Tuple[str, int]]] = []
    for c in crops:
        txt = _ocr_text(_img_bytes_from_pil(c), model=model, prompt=prompt, timeout_s=timeout_s)
        maps.append(_parse_chat_lines_to_rank_map(txt))

    merged = _merge_rank_maps(maps)
    if not merged:
        return None

    max_rank = max(merged.keys())
    expected = _guess_expected_size(roster_size)

    # Jeżeli OCR "widzi" większy max_rank niż expected, respektujemy max_rank
    # (np. expected=29 a jednak screen ma 30)
    target_max = max(max_rank, expected)

    # 2) HARD recovery brakujących (szczególnie ważne przy “przesunięciu”)
    for r in range(1, target_max + 1):
        if r not in merged:
            rec = _hard_recover_single_rank(
                img=img,
                panel_box=panel_box,
                rank=r,
                max_rank=target_max,
                model=model,
                timeout_s=timeout_s,
            )
            if rec:
                merged[r] = rec

    # po recovery ustaw max_rank wg realnie posiadanych
    max_rank = max(merged.keys())
    # budujemy listę 1..max_rank, tylko te które mamy (ale NIE przesuwamy!)
    players: List[PlayerScore] = []
    for r in range(1, max_rank + 1):
        if r in merged:
            name, pts = merged[r]
            players.append(PlayerScore(rank=r, name_raw=name, points=pts, name_norm=None))

    return players if players else None

def _needs_hard_fix(players: List[PlayerScore], roster_size: int) -> bool:
    """
    Kiedy odpalić HARD:
    - duplikaty ranków / dziury
    - podejrzanie mało względem rosteru
    - brak [30] przy rosterze 30 itd.
    """
    if not players:
        return True

    ranks = [p.rank for p in players if isinstance(p.rank, int)]
    if not ranks:
        return True

    # jeśli model sam sobie numeruje, zwykle będzie 1..N bez patrzenia na bracket.
    # To samo w przypadku pominięcia jednej linijki.
    uniq = set(ranks)
    if len(uniq) != len(ranks):
        return True

    mx = max(ranks)
    expected = _guess_expected_size(roster_size)

    # dziury w 1..mx
    if uniq != set(range(1, mx + 1)):
        return True

    # jeśli roster sugeruje większy skład niż mx (np. 30 vs 29) → hard
    if expected > mx:
        return True

    # dodatkowo, jeśli jest "podejrzanie mało"
    if expected >= 25 and len(players) < expected:
        return True

    return False


# -----------------------------
# Main parse functions
# -----------------------------

def parse_single_image(image_bytes: bytes, model: str) -> ParsedImage:
    roster_path = env_str("ROSTER_PATH", "roster.json")
    roster = _load_roster(roster_path)
    roster_text = "\n".join(f"- {n}" for n in roster) if roster else "(brak)"

    system_instructions = f"""Jesteś parserem screenów z gry Empires & Puzzles (Alliance War).
Zwracasz TYLKO dane zgodne ze schematem (Structured Output).

1) Rozpoznaj typ obrazka:
   - CHAT z listą „Najlepsi atakujący na wojnach:” (niebieski panel z rankingiem)
   - SOJUSZ/WOJNA z podsumowaniem (paski wyniku, tekst Zwycięstwo/Porażka, tryb)
2) Wyciągnij dane.

Dodatkowo masz listę oficjalnych nicków sojuszu (ROSTER).
Jeśli parsujesz CHAT, to przy każdym graczu:
- name_raw: wpisz dokładnie jak widzisz (z ozdobnikami)
- rank: przepisz dokładnie numer z [NN] jeśli go widać (NIE WYMYŚLAJ, NIE PRZENUMEROWUJ)
- points to liczba (int) widoczna przy graczu (nie zgaduj)

- name_norm: jeśli jesteś pewien, wybierz DOKŁADNIE jeden z nicków z ROSTER. Jeśli nie jesteś pewien, zostaw null.

ROSTER:
{roster_text}

Zasady:
- Ignoruj UI poza panelem z wynikami / paskami.
- Nie dopisuj komentarzy.
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

def _extract_war_mode_fallback(image_bytes: bytes, model: str) -> Optional[str]:
    """
    Tryb wojenny jest pod ikonką trybu i nad przyciskiem 'POLE BITWY' (prawa strona panelu wojny).
    Robimy crop tej okolicy i prosimy model o zwrócenie WYŁĄCZNIE nazwy trybu.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    x1 = int(w * 0.55)
    y1 = int(h * 0.45)
    x2 = int(w * 0.98)
    y2 = int(h * 0.57)

    crop = img.crop((x1, y1, x2, y2))
    crop_bytes = _img_bytes_from_pil(crop)

    prompt = (
        "Odczytaj WYŁĄCZNIE nazwę trybu wojennego z tego fragmentu ekranu. "
        "Nazwa jest bezpośrednio POD okrągłą ikonką trybu i NAD przyciskiem 'POLE BITWY'. "
        "Zwróć samą nazwę (np. 'HORDA NIEUMARŁYCH', 'GRAD STRZAŁ', 'ŻAR Z NIEBA', 'STAROŻYTNY UPIÓR'). "
        "Nie zwracaj 'POLE BITWY' ani innych napisów. Jeśli nie widać – zwróć null."
    )

    timeout_s = env_int("OPENAI_TIMEOUT_SECONDS", 90)
    txt = _ocr_text(crop_bytes, model=model, prompt=prompt, timeout_s=timeout_s)

    # OCR czasem zwróci kilka słów/linijek – bierzemy “najbardziej sensowną” linię:
    if not txt:
        return None
    # usuń ew. "POLE BITWY"
    txt = txt.replace("POLE BITWY", "").strip()
    # pierwsza niepusta linia
    for line in txt.splitlines():
        l = line.strip()
        if l:
            return l
    return None

def parse_war_from_images(images: List[bytes], model: str) -> Tuple[Optional[WarSummary], Optional[List[PlayerScore]], List[ParsedImage]]:
    """
    Zwraca:
      - summary (WarSummary) jeśli rozpoznany
      - players (lista) jeśli rozpoznana lista z chatu
      - debug (lista ParsedImage)
    Dodatkowo:
      - jeśli war_mode brak → fallback crop
      - jeśli lista graczy podejrzana → HARD OCR po [NN] i ewentualny recovery braków
    """
    roster_path = env_str("ROSTER_PATH", "roster.json")
    roster = _load_roster(roster_path)
    roster_size = len(roster)

    parsed: List[ParsedImage] = []
    parsed_with_bytes: List[Tuple[bytes, ParsedImage]] = []

    for b in images:
        p = parse_single_image(b, model=model)
        parsed.append(p)
        parsed_with_bytes.append((b, p))

    summary: Optional[WarSummary] = None
    players: Optional[List[PlayerScore]] = None
    chat_img_bytes: Optional[bytes] = None
    summary_img_bytes: Optional[bytes] = None

    for b, p in parsed_with_bytes:
        if p.kind == "war_summary" and p.war_summary:
            summary = p.war_summary
            summary_img_bytes = b
        if p.kind == "chat_results" and p.chat_results:
            players = p.chat_results.players
            chat_img_bytes = b

    # --- war_mode fallback ---
    if summary and not summary.war_mode and summary_img_bytes:
        wm = _extract_war_mode_fallback(summary_img_bytes, model=model)
        if wm:
            summary = summary.model_copy(update={"war_mode": wm})

    # --- HARD fix dla listy graczy (żeby nie było “przesunięć”) ---
    if players and chat_img_bytes and _needs_hard_fix(players, roster_size=roster_size):
        hard = _hard_parse_chat_results(chat_img_bytes, model=model, roster_size=roster_size)
        if hard and len(hard) >= max(10, len(players)):  # sanity check
            players = hard

    return summary, players, parsed
