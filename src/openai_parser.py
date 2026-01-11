import base64
import io
import json
from typing import List, Optional, Literal, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image

from .config import env_str, env_int

client = OpenAI()


# ----------------------------
# Models (Structured Outputs)
# ----------------------------

class WarModeOnly(BaseModel):
    war_mode: Optional[str] = None


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


# ----------------------------
# Helpers
# ----------------------------

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
      - name_norm: jeżeli jesteś PEWNY, wybierz DOKŁADNIE jeden z nicków z ROSTER, inaczej null

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
) -> Tuple[Optional[WarSummary], Optional[List[PlayerScore]], List[ParsedImage]]:
    """
    images: lista obrazków (typowo 2 szt.: chat + war summary)
    Zwraca: summary, players, parsed_debug
    """
    parsed_debug: List[ParsedImage] = []
    parsed_with_bytes: List[tuple[bytes, ParsedImage]] = []

    for b in images:
        p = parse_single_image(b, model=model)
        parsed_debug.append(p)
        parsed_with_bytes.append((b, p))

    summary: Optional[WarSummary] = None
    players: Optional[List[PlayerScore]] = None
    summary_image_bytes: Optional[bytes] = None

    for b, p in parsed_with_bytes:
        if p.kind == "war_summary" and p.war_summary:
            summary = p.war_summary
            summary_image_bytes = b
        if p.kind == "chat_results" and p.chat_results:
            players = p.chat_results.players

    # Fallback na war_mode: jeśli model pominął tryb na pełnym screenie, doczytaj z cropa
    if summary and (not summary.war_mode or not summary.war_mode.strip()) and summary_image_bytes:
        wm = _extract_war_mode_fallback(summary_image_bytes, model=model)
        if wm:
            summary.war_mode = wm

    return summary, players, parsed_debug
