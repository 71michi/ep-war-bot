from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class PlayerScore(BaseModel):
    rank: int = Field(ge=1)
    name_raw: str
    points: int = Field(ge=0)

class ChatResults(BaseModel):
    title: str = Field(default="Najlepsi atakujący na wojnach")
    players: List[PlayerScore]

class WarSummary(BaseModel):
    our_alliance: str
    opponent_alliance: str
    result: Literal["Zwycięstwo", "Porażka"]
    our_score: int = Field(ge=0)
    opponent_score: int = Field(ge=0)
    war_mode: Optional[str] = None
    beta_badge: Optional[bool] = None

class ParsedImage(BaseModel):
    kind: Literal["chat_results", "war_summary", "unknown"]
    chat_results: Optional[ChatResults] = None
    war_summary: Optional[WarSummary] = None
    confidence: float = Field(ge=0, le=1)
    notes: Optional[str] = None
