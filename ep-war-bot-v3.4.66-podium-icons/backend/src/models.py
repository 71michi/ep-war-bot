from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class PlayerResult(BaseModel):
    player: str
    points: int = Field(ge=0)
    mode: str
    attacks_used: int = Field(default=6, ge=0)
    roster_status: Literal["active", "outside"] = "active"


class War(BaseModel):
    id: str
    date: str
    opponent: str
    result: Literal["Zwycięstwo", "Porażka", "Remis"]
    alliance_points: int = Field(ge=0)
    opponent_points: int = Field(ge=0)
    participant_count: int = Field(ge=1, le=30)
    mode: str
    players: list[PlayerResult]
