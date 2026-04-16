from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

DATA_FILE = Path(__file__).with_name("sample_data.json")

app = FastAPI(title="EP War Bot API", version="3.4.64")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_data() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def root() -> dict:
    return {"name": "EP War Bot API", "version": "3.4.64", "status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/meta")
def meta() -> dict:
    return {
        "version": "3.4.64",
        "build": "reconstructed",
        "features": [
            "compact-war-details",
            "mode-icon-near-score",
            "average-divider-line",
            "above-below-average-indicator",
        ],
    }


@app.get("/api/wars")
def wars() -> list[dict]:
    return load_data()["wars"]
