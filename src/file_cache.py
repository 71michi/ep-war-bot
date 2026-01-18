"""Tiny file/JSON cache helpers.

Used to avoid repeatedly reading small JSON files (roster, aliases) on hot paths.
The cache is invalidated by mtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _safe_mtime(path: str) -> float:
    try:
        return float(os.stat(path).st_mtime)
    except Exception:
        return -1.0


@dataclass
class CachedJSON:
    mtime: float
    data: Dict[str, Any]


_JSON_CACHE: Dict[str, CachedJSON] = {}


def load_json_cached(path: str, *, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a JSON file with a small mtime cache.

    Args:
        path: path to JSON file
        default: returned when file missing/invalid
    """
    if default is None:
        default = {}

    path = str(path)
    mtime = _safe_mtime(path)
    cached = _JSON_CACHE.get(path)
    if cached is not None and cached.mtime == mtime:
        return cached.data

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = dict(default)
    except Exception:
        data = dict(default)

    _JSON_CACHE[path] = CachedJSON(mtime=mtime, data=data)
    return data


def clear_json_cache() -> None:
    """Clear the in-memory JSON cache (mainly for tests)."""
    _JSON_CACHE.clear()
