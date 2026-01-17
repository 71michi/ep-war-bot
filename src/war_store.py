import json
import os
import time
from typing import Any, Dict, List, Optional


DEFAULT_STORE: Dict[str, Any] = {
    "version": 1,
    "wars": {},  # war_id -> record
    "order": [],  # list[war_id] in insertion order
}


def _now_ts() -> int:
    return int(time.time())


def _safe_stat_mtime(path: str) -> float:
    try:
        return float(os.stat(path).st_mtime)
    except Exception:
        return -1.0


def _deepcopy_default() -> Dict[str, Any]:
    # Manual deep copy (fast + no extra deps)
    return {
        "version": int(DEFAULT_STORE.get("version", 1)),
        "wars": {},
        "order": [],
    }


class WarStore:
    """Tiny JSON store with atomic writes and mtime caching."""

    def __init__(self, path: str):
        self.path = path
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_mtime: float = -2.0

    def load(self) -> Dict[str, Any]:
        mtime = _safe_stat_mtime(self.path)
        if self._cache is not None and mtime == self._cache_mtime:
            return self._cache

        if not os.path.exists(self.path):
            store = _deepcopy_default()
            self._cache = store
            self._cache_mtime = mtime
            return store

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                store = json.load(f)
            if not isinstance(store, dict):
                store = _deepcopy_default()
        except Exception:
            store = _deepcopy_default()

        store.setdefault("version", 1)
        store.setdefault("wars", {})
        store.setdefault("order", [])
        if not isinstance(store["wars"], dict):
            store["wars"] = {}
        if not isinstance(store["order"], list):
            store["order"] = []

        self._cache = store
        self._cache_mtime = mtime
        return store

    def save(self, store: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)
        self._cache = store
        self._cache_mtime = _safe_stat_mtime(self.path)

    def upsert_war(self, war_id: str, record: Dict[str, Any]) -> bool:
        """Insert or update a war. Returns True if it existed already."""
        war_id = str(war_id).strip()
        if not war_id:
            return False

        store = self.load()
        wars: Dict[str, Any] = store.get("wars", {})
        order: List[str] = list(store.get("order", []))

        existed = war_id in wars
        record = dict(record)
        record.setdefault("war_id", war_id)
        record.setdefault("updated_at_ts", _now_ts())

        wars[war_id] = record
        if not existed:
            order.append(war_id)

        store["version"] = int(store.get("version", 1))
        store["wars"] = wars
        store["order"] = order
        self.save(store)
        return existed

    # Backwards compatibility: older code used get_wars_list(newest_first)
    def get_wars_list(self, newest_first: bool = True) -> List[Dict[str, Any]]:
        return self.get_wars(newest_first=newest_first)

    def delete_war(self, war_id: str) -> bool:
        """Delete a war from the store. Returns True if it existed."""
        war_id = str(war_id).strip()
        if not war_id:
            return False

        store = self.load()
        wars: Dict[str, Any] = store.get("wars", {})
        order: List[str] = list(store.get("order", []))

        existed = war_id in wars
        if existed:
            try:
                wars.pop(war_id, None)
            except Exception:
                pass
            order = [x for x in order if x != war_id]
            store["wars"] = wars
            store["order"] = order
            self.save(store)
        return existed

    def get_wars(self, newest_first: bool = True) -> List[Dict[str, Any]]:
        store = self.load()
        wars: Dict[str, Any] = store.get("wars", {})
        order: List[str] = list(store.get("order", []))
        ids = list(reversed(order)) if newest_first else order
        out: List[Dict[str, Any]] = []
        for wid in ids:
            rec = wars.get(wid)
            if isinstance(rec, dict):
                out.append(rec)
        return out

    def get_war(self, war_id: str) -> Optional[Dict[str, Any]]:
        store = self.load()
        wars: Dict[str, Any] = store.get("wars", {})
        rec = wars.get(war_id)
        return rec if isinstance(rec, dict) else None

    def count(self) -> int:
        store = self.load()
        order: List[str] = list(store.get("order", []))
        return len(order)
