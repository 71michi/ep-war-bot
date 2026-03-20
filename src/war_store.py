import json
import os
import time
import logging
import hashlib
from typing import Any, Dict, List, Optional


DEFAULT_STORE: Dict[str, Any] = {
    "version": 1,
    "wars": {},  # war_id -> record
    "order": [],  # list[war_id] in insertion order
}


logger = logging.getLogger("warbot.store")


def _points_checksum(record: Dict[str, Any]) -> str:
    """Stable checksum of a war's *raw* player points.

    Used to prevent accidental historical corruption (e.g. scaled points being written back into the store).
    """
    players = record.get("players")
    if not isinstance(players, list):
        return ""
    norm = []
    for p in players:
        if not isinstance(p, dict):
            continue
        pts = p.get("points_raw")
        if pts is None:
            pts = p.get("points")
        try:
            pts_i = int(pts)
        except Exception:
            pts_i = 0
        try:
            rank_i = int(p.get("rank") or 0)
        except Exception:
            rank_i = 0
        name = str(p.get("name") or p.get("name_display") or "").strip()
        norm.append((rank_i, name, pts_i))
    norm.sort(key=lambda x: (x[0], x[1]))
    blob = json.dumps(norm, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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


        prev_rec = wars.get(war_id) if existed else None

        # Historical corruption guard: do not allow accidental point changes in CONFIRMED wars.
        # Allow override via env ALLOW_CONFIRMED_WAR_MUTATION=1
        if existed and isinstance(prev_rec, dict):
            try:
                prev_status = str(prev_rec.get("status") or "").lower()
                new_status = str(record.get("status") or "").lower() or prev_status
                if prev_status == "confirmed" and new_status == "confirmed":
                    if os.getenv("ALLOW_CONFIRMED_WAR_MUTATION", "0") != "1":
                        prev_sum = _points_checksum(prev_rec)
                        new_sum = _points_checksum(record)
                        if prev_sum and new_sum and prev_sum != new_sum:
                            logger.error(
                                "Refusing to modify CONFIRMED war %s (points checksum changed). Set ALLOW_CONFIRMED_WAR_MUTATION=1 to override.",
                                war_id,
                            )
                            return True
            except Exception:
                pass

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

    # ----------------------------
    # LISTWAR ref-message sequencing
    # ----------------------------
    #
    # Users sometimes want to re-run LISTWAR for the same screenshot message and
    # get a *new* war_id (e.g. after an UNLISTWAR reset). We keep a small
    # per-ref-message sequence counter in the store to make new IDs deterministic
    # and collision-free across restarts.

    def get_ref_sequence(self, ref_message_id: int) -> int:
        """Return current LISTWAR sequence for a referenced screenshot message."""
        store = self.load()
        seq = store.get("ref_sequences")
        if not isinstance(seq, dict):
            return 0
        try:
            return int(seq.get(str(int(ref_message_id)), 0) or 0)
        except Exception:
            return 0

    def bump_ref_sequence(self, ref_message_id: int) -> int:
        """Increment and persist LISTWAR sequence for a referenced screenshot message."""
        store = self.load()
        seq = store.get("ref_sequences")
        if not isinstance(seq, dict):
            seq = {}
        key = str(int(ref_message_id))
        cur = 0
        try:
            cur = int(seq.get(key, 0) or 0)
        except Exception:
            cur = 0
        cur += 1
        seq[key] = cur
        store["ref_sequences"] = seq
        self.save(store)
        return cur

    def count(self) -> int:
        store = self.load()
        order: List[str] = list(store.get("order", []))
        return len(order)
