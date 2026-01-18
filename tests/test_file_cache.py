import json
from pathlib import Path

from src.file_cache import load_json_cached, clear_json_cache


def test_load_json_cached_invalidates_on_mtime(tmp_path: Path):
    clear_json_cache()
    p = tmp_path / "data.json"

    p.write_text(json.dumps({"roster": ["A"]}), encoding="utf-8")
    d1 = load_json_cached(str(p))
    assert d1["roster"] == ["A"]

    # Re-read should come from cache (same mtime)
    d2 = load_json_cached(str(p))
    assert d2 is d1

    # Changing file updates mtime and invalidates cache
    p.write_text(json.dumps({"roster": ["B"]}), encoding="utf-8")
    d3 = load_json_cached(str(p))
    assert d3["roster"] == ["B"]
