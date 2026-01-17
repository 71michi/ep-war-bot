"""Batch runner for offline testing on many screenshots.

This script executes the same pipeline as the Discord bot:
  parse_war_from_images -> build_post -> render_post

It is meant to be run locally (or on a VM) with OPENAI_API_KEY in .env.

Folder layouts supported:

A) Subfolders = cases (recommended)
   input/
     test01/
       chat.png
       summary.png
     test02/
       1.png
       2.png

B) Flat files grouped by prefix before '_' or '-'
   input/
     test01_chat.png
     test01_summary.png
     test02_a.png
     test02_b.png

Outputs (per case):
  - <case>.post.txt   : exactly what the bot would post on Discord
  - <case>.raw.json   : raw parsed data (summary, players, parsed_debug)
  - <case>.log.txt    : detailed step-by-step logs for this run

And a summary table:
  - index.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

from .config import env_str
from .logging_setup import setup_logging, set_trace_id, reset_trace_id
from .openai_parser import parse_war_from_images

# We reuse the exact bot formatting + roster mapping logic.
# NOTE: importing .bot pulls in discord.py, but that's already in requirements.
from .bot import build_post, render_post, _new_trace_id, _attach_per_trace_logger, _redact_secrets


logger = logging.getLogger("warbot.batch")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _kind_score(p: Path) -> int:
    name = p.name.lower()
    # Prefer chat + summary if present, but order does not matter for parsing.
    if "chat" in name:
        return 0
    if "sum" in name or "summary" in name or "wojna" in name:
        return 1
    return 2


def _pick_two(images: List[Path]) -> List[Path]:
    if not images:
        return []

    images = sorted(
        images,
        key=lambda p: (
            _kind_score(p),
            -int(p.stat().st_size) if p.exists() else 0,
            p.name.lower(),
        ),
    )
    if len(images) <= 2:
        return images
    return images[:2]


def collect_cases(input_dir: Path) -> Dict[str, List[Path]]:
    """Return mapping: case_id -> list of up to 2 image paths."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    entries = [p for p in input_dir.iterdir()]
    subdirs = [p for p in entries if p.is_dir()]

    cases: Dict[str, List[Path]] = {}

    if subdirs:
        for d in sorted(subdirs, key=lambda p: p.name.lower()):
            imgs = [p for p in d.iterdir() if _is_image(p)]
            picked = _pick_two(imgs)
            if picked:
                cases[d.name] = picked
        return cases

    # Flat layout
    imgs = [p for p in entries if _is_image(p)]
    for p in imgs:
        stem = p.stem
        m = re.match(r"^(.+?)[_-].+$", stem)
        case_id = m.group(1) if m else stem
        cases.setdefault(case_id, []).append(p)

    # Pick 2 per case
    out: Dict[str, List[Path]] = {}
    for cid, lst in cases.items():
        picked = _pick_two(lst)
        if picked:
            out[cid] = picked
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _safe_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_case(case_id: str, img_paths: List[Path], out_dir: Path, model: str) -> Dict[str, object]:
    trace_id = _new_trace_id(prefix=f"batch-{case_id}")

    # Capture per-trace logs into a dedicated file.
    handler, stream = _attach_per_trace_logger(trace_id)
    token = set_trace_id(trace_id)

    try:
        logger.info("===== START batch case=%s images=%s model=%s =====", case_id, [p.name for p in img_paths], model)

        images_bytes: List[bytes] = [p.read_bytes() for p in img_paths]

        summary, players, expected_max_rank, parsed_debug = parse_war_from_images(
            images_bytes,
            model=model,
            trace_id=trace_id,
        )

        raw = {
            "case_id": case_id,
            "trace_id": trace_id,
            "model": model,
            "images": [str(p) for p in img_paths],
            "summary": summary.model_dump() if summary else None,
            "players": [p.model_dump() for p in (players or [])],
            "expected_max_rank": expected_max_rank,
            "parsed_debug": [p.model_dump() for p in (parsed_debug or [])],
        }

        status = "ERROR"
        post_text = ""
        missing = unknown = out_of_roster = []

        if summary and players:
            post = build_post(summary, players, expected_max_rank)
            post_text = render_post(post)
            missing = post.missing_ranks()
            unknown = post.unknown_ranks()
            out_of_roster = post.out_of_roster_ranks()

            raw["post"] = {
                "missing": missing,
                "unknown": unknown,
                "out_of_roster": out_of_roster,
            }

            if missing or unknown or out_of_roster:
                status = "NEEDS_FIX"
            else:
                status = "OK"
        else:
            # Partial parse: still dump what we have.
            status = "ERROR"

        # Write outputs
        _safe_write_text(out_dir / f"{case_id}.post.txt", post_text)
        _safe_write_json(out_dir / f"{case_id}.raw.json", raw)

        return {
            "case_id": case_id,
            "trace_id": trace_id,
            "model": model,
            "num_images": len(img_paths),
            "status": status,
            "our_score": int(summary.our_score) if summary else None,
            "opp_score": int(summary.opponent_score) if summary else None,
            "diff": (int(summary.our_score) - int(summary.opponent_score)) if summary else None,
            "war_mode": summary.war_mode if summary else None,
            "missing": len(missing) if isinstance(missing, list) else None,
            "unknown": len(unknown) if isinstance(unknown, list) else None,
            "out_of_roster": len(out_of_roster) if isinstance(out_of_roster, list) else None,
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Case failed: %s", case_id)

        _safe_write_text(out_dir / f"{case_id}.post.txt", "")
        _safe_write_json(
            out_dir / f"{case_id}.raw.json",
            {
                "case_id": case_id,
                "trace_id": trace_id,
                "model": model,
                "images": [str(p) for p in img_paths],
                "error": str(e),
                "traceback": tb,
            },
        )
        return {
            "case_id": case_id,
            "trace_id": trace_id,
            "model": model,
            "num_images": len(img_paths),
            "status": "ERROR",
            "our_score": None,
            "opp_score": None,
            "diff": None,
            "war_mode": None,
            "missing": None,
            "unknown": None,
            "out_of_roster": None,
        }

    finally:
        # Write per-trace log
        try:
            logs = stream.getvalue()
        except Exception:
            logs = ""
        logs = _redact_secrets(logs)
        _safe_write_text(out_dir / f"{case_id}.log.txt", logs)

        # Detach capture handler
        try:
            logging.getLogger().removeHandler(handler)
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass

        reset_trace_id(token)
        logger.info("===== END batch case=%s =====", case_id)


def write_index(rows: List[Dict[str, object]], out_dir: Path) -> None:
    out_path = out_dir / "index.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "case_id",
        "status",
        "num_images",
        "diff",
        "our_score",
        "opp_score",
        "war_mode",
        "missing",
        "unknown",
        "out_of_roster",
        "trace_id",
        "model",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main() -> None:
    setup_logging()

    ap = argparse.ArgumentParser(description="Batch test war screenshots using the same pipeline as the Discord bot.")
    ap.add_argument("--input", required=True, help="Input directory with screenshots")
    ap.add_argument("--out", required=True, help="Output directory for results")
    ap.add_argument(
        "--model",
        default=env_str("OPENAI_MODEL", "gpt-4o"),
        help="OpenAI model to use (default: OPENAI_MODEL env var)",
    )
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    model = str(args.model).strip()

    cases = collect_cases(input_dir)
    if not cases:
        raise SystemExit(f"No images found in {input_dir}")

    logger.info("Batch cases found: %d", len(cases))

    rows: List[Dict[str, object]] = []
    for case_id, paths in cases.items():
        rows.append(run_case(case_id, paths, out_dir, model=model))

    write_index(rows, out_dir)

    ok = sum(1 for r in rows if r.get("status") == "OK")
    needs = sum(1 for r in rows if r.get("status") == "NEEDS_FIX")
    err = sum(1 for r in rows if r.get("status") == "ERROR")
    logger.info("Batch finished: OK=%d NEEDS_FIX=%d ERROR=%d (results: %s)", ok, needs, err, out_dir)


if __name__ == "__main__":
    main()
