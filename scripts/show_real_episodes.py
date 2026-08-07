#!/usr/bin/env python
"""Map real-robot EPISODES to the trial folders CaP-X wrote.

Why this exists
---------------
Three different numbering schemes overlap in ``outputs/franky_real``, which is
genuinely confusing:

With ``capx.envs.real_launch`` the layout is now episode-shaped:

    <root>/episode_024/attempt_1/   <- a full CaP-X trial dir
                       attempt_2/
                       episode.json
           episode_025/attempt_1/
           episodes.jsonl

so most of the time you can just ``ls``. This script summarises it, and still
works for OLD runs made with ``capx.envs.launch``, whose flat
``trial_NN_sandboxrc_..._reward_...`` dirs carry no episode id -- for those,
``episodes.jsonl`` is the only join key.

Usage
-----
    python scripts/show_real_episodes.py                    # newest run
    python scripts/show_real_episodes.py --all              # every run
    python scripts/show_real_episodes.py --root outputs/franky_real/2026...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path("outputs/franky_real")


def _ledgers(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(root.rglob("episodes.jsonl"), key=lambda p: p.stat().st_mtime)


def _trial_dirs(out_dir: pathlib.Path) -> dict[int, pathlib.Path]:
    """trial number -> its directory (the name also encodes rc/reward)."""
    found: dict[int, pathlib.Path] = {}
    if not out_dir.is_dir():
        return found
    for d in out_dir.iterdir():
        if d.is_dir() and d.name.startswith("attempt_"):
            try:
                found[int(d.name.split("_")[1])] = d
            except (IndexError, ValueError):
                continue
        if d.is_dir() and d.name.startswith("trial_"):
            try:
                found[int(d.name.split("_")[1])] = d
            except (IndexError, ValueError):
                continue
    return found


def render(ledger: pathlib.Path) -> None:
    rows = [
        json.loads(line)
        for line in ledger.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        return
    out_dir = ledger.parent
    trials = _trial_dirs(out_dir)

    print(f"\n\033[1m{out_dir}\033[0m")
    by_ep: dict[object, list[dict]] = {}
    for r in rows:
        by_ep.setdefault(r["episode_id"], []).append(r)

    for ep, attempts in by_ep.items():
        task = attempts[0].get("task") or "(unknown)"
        print(f"\n  \033[36mepisode {ep}\033[0m  {task!r}")
        for a in attempts:
            d = trials.get(a.get("trial"))
            where = d.name if d else "\033[33m(no trial dir yet)\033[0m"
            n_code = len(list(d.glob("code.py"))) if d else 0
            extra = ""
            if d:
                imgs = len(list(d.glob("visual_feedback_*.png")))
                extra = f"  code.py={'yes' if n_code else 'no'}  frames={imgs}"
            print(
                f"    attempt {a['attempt']}  {a['iso']}  "
                f"trial={a.get('trial')}  {where}{extra}"
            )

    print(
        f"\n  {len(by_ep)} episode(s), {len(rows)} plan attempt(s). "
        "Success is scored by the operator -- reward/taskcompleted in the "
        "folder names are NOT valid on hardware."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--all", action="store_true", help="every run, not just the newest")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        print(f"no such path: {root}", file=sys.stderr)
        return 1

    found = _ledgers(root)
    if not found:
        print(
            f"no episodes.jsonl under {root}\n"
            "(only runs started after the ledger was added will have one)",
            file=sys.stderr,
        )
        return 1

    for led in found if args.all else found[-1:]:
        render(led)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
