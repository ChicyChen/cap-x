"""Replay CaP-X gt_state dumps through the vlm-orchestrator Aspect-1 logger.

CaP-X owns the env loop, so the orchestrator's online logger never
sees these episodes. Instead, ``FrankaRobolabEnv`` writes per-step
``gt_state.<pid>.<ts>.jsonl`` files. This script walks those files
and replays them through ``TaskFailureLogger`` to produce
``task_failures.jsonl`` files in the on-disk shape the orchestrator's
aggregation scripts already expect.

Output layout (matches vlm-orchestrator's per-episode dir):

    <out_root>/<task_slug>/episode_<N>/
        task_failures.jsonl   # written by TaskFailureLogger
        metadata.json         # written by this script

Usage:
    python scripts/score_capx_with_aspect1.py \\
        --capx-output-dir ./outputs/franka_robolab_privileged \\
        --out-root ./outputs/franka_robolab_privileged/aspect1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def _slugify(text: str) -> str:
    """Match vlm-orchestrator's instruction → directory slug convention."""
    return re.sub(r"[^\w\s-]", "", text or "").strip().replace(" ", "_") or "episode"


def _iter_episodes(dump_files: list[Path]):
    """Yield (gt_state_path, instruction, num_steps, last_record) per file.

    Each file is one CaP-X trial. The instruction is taken from the
    first record (every record carries it, but we only need it once).
    """
    for path in sorted(dump_files):
        try:
            with path.open() as fh:
                first = fh.readline().strip()
                if not first:
                    continue
                first_rec = json.loads(first)
                count = 1
                last_rec = first_rec
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    last_rec = json.loads(line)
                    count += 1
        except (OSError, json.JSONDecodeError) as e:
            print(f"[scorer] skipping {path}: {e}", file=sys.stderr)
            continue
        instruction = first_rec.get("instruction", "")
        yield path, instruction, count, last_rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capx-output-dir",
        required=True,
        help="CaP-X output dir containing gt_state.*.jsonl files (recursive scan).",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Root dir to write per-episode task_failures.jsonl into.",
    )
    parser.add_argument(
        "--scene-name",
        default=None,
        help=(
            "Override the task slug used in <out_root>/<slug>/. Defaults to "
            "a slug derived from the instruction string in each dump."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk dumps and print plan without writing.",
    )
    args = parser.parse_args()

    try:
        # vlm_orchestrator renamed `task_failure_logger` → `metrics1_failure_logger`
        # in 2026-05. Keep the old path as a fallback for stale Lustre checkouts.
        try:
            from vlm_orchestrator.diagnostics.metrics1_failure_logger import TaskFailureLogger
        except ModuleNotFoundError:
            from vlm_orchestrator.diagnostics.task_failure_logger import TaskFailureLogger
    except ImportError as e:
        print(
            "vlm-orchestrator not importable. Install it editable into this venv:\n"
            "    pip install -e /path/to/vlm-orchestrator\n"
            f"original error: {e}",
            file=sys.stderr,
        )
        return 1

    capx_dir = Path(args.capx_output_dir).resolve()
    out_root = Path(args.out_root).resolve()
    if not capx_dir.is_dir():
        print(f"capx-output-dir not found: {capx_dir}", file=sys.stderr)
        return 1

    dump_files = sorted(capx_dir.rglob("gt_state.*.jsonl"))
    if not dump_files:
        # Fall back to the simpler single-file pattern.
        dump_files = sorted(capx_dir.rglob("gt_state.jsonl"))
    if not dump_files:
        print(
            f"No gt_state.jsonl dumps found under {capx_dir}. Make sure the "
            "YAML config sets `gt_state_dump_path` on FrankaRobolabTask.",
            file=sys.stderr,
        )
        return 1

    # Group episodes by slug so episode_<N> indices are sequential per task.
    grouped: dict[str, list[tuple[Path, str, int, dict]]] = {}
    for path, instruction, count, last in _iter_episodes(dump_files):
        slug = args.scene_name or _slugify(instruction)
        grouped.setdefault(slug, []).append((path, instruction, count, last))

    total_episodes = sum(len(v) for v in grouped.values())
    print(
        f"[scorer] found {len(dump_files)} dump(s) → "
        f"{total_episodes} episode(s) across {len(grouped)} task slug(s)"
    )

    if args.dry_run:
        for slug, eps in grouped.items():
            print(f"  {slug}: {len(eps)} episode(s)")
        return 0

    written = 0
    for slug, eps in grouped.items():
        for ep_idx, (path, instruction, num_steps, last_rec) in enumerate(eps):
            episode_dir = out_root / slug / f"episode_{ep_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            logger = TaskFailureLogger()
            logger.on_episode_start(
                episode_log_dir=str(episode_dir),
                episode_id=ep_idx,
                instruction=instruction,
            )

            # Replay step-by-step.  ``observe`` short-circuits when no
            # ``gt_state`` is in the obs, so wrap each record correctly.
            try:
                with path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        gt_state = rec.get("gt_state")
                        if gt_state is None:
                            continue
                        logger.observe(
                            {"gt_state": gt_state},
                            step=int(rec.get("step", 0)),
                        )
            finally:
                logger.on_episode_end()

            # Mirror the orchestrator's metadata.json shape so
            # downstream aggregators don't have to special-case capx.
            metadata = {
                "episode_id": ep_idx,
                "instruction": instruction,
                "task_slug": slug,
                "episode_dir": str(episode_dir),
                "num_steps": num_steps,
                "success": bool(last_rec.get("task_completed", False)),
                "source": "capx",
                "source_dump": str(path),
                "end_timestamp": time.time(),
            }
            (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            written += 1

    print(f"[scorer] wrote {written} episode dir(s) under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
