"""Episode-shaped runner for the real robot.

Why this file exists
--------------------
On hardware the OPERATOR owns episodes: the driver increments ``episode_id``
whenever a new episode is started at the robot. CaP-X's own numbering does not
line up with that:

* ``pass_N`` is one python **process**. Ending the process per episode would
  close the WebSocket port, and ``openpi_client`` never reconnects -- that is the
  "no close frame received or sent" failure. So one process must serve many
  episodes.
* ``trial_NN_...`` is one **plan attempt**. When a plan finishes, the runner
  immediately starts another trial on the SAME episode.

Result: artifacts for two different episodes landed side by side as consecutive
trials in one flat directory, and the pass-level ``all_responses.json`` /
``initial_prompt.txt`` are rewritten by every trial (``trial.py`` opens them in
mode ``"w"`` at ``config["output_dir"]``), so they only ever describe the last
attempt.

This runner fixes the layout by **redirecting** ``config["output_dir"]`` per
episode before each trial. Every one of ``trial.py``'s 13 read sites resolves the
path at use time, so no CaP-X file needs to change:

    <root>/
      episodes.jsonl                  ledger: (episode_id, attempt) -> dir
      episode_024/
        attempt_1/                    <- a full CaP-X trial dir
          code.py  summary.txt  all_responses.json  prompts_and_responses/
        attempt_2/
        episode.json                  task, timestamps, attempt count
      episode_025/
        attempt_1/

Nothing here is imported by the sim/robolab path: ``launch.py`` only reaches it
via ``env_configs/real/*`` setting ``real_episode_layout: true``.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

from capx.envs.configs.instantiate import instantiate
from capx.envs.runner import _run_trial_with_retries, _setup_output_dir
from capx.utils.launch_utils import _print_and_save_summary


def _episode_dir(root: str, episode: Any) -> pathlib.Path:
    """``episode_024`` for ints, sanitised text otherwise."""
    if episode is None:
        # No episode at all: this trial timed out waiting for the driver to
        # connect, so it is a failed startup wait rather than an episode. Keep it
        # out of the episode_* namespace entirely -- it was confusing to see it
        # listed beside real episodes (and note episode_id 0 IS valid, so the
        # check must be `is None`, not falsiness).
        return pathlib.Path(root) / "no_episode_trials"
    try:
        name = f"episode_{int(episode):03d}"
    except (TypeError, ValueError):
        safe = "".join(c if c.isalnum() else "_" for c in str(episode))
        name = f"episode_{safe or 'unknown'}"
    return pathlib.Path(root) / name


def _write_episode_json(
    ep_dir: pathlib.Path,
    *,
    episode: Any,
    task: str | None,
    attempts: int,
    started: float,
) -> None:
    """Per-episode manifest. Best-effort; never breaks a run."""
    try:
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "episode.json").write_text(
            json.dumps(
                {
                    "episode_id": episode,
                    "task": task,
                    "attempts": attempts,
                    "started": started,
                    "started_iso": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(started)
                    ),
                    "updated_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "note": (
                        "Success on hardware is scored MANUALLY by the operator. "
                        "reward / taskcompleted in trial dir names are always "
                        "0 here and must not be aggregated."
                    ),
                },
                indent=2,
            )
        )
    except Exception:
        pass


def run_real_episode_trials(
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
    start_time: float,
) -> None:
    """Serve many operator episodes from one process, one directory each.

    Mirrors ``runner._run_headless_trials`` for the sequential case, but chooses
    the output directory from the live ``episode_id`` instead of a flat trial
    counter. ``num_workers > 1`` is meaningless with a single physical robot, so
    it is not supported here.
    """
    if config["total_trials"] <= 0:
        print("No trials requested; exiting.")
        return

    _setup_output_dir(args, config)
    root = config["output_dir"]
    ledger = os.path.join(root, "episodes.jsonl")
    os.makedirs(root, exist_ok=True)

    _adopt_orphans(root)

    env = instantiate(env_factory)
    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)

    print(
        f"[real] episode-shaped output under {root}\n"
        "[real] one directory per operator episode; attempt_N = plan attempt",
        flush=True,
    )

    summaries = []
    # Seed from disk: the launcher restarts this process on a crash and reuses
    # the same root, so attempt numbering has to continue rather than restart
    # at 1 and collide with an existing directory.
    attempts_by_ep: dict[Any, int] = _existing_attempts(root)
    if attempts_by_ep:
        print(
            f"[real] resuming: {len(attempts_by_ep)} episode(s) already on disk "
            f"({', '.join(f'{k}={v}' for k, v in sorted(attempts_by_ep.items(), key=str))})",
            flush=True,
        )
    started_by_ep: dict[Any, float] = {}
    task_by_ep: dict[Any, str | None] = {}

    for trial in range(1, config["total_trials"] + 1):
        # The episode id does not exist until the driver's first frame arrives,
        # and that happens INSIDE reset(). So run the trial in a staging dir and
        # file it under its episode afterwards -- naming it up front produced an
        # "episode_pending" directory for every first trial.
        staging = pathlib.Path(root) / f".staging_trial_{trial:03d}"
        staging.mkdir(parents=True, exist_ok=True)
        config["output_dir"] = str(staging)
        try:
            summary = _run_trial_with_retries(
                env, trial, args, config, multi_turn_prompt
            )
            summaries.append(summary)
        finally:
            config["output_dir"] = root

        # Now the episode is known (reset() adopted it from the wire).
        episode = _peek_episode(env)
        attempt = attempts_by_ep.get(episode, 0) + 1
        attempts_by_ep[episode] = attempt
        started_by_ep.setdefault(episode, time.time())

        ep_dir = _episode_dir(root, episode)
        trial_out = ep_dir / f"attempt_{attempt}"
        _relocate(staging, trial_out)

        task = getattr(env, "_wire_task", None)
        # Keep the FIRST task seen for this episode: if the operator advanced
        # mid-trial, env._wire_task already describes the NEXT episode.
        task_by_ep.setdefault(episode, task)
        if episode is None:
            # A failed startup wait: no manifest, and say plainly what it was.
            print(
                f"[real] trial {trial} never received a driver frame; "
                f"artifacts under {trial_out.parent.name}/",
                flush=True,
            )
            _append_ledger(
                ledger,
                {
                    "ts": time.time(),
                    "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "episode_id": None,
                    "attempt": attempt,
                    "trial": trial,
                    "task": None,
                    "output_dir": str(trial_out),
                    "note": "no driver frame arrived; not an episode",
                },
            )
            continue
        _write_episode_json(
            ep_dir,
            episode=episode,
            task=task_by_ep[episode],
            attempts=attempt,
            started=started_by_ep[episode],
        )
        _append_ledger(
            ledger,
            {
                "ts": time.time(),
                "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                "episode_id": episode,
                "attempt": attempt,
                "trial": trial,
                "task": task_by_ep.get(episode),
                "output_dir": str(trial_out),
            },
        )

    config["output_dir"] = root
    summaries.sort(key=lambda s: s.trial)
    _print_and_save_summary(summaries, args, config, start_time)


def _adopt_orphans(root: str) -> None:
    """Rescue artifacts from a trial that was killed mid-flight.

    ``_relocate`` only runs after a trial RETURNS, so a process killed during a
    trial leaves its output in ``.staging_trial_NNN`` and the episode's results
    look missing. Park those under ``orphaned/`` so they are visible rather than
    hidden behind a dot-directory.
    """
    try:
        base = pathlib.Path(root)
        for staging in sorted(base.glob(".staging_trial_*")):
            if not any(staging.iterdir()):
                staging.rmdir()
                continue
            dest = base / "orphaned" / staging.name.lstrip(".")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest = dest.with_name(f"{dest.name}_{int(time.time())}")
            staging.rename(dest)
            print(
                f"[real] rescued artifacts from an interrupted trial -> {dest}",
                flush=True,
            )
        # Leftovers from an interrupted migration. If the real attempt dir was
        # already written, fold any remaining files into it instead of leaving a
        # confusing ".migrating" directory behind.
        for stray in base.glob("episode_*/attempt_*.migrating"):
            target = stray.with_name(stray.name.replace(".migrating", ""))
            if not target.exists():
                stray.rename(target)
                continue
            for item in list(stray.rglob("*")):
                if not item.is_file():
                    continue
                rel = item.relative_to(stray)
                out = target / rel
                if out.exists():
                    # Same name, different content (trial.py rewrites
                    # all_responses.json as a trial progresses). Keep both --
                    # deleting the older copy would silently discard data.
                    out = target / f"{rel.name}.pre_migration"
                    if out.exists():
                        continue
                out.parent.mkdir(parents=True, exist_ok=True)
                item.rename(out)
            for d in sorted(
                (d for d in stray.rglob("*") if d.is_dir()), reverse=True
            ):
                try:
                    d.rmdir()
                except OSError:
                    pass
            try:
                stray.rmdir()
            except OSError:
                print(f"[real] left non-empty {stray}", flush=True)
    except Exception:
        pass


def _existing_attempts(root: str) -> dict[Any, int]:
    """episode -> highest attempt already present under *root*."""
    out: dict[Any, int] = {}
    try:
        for ep_dir in pathlib.Path(root).glob("episode_*"):
            if not ep_dir.is_dir():
                continue
            label = ep_dir.name[len("episode_") :]
            try:
                key: Any = int(label)
            except ValueError:
                key = label
            best = 0
            for att in ep_dir.glob("attempt_*"):
                try:
                    best = max(best, int(att.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            if best:
                out[key] = best
    except Exception:
        pass
    return out


def _relocate(staging: pathlib.Path, dest: pathlib.Path) -> None:
    """Move a finished trial's artifacts into its episode directory.

    ``trial.py`` does NOT write into ``config["output_dir"]`` directly: it writes
    into a ``trial_NN_sandboxrc_X_reward_Y_taskcompleted_Z`` SUBDIR of it
    (``_trial_video_dir``, trial.py:112). Renaming the staging dir verbatim
    therefore produced ``attempt_1/trial_01_.../<files>`` and ``attempt_1``
    looked empty. Flatten that redundant level, keeping the trial dir's name in
    ``trial_dir_name.txt`` since it encodes the sandbox return code.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():                      # never clobber a prior attempt
            dest = dest.with_name(dest.name + f"_{int(time.time())}")

        inner = [
            d
            for d in staging.iterdir()
            if d.is_dir() and d.name.startswith("trial_")
        ]
        if inner:
            # A retry (MAX_TRIAL_RETRIES) writes a SECOND trial_NN dir with a
            # different sandbox rc. The newest is the final outcome; keep it as
            # the attempt and preserve the earlier ones beside it.
            inner.sort(key=lambda d: d.stat().st_mtime)
            trial_dir = inner[-1]
            superseded = inner[:-1]
            trial_dir.rename(dest)
            for i, old_try in enumerate(superseded, start=1):
                try:
                    old_try.rename(dest / f"superseded_try_{i}_{old_try.name}")
                except Exception:
                    pass
            # Anything trial.py wrote at the output_dir level (initial_prompt.txt,
            # the rewritten all_responses.json) belongs with the attempt too.
            for leftover in staging.iterdir():
                try:
                    target = dest / leftover.name
                    if not target.exists():
                        leftover.rename(target)
                except Exception:
                    pass
            try:
                (dest / "trial_dir_name.txt").write_text(trial_dir.name + "\n")
            except Exception:
                pass
            # trial.py rewrites all_responses.json at the output_dir level, so a
            # file can reappear after the fold above and leave the staging dir
            # non-empty. Sweep again, keeping differing copies rather than
            # discarding them.
            for leftover in list(staging.iterdir()):
                try:
                    target = dest / leftover.name
                    if target.exists():
                        target = dest / f"{leftover.name}.staged"
                        if target.exists():
                            leftover.unlink()
                            continue
                    leftover.rename(target)
                except Exception:
                    pass
            try:
                staging.rmdir()
            except OSError as exc:
                print(f"[real] staging not empty, left in place: {exc}", flush=True)
        else:
            staging.rename(dest)
    except Exception as exc:
        print(
            f"[real] could not file {staging} under {dest}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def _peek_episode(env) -> Any:
    """Live ``episode_id`` if a driver frame has arrived, else None."""
    for attr in ("_wire_episode", "_announced_episode"):
        val = getattr(env, attr, None)
        if val is not None:
            return val
    try:
        low = getattr(env, "low_level_env", None)
        return getattr(low, "_ep", None).episode_seen  # type: ignore[union-attr]
    except Exception:
        return None


def _append_ledger(path: str, row: dict[str, Any]) -> None:
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        pass
