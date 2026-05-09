"""Cheap import + reset + step smoke test for FrankaRobolabEnv.

Verifies the adapter loads inside Isaac Sim, the env constructs,
reset+step+_get_object_pose work, and gt_state.jsonl is being written.

Does NOT exercise the LLM agent loop, pyroki server, or full trial
runner — those need separate infrastructure.

Usage:
    conda activate capx-robolab
    python scripts/smoke_test_robolab_env.py [--scene SortFoodVsNonFoodTaskHomeOffice]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback


def _say(msg: str = "") -> None:
    """Isaac Sim swallows stdout. Print to stderr with flush so we always see progress."""
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        default=None,
        help="Robolab scene id. If omitted, the first long_horizon scene found is used.",
    )
    parser.add_argument("--n-steps", type=int, default=4, help="Steps to take after reset.")
    args = parser.parse_args()

    _say("[smoke] launching IsaacLab AppLauncher (headless)...")
    try:
        from isaaclab.app import AppLauncher
        # Robolab scenes have cameras; Isaac Sim refuses to spawn them
        # without --enable_cameras.
        app_launcher = AppLauncher(headless=True, enable_cameras=True)
        simulation_app = app_launcher.app  # noqa: F841
    except Exception:
        _say("[smoke] AppLauncher failed:")
        traceback.print_exc()
        return 1

    try:
        _say("[smoke] importing capx adapter...")
        from capx.envs.simulators.robolab import FrankaRobolabTask

        # Discover a scene if not provided
        scene = args.scene
        if scene is None:
            try:
                import gymnasium as gym
                from robolab.policies.droid_jointpos.auto_env_registrations import (
                    auto_register_droid_envs,
                )
                # LH common-sense tasks aren't in the default subdirs; ask
                # auto-register to walk that subfolder explicitly.
                auto_register_droid_envs(task_dirs=["long_horizon/common_sense"])
                candidates = sorted(
                    s
                    for s in gym.envs.registry.keys()
                    if "Sort" in s or "Kit" in s or "Recover" in s or "Infer" in s
                )
                if not candidates:
                    _say("[smoke] no LH scenes registered after auto_register_droid_envs()")
                    return 1
                scene = candidates[0]
                _say(f"[smoke] auto-selected scene: {scene}  (out of {len(candidates)})")
            except Exception:
                _say("[smoke] scene discovery failed:")
                traceback.print_exc()
                return 1

        with tempfile.TemporaryDirectory() as tmp:
            dump_path = os.path.join(tmp, "gt_state.jsonl")
            _say(f"[smoke] constructing FrankaRobolabTask(scene_name='{scene}')...")
            env = FrankaRobolabTask(
                scene_name=scene,
                seed=1,
                gt_state_dump_path=dump_path,
            )
            _say(
                f"[smoke] env built. instruction='{env.env_cfg.instruction}', "
                f"object_count={len(env._gt_exporter._object_names)}"
            )

            _say("[smoke] running reset() — already invoked in __init__, taking a fresh one...")
            obs, info = env.reset()
            _say(f"[smoke] reset OK. obs keys: {sorted(obs.keys())}")
            _say(f"[smoke] task_prompt: {info['task_prompt']}")

            cartesian = obs.get("robot_cartesian_pos")
            joints = obs.get("robot_joint_pos")
            _say(
                f"[smoke] robot_cartesian_pos={None if cartesian is None else cartesian.shape}, "
                f"robot_joint_pos={None if joints is None else joints.shape}"
            )

            # Try _get_object_pose for the first scene object
            object_names = env._gt_exporter._object_names
            if object_names:
                target_obj = object_names[0]
                pos, quat = env._get_object_pose(target_obj)
                _say(f"[smoke] _get_object_pose('{target_obj}') → pos={pos}, quat_wxyz={quat}")

            # Hold-and-tick: emit n_steps env ticks with current joints
            _say(f"[smoke] running {args.n_steps} hold ticks via _step_once()...")
            for i in range(args.n_steps):
                env._step_once()
            _say(
                f"[smoke] task_completed={env.task_completed()}, "
                f"sim_step_count={env._sim_step_count}"
            )

            # Verify gt_state dump produced records
            resolved = env._gt_state_dump_path_resolved
            _say(f"[smoke] gt_state dump path: {resolved}")
            env.close()
            if resolved is None or not os.path.isfile(resolved):
                _say("[smoke] FAIL: gt_state dump not written")
                return 1
            with open(resolved) as fh:
                lines = [ln for ln in fh if ln.strip()]
            _say(f"[smoke] gt_state dump contains {len(lines)} record(s)")
            if not lines:
                _say("[smoke] FAIL: gt_state dump empty")
                return 1
            first = json.loads(lines[0])
            _say(
                f"[smoke] first record keys: {sorted(first.keys())}, "
                f"objects in gt_state: {sorted((first.get('gt_state') or {}).get('objects', {}).keys())[:6]}"
            )

        _say("[smoke] PASS")
        return 0
    except Exception:
        _say("[smoke] FAIL with exception:")
        traceback.print_exc()
        return 1
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
