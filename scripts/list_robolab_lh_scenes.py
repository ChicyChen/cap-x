"""List robolab long-horizon scenes registered with gym.

Run this after the robolab venv is set up to discover valid scene
names to drop into the YAML config's ``scene_name`` field.

Usage:
    python scripts/list_robolab_lh_scenes.py [--filter common_sense]
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter",
        default="",
        help="Substring filter on scene id (e.g. 'common_sense' or 'Sort').",
    )
    args = parser.parse_args()

    try:
        import gymnasium as gym
        from robolab.policies.droid_jointpos.auto_env_registrations import (
            auto_register_droid_envs,
        )
    except ImportError as e:
        print(f"robolab import failed: {e}", file=sys.stderr)
        print(
            "Install robolab into this venv (pip install -e /path/to/robolab) and retry.",
            file=sys.stderr,
        )
        return 1

    auto_register_droid_envs()

    needle = args.filter.lower()
    matches = sorted(
        scene_id
        for scene_id in gym.envs.registry.keys()
        if not needle or needle in scene_id.lower()
    )
    for scene_id in matches:
        print(scene_id)
    print(f"\n{len(matches)} scene(s) listed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
