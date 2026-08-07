"""Launch entry point for the real robot: one directory per operator episode.

Use this instead of ``capx.envs.launch`` when driving the physical Franka via
``franky_service``:

    python -m capx.envs.real_launch --config-path env_configs/real/real_franky.yaml ...

It is a thin wrapper: same CLI (``LaunchArgs``), same config loading, same API
servers. The only difference is that trials are dispatched through
``real_episode_runner.run_real_episode_trials``, which lays artifacts out by the
driver's ``episode_id`` instead of a flat trial counter.

Kept as a separate module so ``capx/envs/launch.py`` -- shared with every sim
benchmark -- needs no changes at all.
"""

from __future__ import annotations

import time

import tyro

from capx.envs.launch import LaunchArgs, _run_web_ui
from capx.envs.real_episode_runner import run_real_episode_trials
from capx.envs.runner import _start_api_servers, _stop_api_servers
from capx.utils.launch_utils import _load_config


def main(args: LaunchArgs) -> None:
    start_time = time.time()
    env_factory, config, api_servers = _load_config(args)
    server_procs = _start_api_servers(api_servers)

    try:
        if config.get("web_ui", False):
            # The browser UI owns its own trial loop; episode layout does not
            # apply there.
            _run_web_ui(args, config)
        else:
            run_real_episode_trials(args, env_factory, config, start_time)
    finally:
        try:
            _stop_api_servers(server_procs)
        except KeyboardInterrupt:
            import sys

            sys.exit(1)

    # Same rationale as launch.py: skip atexit handlers (Isaac Sim's destructor
    # can hang for over an hour). All artifacts are already flushed.
    import os

    os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(LaunchArgs))
