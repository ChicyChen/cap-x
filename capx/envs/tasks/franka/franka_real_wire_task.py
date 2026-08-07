"""Real-robot code env whose task comes from the wire, not from YAML.

Why this file exists
--------------------
``franky_service`` sends the instruction on **every** observation frame
(``prompt``), and marks episodes with ``episode_id``. But
``CodeExecutionEnvBase.__init__`` builds ``_full_prompt`` **once** from the YAML
config, so a launch-time task string is all the agent would ever see.

Rather than change ``CodeExecutionEnvBase`` -- which every simulator env in this
repo shares, and which the published sim results depend on -- this subclass
overrides two small hooks. **Nothing on the sim path is touched:** simulator
configs keep using their existing code-env classes, so their behaviour is
bit-identical.

What it overrides
-----------------
``reset()``
    Blocks until the driver has sent a frame, adopts that frame's ``prompt`` as
    the task, and rebuilds ``_full_prompt`` before the agent sees anything.

``_get_observation()``
    Re-checks the live ``prompt`` / ``episode_id`` each time the agent looks at
    the world, and records when they change so the caller can end the episode.

Success is NOT inferred here: the reference system's real-robot logs carry no
success field and trials are scored manually by the operator. Adding automatic
success detection for a baseline that the comparison system does not have would
bias the comparison.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from capx.envs.simulators.franky_ws import EpisodeSuperseded
from capx.envs.tasks.base import CodeExecutionEnvBase

logger = logging.getLogger(__name__)

#: How long to wait for the driver's first frame before giving up. The operator
#: usually starts the driver after the agent, so this is generous.
FIRST_FRAME_TIMEOUT_S = 600.0

_FALLBACK_PROMPT = (
    "You are controlling a Franka Emika robot with the API described below.\n"
    "You may write Python comments for reasoning but ONLY write executable "
    "Python code and do not use code fences.\n"
    "After this code executes you will get a new observation and may write "
    "new code to continue the task.\n"
    "The functions (APIs) below are already imported. Import numpy explicitly "
    "if you need it."
)


class FrankaRealWireTaskCodeEnv(CodeExecutionEnvBase):
    """Code env that takes its task from the driver's per-frame ``prompt``."""

    #: Used only until the first frame arrives; immediately replaced.
    prompt = _FALLBACK_PROMPT

    # -- helpers -------------------------------------------------------
    def _live_obs(self) -> dict:
        """Latest observation, without blocking.

        Ask the low-level env for its current frame. Do NOT reach for a
        ``.obs`` attribute: that existed on the older ``franka_real`` backend but
        not on ``FrankyWsLowLevel``, and silently returning ``{}`` made the task
        never arrive (the run just sat in ``_wait_for_first_frame``).
        """
        env = self.low_level_env
        peek = getattr(env, "peek_observation", None)
        if callable(peek):
            got = peek()
            return got if isinstance(got, dict) else {}
        obs = getattr(env, "obs", None)
        return obs if isinstance(obs, dict) else {}

    def _wire_prompt(self) -> str | None:
        p = self._live_obs().get("wire_prompt")
        return p if isinstance(p, str) and p.strip() else None

    def _wire_episode_id(self) -> Any:
        return self._live_obs().get("wire_episode_id")

    def _adopt_wire_task(self, instruction: str) -> None:
        """Rebuild the prompt around *instruction*.

        Mirrors ``__init__``'s construction so the agent sees exactly the same
        shape it would from a YAML task, with the API docs re-rendered (they
        depend on the live API objects).
        """
        if instruction == getattr(self, "_wire_task", None):
            return
        self._wire_task = instruction
        base = self.__class__.prompt if self.__class__.prompt else _FALLBACK_PROMPT
        self._task_prompt = f"{base}\nGoal: {instruction}"
        self._full_prompt = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._get_complete_prompt()}
                ],
            },
        ]
        logger.info("[real] task adopted from the wire: %r", instruction)
        print(f"[real] task from driver: {instruction!r}", flush=True)

    def _announce_episode_to_monitor(self, instruction: str) -> None:
        """Open a fresh dashboard episode and attach its task. Never raises."""
        try:
            from capx.monitor import hooks as _h

            ep = getattr(self, "_wire_episode", None)
            _h.episode_start(ep, task=instruction)
        except Exception:
            pass

    def _record_attempt(self, instruction: str, attempt: int) -> None:
        """Append this plan attempt to ``episodes.jsonl``. Never raises.

        The on-disk ``trial_NN_...`` directories are numbered by CaP-X TRIAL,
        which is one plan attempt -- they carry no episode id, so results cannot
        otherwise be mapped back to the operator's episodes. This ledger is the
        join key: (episode_id, attempt) -> trial number + output dir.
        """
        try:
            import json
            import os
            import time as _t

            cfg = getattr(self, "cfg", None)
            out = (
                cfg.get("output_dir")
                if isinstance(cfg, dict)
                else getattr(cfg, "output_dir", None)
            )
            if not out:
                return
            os.makedirs(out, exist_ok=True)
            row = {
                "ts": _t.time(),
                "iso": _t.strftime("%Y-%m-%d %H:%M:%S"),
                "episode_id": self._wire_episode,
                "attempt": attempt,
                "trial": getattr(self, "_trial_index", None),
                "task": instruction,
                "output_dir": out,
            }
            with open(os.path.join(out, "episodes.jsonl"), "a") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception:
            pass

    def _announce_attempt_to_monitor(self, attempt: int) -> None:
        """Report a re-plan within the SAME episode. Never raises."""
        try:
            from capx.monitor import hooks as _h

            _h.attempt(getattr(self, "_wire_episode", None), attempt)
        except Exception:
            pass

    def _wait_for_first_frame(self, timeout_s: float) -> None:
        """Block until the driver is ACTIVELY streaming, not merely connected.

        ``_wire_prompt()`` keeps returning the last frame after the operator
        stops an episode, so testing it alone let a new trial re-plan instantly
        against a dead episode. Require a recent frame as well.
        """
        deadline = time.time() + timeout_s
        announced = False
        while time.time() < deadline:
            live = True
            try:
                live = self.low_level_env.frames_are_live()
            except Exception:
                live = True          # older/other envs: keep the old behaviour
            if self._wire_prompt() is not None and live:
                return
            if not announced:
                print(
                    "[real] waiting for the driver to stream observations "
                    "(a fresh frame supplies the task instruction)...",
                    flush=True,
                )
                announced = True
            time.sleep(0.2)
        raise TimeoutError(
            f"no observation with a 'prompt' arrived within {timeout_s:.0f}s. "
            "Is the robot driver connected to the bridge port?"
        )

    # -- overrides -----------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        # low_level_env.reset() already blocks for the first observation; wait
        # again here because we additionally need the *prompt* to be present.
        # trial.py passes options={"trial": N}; keep it so the ledger can join
        # an episode to its trial_NN_... directory.
        if isinstance(options, dict) and "trial" in options:
            self._trial_index = options["trial"]
        obs, info = super().reset(seed=seed, options=options)
        self._wait_for_first_frame(FIRST_FRAME_TIMEOUT_S)
        instruction = self._wire_prompt()
        if instruction is None:  # pragma: no cover - guarded by the wait
            raise RuntimeError("first frame carried no 'prompt'")
        # Set the episode id BEFORE adopting the task: _adopt_wire_task
        # publishes a monitor event tagged with self._wire_episode, so doing it
        # after would file the task under the wrong (or no) episode -- which is
        # exactly the "episode 17 shows a task but no start" symptom.
        # A CaP-X trial is one PLAN ATTEMPT, not an operator episode: when a
        # plan finishes, the runner immediately starts another trial on the SAME
        # episode. So compare against the previous episode id and only open a
        # new dashboard episode when the operator actually advanced it.
        # Compare against the episode of the last ANNOUNCED plan, not
        # _wire_episode: super().reset() above calls _get_observation(), which
        # already advances _wire_episode when the operator moved on, so the two
        # would always look equal here.
        previous_episode = getattr(self, "_announced_episode", None)
        self._wire_episode = self._wire_episode_id()
        is_new_episode = (
            previous_episode is None or self._wire_episode != previous_episode
        )
        self._announced_episode = self._wire_episode
        self._adopt_wire_task(instruction)
        # Announce here, NOT inside _adopt_wire_task: that early-returns when
        # the instruction is unchanged, and the env object is reused across
        # trials, so re-running the SAME task left the episode with no task.
        if is_new_episode:
            self._announce_episode_to_monitor(instruction)
            self._episode_attempt = 1
            self._record_attempt(instruction, 1)
        else:
            # Same episode, fresh plan: report an attempt, not a new episode.
            self._episode_attempt = getattr(self, "_episode_attempt", 1) + 1
            self._announce_attempt_to_monitor(self._episode_attempt)
            self._record_attempt(instruction, self._episode_attempt)
        # Rebind: from here on, tool calls belong to THIS episode.
        try:
            self.low_level_env.begin_plan()
        except Exception:
            pass
        self._wire_task_changed = False
        self._episode_boundary = False
        if is_new_episode:
            # Block numbering is per EPISODE; a re-plan continues the count so
            # the dashboard cannot show "code block 0" twice in one episode.
            self._episode_code_blocks = 0
        # Re-emit the observation so it carries the rebuilt prompt.
        obs.update(self._get_observation())
        info.update(
            {
                "task_prompt": self._task_prompt,
                "wire_task": instruction,
                "wire_episode_id": self._wire_episode,
            }
        )
        return obs, info

    def _get_observation(self) -> dict[str, Any]:
        # Adopt a changed instruction as soon as the driver sends one, so the
        # next code-gen turn is about the current task.
        live = self._wire_prompt()
        if live is not None and live != getattr(self, "_wire_task", None):
            logger.info("[real] instruction changed on the wire")
            self._adopt_wire_task(live)
            self._wire_task_changed = True
        ep = self._wire_episode_id()
        prev = getattr(self, "_wire_episode", None)
        # prev is None only before reset() has recorded the trial's starting
        # episode -- super().reset() calls _get_observation() first. Treating
        # that as a boundary ended the trial immediately, and with
        # --total-trials 1 the whole process exited and closed port 8041
        # mid-episode ("no close frame received or sent").
        if ep is not None and prev is not None and ep != prev:
            logger.info("[real] episode_id %r -> %r", prev, ep)
            print(
                f"[real] operator started a new episode ({prev} -> {ep}); "
                "ending this trial so the next one plans from the new scene",
                flush=True,
            )
            self._wire_episode = ep
            self._wire_task_changed = True
            self._episode_boundary = True
        obs = super()._get_observation()
        obs["wire_task_changed"] = getattr(self, "_wire_task_changed", False)
        obs["wire_episode_boundary"] = getattr(self, "_episode_boundary", False)
        return obs

    def step(self, code: str):
        """Run one code-gen turn, then end the trial on an episode boundary.

        The operator's stop/start is the source of truth: the driver increments
        ``episode_id`` per episode ("so the server can detect boundaries" --
        franky_service episode_runner.py). Without this the trial would keep
        executing the previous episode's plan against a rearranged scene.

        Signalled via ``truncated`` (not ``terminated``) because a boundary is
        not task success -- success on hardware is scored by the operator.
        """
        try:
            from capx.monitor import hooks as _h

            # Count within THIS episode. base._step_count is cumulative for the
            # trial and only zeroed by base.reset(), so using it made a new
            # episode's first block appear as "block 2" whenever the previous
            # plan had already run blocks.
            self._episode_code_blocks = (
                getattr(self, "_episode_code_blocks", 0) + 1
            )
            _h.code_block(code, index=self._episode_code_blocks - 1)
        except Exception:
            pass
        # Bind this code block to the current episode. Any tool call it makes
        # after the operator starts a new episode raises EpisodeSuperseded, so
        # the old plan stops instead of racing the new one.
        #
        # If the episode has ENDED, begin_plan raises and we must NOT run the
        # block at all. Catching that here (rather than letting it escape to
        # runner.py, which would kill the process) ends the trial cleanly with
        # the phrase trial.py:805 looks for.
        try:
            self.low_level_env.begin_plan()
        except EpisodeSuperseded as exc:
            print(f"[real] not running this code block: {exc}", flush=True)
            self._episode_boundary = True
            obs = self._get_observation()
            info = {
                "sandbox_rc": 0,
                "stdout": "",
                "stderr": f"{exc}",
                "wire_episode_boundary": True,
                "wire_episode_id": getattr(self, "_wire_episode", None),
            }
            return obs, 0.0, False, True, info
        except Exception:
            pass
        obs, reward, terminated, truncated, info = super().step(code)
        try:
            from capx.monitor import hooks as _h

            err = (info or {}).get("stderr") or ""
            if err.strip():
                _h.error(err.strip()[:2000])
        except Exception:
            pass
        if getattr(self, "_episode_boundary", False):
            truncated = True
            info = dict(info)
            info["wire_episode_boundary"] = True
            info["wire_episode_id"] = getattr(self, "_wire_episode", None)
            self._episode_boundary = False
            # trial.py does NOT break its multi-turn loop on `truncated`; it
            # breaks only when stderr contains "terminated episode"
            # (trial.py:805, the convention robolab.py:500 uses). Without this
            # the boundary was detected and then blocks 6,7,8,9 of the void plan
            # kept running -- which is why a new episode's first block was
            # reported as "block 2" instead of "block 0".
            info["stderr"] = (
                (info.get("stderr") or "")
                + "\n[real] operator started a new episode; terminated episode."
            )
        return obs, reward, terminated, truncated, info

    def task_completed(self) -> bool:
        """No ground truth on hardware.

        The comparison system's real-robot logs contain no success field and
        trials are scored manually. Inventing a success signal here would bias
        any comparison, so always report False and let the operator score.
        """
        return False


__all__ = ["FrankaRealWireTaskCodeEnv"]
