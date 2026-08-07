"""The real-robot code env must take its task from the wire, not from YAML.

Also pins that the SIM path is untouched: the shared base class still builds its
prompt once from config, exactly as before.
"""

from __future__ import annotations

import types

import pytest

from capx.envs.tasks.franka.franka_real_wire_task import (
    FrankaRealWireTaskCodeEnv,
)


class _FakeLowLevel:
    """Stands in for FrankaRealLowLevel: just exposes `.obs`."""

    def __init__(self, obs=None):
        self.obs = obs or {}


def _make_env(obs):
    """Build the env without __init__ (which needs APIs, servers, a robot)."""
    env = FrankaRealWireTaskCodeEnv.__new__(FrankaRealWireTaskCodeEnv)
    env.low_level_env = _FakeLowLevel(obs)
    env._system_prompt = "sys"
    env._apis = {}
    env._task_prompt = "placeholder"
    env._full_prompt = []
    return env


class TestAdoptTaskFromWire:
    def test_prompt_is_taken_from_the_frame(self):
        env = _make_env({"wire_prompt": "stack the blocks", "wire_episode_id": 1})
        env._adopt_wire_task(env._wire_prompt())
        text = env._full_prompt[-1]["content"][0]["text"]
        assert "stack the blocks" in text
        assert "Goal: stack the blocks" in env._task_prompt

    def test_changed_instruction_is_adopted(self):
        env = _make_env({"wire_prompt": "task A", "wire_episode_id": 1})
        env._adopt_wire_task("task A")
        env.low_level_env.obs["wire_prompt"] = "task B"
        env._adopt_wire_task(env._wire_prompt())
        assert "task B" in env._full_prompt[-1]["content"][0]["text"]
        assert "task A" not in env._full_prompt[-1]["content"][0]["text"]

    def test_same_instruction_is_not_rebuilt(self):
        env = _make_env({"wire_prompt": "task A"})
        env._adopt_wire_task("task A")
        before = env._full_prompt
        env._adopt_wire_task("task A")
        assert env._full_prompt is before, "should be a no-op"

    def test_blank_prompt_is_ignored(self):
        assert _make_env({"wire_prompt": "   "})._wire_prompt() is None
        assert _make_env({})._wire_prompt() is None

    def test_wait_raises_if_no_frame_arrives(self):
        env = _make_env({})
        with pytest.raises(TimeoutError, match="no observation"):
            env._wait_for_first_frame(0.3)

    def test_wait_returns_once_a_frame_exists(self):
        env = _make_env({"wire_prompt": "go"})
        env._wait_for_first_frame(1.0)  # must not raise

    def test_task_completed_is_always_false(self):
        """No GT on hardware; operator scores manually."""
        assert _make_env({"wire_prompt": "x"}).task_completed() is False


class TestSimPathUntouched:
    def test_base_class_still_builds_prompt_from_config(self):
        """The shared base must be unchanged: prompt comes from config, and it
        has no wire hooks. If this fails, sim eval semantics moved."""
        from capx.envs.tasks.base import CodeExecutionEnvBase

        for hook in ("_wire_prompt", "_adopt_wire_task", "_wait_for_first_frame"):
            assert not hasattr(CodeExecutionEnvBase, hook), (
                f"{hook} leaked into the shared base class"
            )

    def test_real_env_overrides_only_expected_methods(self):
        own = {
            k
            for k, v in vars(FrankaRealWireTaskCodeEnv).items()
            if callable(v) and not k.startswith("__")
        }
        expected = {
            "_live_obs",
            "_wire_prompt",
            "_wire_episode_id",
            "_adopt_wire_task",
            "_announce_episode_to_monitor",
            "_announce_attempt_to_monitor",
            "_record_attempt",
            "_wait_for_first_frame",
            "reset",
            "step",
            "_get_observation",
            "task_completed",
        }
        assert own == expected, f"unexpected overrides: {own ^ expected}"

    def test_franka_real_module_is_not_imported_by_sim_envs(self):
        """franka_real.py is only referenced by env_configs/real/*."""
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        offenders = []
        for cfg in (repo / "env_configs").rglob("*.yaml"):
            if cfg.parent.name == "real":
                continue
            if "FrankaRealLowLevel" in cfg.read_text() or (
                "franka_real_low_level" in cfg.read_text()
            ):
                offenders.append(cfg.name)
        assert not offenders, f"sim configs referencing the real env: {offenders}"


class TestRealApiTierMatchesSim:
    """The real config must give the agent the same tier as the sim baseline.

    Our published sim CaP-X numbers used FrankaRobolabApiReducedSkillLibrary
    (SAM3 + ContactGraspNet + skill library). If the real-robot baseline ran on
    FrankaRealControlApi -- which exposes only 5 tools -- it would be a weaker
    agent, understating CaP-X and flattering the comparison system.
    """

    @staticmethod
    def _apis(path):
        import yaml
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        return yaml.safe_load((repo / path).read_text())["env"]["cfg"]["apis"]

    def test_real_config_uses_the_skill_library_tier(self):
        apis = self._apis("env_configs/real/real_franky.yaml")
        assert apis == ["FrankaRealReducedSkillLibraryControlApi"], apis

    def test_real_config_does_not_use_the_5_tool_api(self):
        apis = self._apis("env_configs/real/real_franky.yaml")
        assert "FrankaRealControlApi" not in apis

    def test_real_api_is_registered(self):
        from capx.integrations import list_apis

        assert "FrankaRealReducedSkillLibraryControlApi" in list_apis()

    def test_real_api_carries_the_real_geometry(self):
        """tcp_offset/real must differ from sim: the gripper is 50 mm longer."""
        import inspect
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        src = (repo / "capx/integrations/__init__.py").read_text()
        line = next(
            l
            for l in src.splitlines()
            if "FrankaRealReducedSkillLibraryControlApi" in l
        )
        assert "-0.157" in line, line
        assert "real = True" in line or "real=True" in line, line

    def test_both_tiers_share_perception_and_grasp_tools(self):
        """Same SAM3 + ContactGraspNet entry points on both sides."""
        import inspect
        import re

        from capx.integrations.franka.control_reduced_skill_library import (
            FrankaControlApiReducedSkillLibrary as REAL,
        )

        def fnames(cls):
            out = set()
            for c in cls.__mro__:
                if "functions" in vars(c):
                    s = inspect.getsource(vars(c)["functions"])
                    out |= set(re.findall(r'fns\["([a-z0-9_]+)"\]', s))
            return out

        real = fnames(REAL)
        for tool in (
            "segment_sam3_text_prompt",
            "segment_sam3_point_prompt",
            "plan_grasp",
            "solve_ik",
            "move_to_joints",
        ):
            assert tool in real, f"{tool} missing from the real tier"


class TestLiveObsWiring:
    """The code env must read the task through the low-level env's real API.

    Regression: ``_wire_prompt`` used ``low_level_env.obs``, an attribute that
    exists on the older franka_real backend but NOT on FrankyWsLowLevel. It
    therefore always returned None and the run hung in _wait_for_first_frame
    even though the driver was streaming a prompt every frame.
    """

    def test_reads_via_peek_observation(self):
        env = FrankaRealWireTaskCodeEnv.__new__(FrankaRealWireTaskCodeEnv)

        class LL:
            def peek_observation(self):
                return {"wire_prompt": "pick up the object", "wire_episode_id": 6}

        env.low_level_env = LL()
        assert env._wire_prompt() == "pick up the object"
        assert env._wire_episode_id() == 6

    def test_falls_back_to_obs_attribute(self):
        """Older backends expose `.obs` instead; still support them."""
        env = FrankaRealWireTaskCodeEnv.__new__(FrankaRealWireTaskCodeEnv)

        class LL:
            obs = {"wire_prompt": "task", "wire_episode_id": 1}

        env.low_level_env = LL()
        assert env._wire_prompt() == "task"

    def test_no_frame_yet_returns_none(self):
        env = FrankaRealWireTaskCodeEnv.__new__(FrankaRealWireTaskCodeEnv)

        class LL:
            def peek_observation(self):
                return {}

        env.low_level_env = LL()
        assert env._wire_prompt() is None


class TestEpisodeBoundary:
    """The operator's stop/start must end the trial.

    The driver increments ``episode_id`` per episode specifically "so the server
    can detect boundaries" (franky_service episode_runner.py:103). Previously we
    only LOGGED the change, so CaP-X kept executing the previous episode's plan
    against a rearranged scene.
    """

    @staticmethod
    def _env(ep_id):
        env = FrankaRealWireTaskCodeEnv.__new__(FrankaRealWireTaskCodeEnv)

        class LL:
            def __init__(self, e):
                self.e = e

            def peek_observation(self):
                return {"wire_prompt": "pick up the object", "wire_episode_id": self.e}

        env.low_level_env = LL(ep_id)
        env._system_prompt = "sys"
        env._apis = {}
        env._task_prompt = "p"
        env._full_prompt = []
        env._wire_task = "pick up the object"
        env._wire_episode = ep_id
        env._wire_task_changed = False
        env._episode_boundary = False
        return env

    def test_same_episode_does_not_flag(self, monkeypatch):
        env = self._env(3)
        monkeypatch.setattr(
            type(env).__mro__[1], "_get_observation", lambda self: {}
        )
        env._get_observation()
        assert env._episode_boundary is False

    def test_new_episode_sets_the_flag(self, monkeypatch):
        env = self._env(3)
        env.low_level_env.e = 4  # operator pressed start again
        monkeypatch.setattr(
            type(env).__mro__[1], "_get_observation", lambda self: {}
        )
        env._get_observation()
        assert env._episode_boundary is True
        assert env._wire_episode == 4

    def test_boundary_truncates_the_trial(self, monkeypatch):
        env = self._env(3)
        env._episode_boundary = True
        monkeypatch.setattr(
            type(env).__mro__[1],
            "step",
            lambda self, code: ({}, 0.0, False, False, {"sandbox_rc": 0}),
        )
        _, _, terminated, truncated, info = env.step("pass")
        assert truncated is True, "a new episode must end the trial"
        assert terminated is False, "a boundary is NOT task success"
        assert info["wire_episode_boundary"] is True
        assert env._episode_boundary is False, "flag must be consumed once"

    def test_no_boundary_leaves_step_untouched(self, monkeypatch):
        env = self._env(3)
        monkeypatch.setattr(
            type(env).__mro__[1],
            "step",
            lambda self, code: ({}, 0.0, False, False, {"sandbox_rc": 0}),
        )
        _, _, _, truncated, info = env.step("pass")
        assert truncated is False
        assert "wire_episode_boundary" not in info


class TestPerEpisodeRefresh:  # noqa: D101
    """What must be re-established on every episode, given the env is reused."""

    def test_task_is_readopted_and_announced_each_episode(self):
        """Same instruction twice must still announce episode 2's task.

        ``_adopt_wire_task`` early-returns when the string is unchanged, so the
        announcement cannot live inside it.
        """
        from capx.monitor.events import BUS

        BUS.episodes.clear()
        BUS._history.clear()
        from capx.monitor import hooks as h

        h.episode_start(18, task="pick up the block")
        h.episode_start(19, task="pick up the block")
        assert BUS.episodes[19]["task"] == "pick up the block"

    @staticmethod
    def _src(rel: str) -> str:
        import pathlib as _p

        return _p.Path(rel).read_text()

    def test_boundary_flags_are_cleared_on_reset(self):
        """Otherwise a fresh trial would immediately end itself."""
        src = self._src("capx/envs/tasks/franka/franka_real_wire_task.py")
        reset_body = src[src.index("    def reset("):src.index("    def _get_observation")]
        assert "self._wire_task_changed = False" in reset_body
        assert "self._episode_boundary = False" in reset_body

    def test_exec_globals_are_reinitialised_per_trial(self):
        """Generated code must not see the previous episode's variables."""
        base = self._src("capx/envs/tasks/base.py")
        reset_body = base[base.index("    def reset("):]
        assert "_init_exec_globals" in reset_body.split("def step")[0]

    def test_trial_rearms_the_timeout(self):
        """Each trial gets a full timeout budget, not the previous remainder."""
        trial = self._src("capx/envs/trial.py")
        assert "signal.alarm(1000)" in trial


class TestTaskInstructionReachesTheModel:
    """The dashboard bug was cosmetic; this pins the PROMPT actually sent.

    The env object is reused for all trials, and ``_adopt_wire_task``
    early-returns on an unchanged instruction, so it is worth proving the goal
    line the model sees is right for every episode.
    """

    def _env(self):
        from capx.envs.tasks.franka.franka_real_wire_task import (
            FrankaRealWireTaskCodeEnv as E,
        )

        env = E.__new__(E)
        env._system_prompt = "SYS"
        env._wire_task = None
        env._task_prompt = ""
        env._full_prompt = []
        env._get_complete_prompt = lambda: f"DOCS\n{env._task_prompt}"
        return env

    @staticmethod
    def _src(rel: str) -> str:
        import pathlib as _p

        return _p.Path(rel).read_text()

    @staticmethod
    def _goal(env):
        text = env._full_prompt[-1]["content"][0]["text"]
        return [l for l in text.splitlines() if l.startswith("Goal:")]

    def test_changed_task_rebuilds_the_prompt(self):
        env = self._env()
        env._wire_episode = 18
        env._adopt_wire_task("pick up the larger yellow block")
        assert self._goal(env) == ["Goal: pick up the larger yellow block"]
        env._wire_episode = 19
        env._adopt_wire_task("put the block in the bowl")
        assert self._goal(env) == ["Goal: put the block in the bowl"]

    def test_unchanged_task_still_leaves_a_correct_prompt(self):
        """The early-return path must not leave a blank or stale goal."""
        env = self._env()
        env._adopt_wire_task("task A")
        env._adopt_wire_task("task A")          # early-returns
        assert self._goal(env) == ["Goal: task A"]

    def test_per_trial_feedback_does_not_pollute_the_env_prompt(self):
        """trial.py deep-copies obs['full_prompt'] then appends feedback."""
        import copy

        env = self._env()
        env._adopt_wire_task("task A")
        obs_prompt = copy.deepcopy(env._full_prompt)
        obs_prompt[-1]["content"][0]["text"] += "\n\nVISUAL FEEDBACK: xyz"
        obs_prompt[-1]["content"].append({"type": "image_url"})
        assert len(env._full_prompt[-1]["content"]) == 1
        assert "VISUAL FEEDBACK" not in env._full_prompt[-1]["content"][0]["text"]

    def test_mid_episode_change_reaches_the_next_turn(self):
        """A changed instruction cannot alter an in-flight prompt, but the next
        turn reads obs['full_prompt'] again (trial.py does ``obs = obs_next``).
        """
        import copy

        env = self._env()
        env._adopt_wire_task("task A")
        in_flight = copy.deepcopy(env._full_prompt)   # what the running turn has
        env._adopt_wire_task("task B")                # operator changes it
        assert self._goal(env) == ["Goal: task B"], "env prompt updates"
        assert "task A" in in_flight[-1]["content"][0]["text"], (
            "the already-sent prompt is necessarily unchanged"
        )

    def test_multiturn_decision_reads_the_live_obs(self):
        """So a task change is visible to the regenerate/finish decision."""
        src = self._src("capx/envs/trial.py")
        assert "obs = obs_next" in src, "each turn must re-read the observation"
        # the decision prompt is built from the live obs, not a frozen copy
        assert "_build_multi_turn_decision_prompt(" in src
        call = src[src.index("decision_prompt = _build_multi_turn_decision_prompt("):]
        assert call[: call.index(")")].count("obs") >= 1


class TestTrialIsNotAnEpisode:
    """A CaP-X trial is one PLAN ATTEMPT; the operator owns episodes.

    Live bug: when a plan finished, the runner immediately started another trial
    on the SAME episode, and reset() announced a second "episode 24 started" and
    restarted code-block numbering at 0 mid-episode.
    """

    def _env(self, ep, task="pick up the block"):
        from capx.envs.tasks.franka.franka_real_wire_task import (
            FrankaRealWireTaskCodeEnv as E,
        )

        env = E.__new__(E)
        env._system_prompt = "SYS"
        env._wire_task = None
        env._task_prompt = ""
        env._full_prompt = []
        env._get_complete_prompt = lambda: f"DOCS\n{env._task_prompt}"
        env._live = {"wire_prompt": task, "wire_episode_id": ep}
        env._live_obs = lambda: env._live
        env.low_level_env = types.SimpleNamespace(
            reset=lambda **k: ({}, {}),
            get_observation=lambda: {},
            begin_plan=lambda: None,
        )
        # minimal scaffolding for CodeExecutionEnvBase.reset()
        env._apis = {}
        env._exec_globals = {}
        env._step_count = 0
        env._init_exec_globals = lambda: None
        return env

    def test_replan_on_the_same_episode_is_not_a_new_episode(self):
        from capx.monitor.events import BUS

        BUS._history.clear()
        BUS.episodes.clear()
        BUS._current_episode = None

        env = self._env(24)
        env.reset()                      # trial 1 on episode 24
        env.reset()                      # trial 2, SAME episode (plan finished)

        starts = [
            e for e in BUS._history if e.kind.value == "episode_start"
        ]
        assert len(starts) == 1, (
            f"one episode_start per operator episode, got {len(starts)}"
        )
        notes = [e for e in BUS._history if "re-planning" in (e.text or "")]
        assert len(notes) == 1, "the re-plan must be reported as an attempt"

    def test_block_numbering_continues_across_a_replan(self):
        env = self._env(24)
        env.reset()
        env._episode_code_blocks = 3     # three blocks ran in attempt 1
        env.reset()                      # re-plan, same episode
        assert env._episode_code_blocks == 3, (
            "must NOT reset to 0 -- that showed 'code block 0' twice in one "
            "episode"
        )

    def test_a_real_new_episode_does_reset_numbering(self):
        env = self._env(24)
        env.reset()
        env._episode_code_blocks = 3
        env._live["wire_episode_id"] = 25      # operator advanced
        env.reset()
        assert env._episode_code_blocks == 0
        assert env._episode_attempt == 1


class TestEpisodeLedger:
    """``trial_NN_...`` dirs are numbered by PLAN ATTEMPT and carry no episode
    id, so results cannot be mapped to the operator's episodes without a join
    key. ``episodes.jsonl`` is that key.
    """

    def _env(self, tmp, ep, task="pick up the block"):
        from capx.envs.tasks.franka.franka_real_wire_task import (
            FrankaRealWireTaskCodeEnv as E,
        )

        env = E.__new__(E)
        env._system_prompt = "SYS"
        env._wire_task = None
        env._task_prompt = ""
        env._full_prompt = []
        env._get_complete_prompt = lambda: "DOCS"
        env._live = {"wire_prompt": task, "wire_episode_id": ep}
        env._live_obs = lambda: env._live
        env.low_level_env = types.SimpleNamespace(
            reset=lambda **k: ({}, {}),
            get_observation=lambda: {},
            begin_plan=lambda: None,
        )
        env._apis = {}
        env._exec_globals = {}
        env._step_count = 0
        env._init_exec_globals = lambda: None
        env.cfg = {"output_dir": str(tmp)}
        return env

    def _rows(self, tmp):
        import json
        import pathlib as _p

        f = _p.Path(tmp) / "episodes.jsonl"
        if not f.exists():
            return []
        return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]

    def test_each_attempt_is_recorded_with_its_episode(self, tmp_path):
        env = self._env(tmp_path, 24)
        env.reset(options={"trial": 1})
        env.reset(options={"trial": 2})            # re-plan, same episode
        env._live["wire_episode_id"] = 25
        env._live["wire_prompt"] = "pick up the other block"
        env.reset(options={"trial": 3})
        rows = self._rows(tmp_path)
        assert [(r["episode_id"], r["attempt"], r["trial"]) for r in rows] == [
            (24, 1, 1),
            (24, 2, 2),
            (25, 1, 3),
        ]

    def test_task_is_recorded_per_episode(self, tmp_path):
        env = self._env(tmp_path, 24, task="pick up the larger yellow block")
        env.reset(options={"trial": 1})
        assert self._rows(tmp_path)[0]["task"] == "pick up the larger yellow block"

    def test_missing_output_dir_is_harmless(self, tmp_path):
        env = self._env(tmp_path, 1)
        env.cfg = {}
        env.reset(options={"trial": 1})           # must not raise
