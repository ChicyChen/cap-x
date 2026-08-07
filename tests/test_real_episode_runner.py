"""Episode-shaped output layout for the real robot.

The operator owns episodes; CaP-X owns trials (one plan attempt each) and
``pass_N`` (one process). Those never lined up, so artifacts for two episodes
landed side by side as consecutive trials in one flat directory.

This runner redirects ``config["output_dir"]`` per episode. It must do so
WITHOUT modifying any shared CaP-X module, because the sim/robolab path uses
them.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from capx.envs.real_episode_runner import (
    _append_ledger,
    _episode_dir,
    _write_episode_json,
)


class TestEpisodeDirNaming:
    def test_ints_are_zero_padded(self):
        assert _episode_dir("/r", 24).name == "episode_024"
        assert _episode_dir("/r", 7).name == "episode_007"
        assert _episode_dir("/r", 1234).name == "episode_1234"

    def test_none_is_kept_out_of_the_episode_namespace(self):
        """A trial that never got a driver frame is not an episode at all.

        It must not sit beside real episodes as "episode_pending", and note that
        episode_id 0 IS valid -- so the check is `is None`, not falsiness.
        """
        assert _episode_dir("/r", None).name == "no_episode_trials"
        assert _episode_dir("/r", 0).name == "episode_000"

    def test_non_numeric_is_sanitised(self):
        assert _episode_dir("/r", "a/b c").name == "episode_a_b_c"


class TestEpisodeManifest:
    def test_manifest_records_task_and_attempts(self, tmp_path):
        _write_episode_json(
            tmp_path, episode=24, task="pick up the block", attempts=3,
            started=1.0,
        )
        data = json.loads((tmp_path / "episode.json").read_text())
        assert data["episode_id"] == 24
        assert data["task"] == "pick up the block"
        assert data["attempts"] == 3

    def test_manifest_warns_that_success_is_manual(self, tmp_path):
        """No GT success exists on hardware; the file must say so."""
        _write_episode_json(
            tmp_path, episode=1, task="t", attempts=1, started=1.0
        )
        note = json.loads((tmp_path / "episode.json").read_text())["note"]
        assert "MANUALLY" in note
        assert "must not be aggregated" in note

    def test_unwritable_path_is_harmless(self):
        _write_episode_json(
            pathlib.Path("/proc/nope/nope"), episode=1, task=None,
            attempts=1, started=1.0,
        )   # must not raise


class TestLedger:
    def test_rows_append(self, tmp_path):
        p = str(tmp_path / "episodes.jsonl")
        _append_ledger(p, {"episode_id": 24, "attempt": 1})
        _append_ledger(p, {"episode_id": 24, "attempt": 2})
        rows = [
            json.loads(l)
            for l in pathlib.Path(p).read_text().splitlines()
            if l.strip()
        ]
        assert [r["attempt"] for r in rows] == [1, 2]

    def test_unwritable_ledger_is_harmless(self):
        _append_ledger("/proc/nope/x.jsonl", {"a": 1})   # must not raise


class TestSharedCapxUntouched:
    """The whole point of a separate runner: sim code must not change."""

    @pytest.mark.parametrize(
        "rel",
        [
            "capx/envs/launch.py",
            "capx/envs/runner.py",
            "capx/envs/trial.py",
            "capx/utils/launch_utils.py",
            "capx/envs/tasks/base.py",
        ],
    )
    def test_no_real_robot_references_leaked_into_shared_files(self, rel):
        src = pathlib.Path(rel).read_text()
        for token in ("real_episode_runner", "episode_pending", "franky"):
            assert token not in src, (
                f"{rel} must stay generic; found {token!r}"
            )

    def test_runner_reuses_capx_helpers_rather_than_forking_them(self):
        src = pathlib.Path("capx/envs/real_episode_runner.py").read_text()
        assert "_run_trial_with_retries" in src, "reuse the retry/timeout logic"
        assert "_print_and_save_summary" in src
        assert "_setup_output_dir" in src


class TestOutputRedirection:
    def test_output_dir_is_restored_after_each_trial(self):
        """A crash mid-trial must not leave config pointing at an attempt dir."""
        src = pathlib.Path("capx/envs/real_episode_runner.py").read_text()
        body = src[src.index("for trial in range("):]
        assert "finally:" in body
        assert 'config["output_dir"] = root' in body


class TestNoPendingDirectory:
    """Operator report: the first episode of every run landed in
    ``episode_pending``.

    Cause: the directory was chosen BEFORE ``reset()``, but the episode id only
    arrives with the driver's first frame, which happens INSIDE ``reset()``.
    Trials therefore run in a staging dir and are filed afterwards.
    """

    def _run(self, tmp, episodes):
        """Drive the real runner with a fake trial fn; return the layout."""
        import types

        import capx.envs.real_episode_runner as R

        class Env:
            def __init__(self):
                self._wire_episode = None      # nothing known before reset()
                self._wire_task = None

        env = Env()
        seq = list(episodes)

        def fake(env_, trial, args, config, mtp):
            """Write artifacts EXACTLY where trial.py writes them.

            trial.py does not use config["output_dir"] directly: artifacts go to
            ``<output_dir>/trial_NN_sandboxrc_X_reward_Y_taskcompleted_Z/``
            (``_trial_video_dir``), with initial_prompt.txt / all_responses.json
            at the output_dir level. An earlier version of this mock wrote
            straight into output_dir, so it passed while the real runner buried
            everything one level too deep and ``attempt_1`` looked EMPTY.
            """
            ep, task = seq[trial - 1]
            env_._wire_episode = ep
            env_._wire_task = task
            out = pathlib.Path(config["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "initial_prompt.txt").write_text("Goal: ...")
            (out / "all_responses.json").write_text("[]")
            d = out / (
                f"trial_{trial:02d}_sandboxrc_0_reward_0.000_taskcompleted_0"
            )
            d.mkdir(parents=True, exist_ok=True)
            (d / "code.py").write_text(f"# trial {trial}")
            (d / "summary.txt").write_text("s")
            (d / "prompts_and_responses").mkdir(exist_ok=True)
            return types.SimpleNamespace(trial=trial)

        orig = (
            R._run_trial_with_retries,
            R._setup_output_dir,
            R._print_and_save_summary,
            R.instantiate,
        )
        R._run_trial_with_retries = fake
        R._setup_output_dir = lambda a, c: None
        R._print_and_save_summary = lambda *a, **k: None
        R.instantiate = lambda f: env
        try:
            R.run_real_episode_trials(
                types.SimpleNamespace(model="m"),
                {"cfg": {}},
                {"total_trials": len(seq), "output_dir": str(tmp), "num_workers": 1},
                0.0,
            )
        finally:
            (
                R._run_trial_with_retries,
                R._setup_output_dir,
                R._print_and_save_summary,
                R.instantiate,
            ) = orig
        return sorted(p.name for p in pathlib.Path(tmp).iterdir() if p.is_dir())

    def test_first_episode_is_not_pending(self, tmp_path):
        dirs = self._run(tmp_path, [(27, "pick up the block")])
        assert dirs == ["episode_027"], f"got {dirs}"

    def test_no_staging_dirs_are_left_behind(self, tmp_path):
        self._run(tmp_path, [(27, "t"), (27, "t"), (28, "u")])
        leftovers = list(pathlib.Path(tmp_path).glob(".staging*"))
        assert not leftovers, f"staging not cleaned: {leftovers}"

    def test_attempts_group_under_their_episode(self, tmp_path):
        self._run(tmp_path, [(27, "t"), (27, "t"), (28, "u")])
        assert sorted(
            p.name for p in (pathlib.Path(tmp_path) / "episode_027").iterdir()
        ) == ["attempt_1", "attempt_2", "episode.json"]
        assert (pathlib.Path(tmp_path) / "episode_028/attempt_1/code.py").exists()

    def test_numbering_resumes_after_a_process_restart(self, tmp_path):
        """The launcher reuses the same root, so attempt_1 must not collide."""
        from capx.envs.real_episode_runner import _existing_attempts

        self._run(tmp_path, [(27, "t"), (27, "t")])
        assert _existing_attempts(str(tmp_path)) == {27: 2}
        self._run(tmp_path, [(27, "t")])           # simulated restart
        assert (pathlib.Path(tmp_path) / "episode_027/attempt_3").is_dir()


class TestArtifactsAreDirectlyInsideAttempt:
    """Operator report: the saved outputs "become empty".

    ``trial.py`` writes into ``<output_dir>/trial_NN_sandboxrc_.../``, not into
    ``output_dir`` itself, so renaming the staging dir verbatim produced
    ``attempt_1/trial_01_.../code.py`` -- and ``attempt_1`` looked empty. The
    redundant level must be flattened.
    """

    def _layout(self, tmp_path):
        return TestNoPendingDirectory()._run(tmp_path, [(27, "t"), (27, "t")])

    def test_code_py_is_directly_under_attempt(self, tmp_path):
        self._layout(tmp_path)
        p = pathlib.Path(tmp_path) / "episode_027/attempt_1/code.py"
        assert p.exists(), (
            "code.py must be directly inside attempt_1, not nested in a "
            "trial_NN_... subdir"
        )

    def test_no_trial_subdir_remains(self, tmp_path):
        self._layout(tmp_path)
        att = pathlib.Path(tmp_path) / "episode_027/attempt_1"
        nested = [
            d for d in att.iterdir() if d.is_dir() and d.name.startswith("trial_")
        ]
        assert not nested, f"redundant level not flattened: {nested}"

    def test_output_dir_level_files_are_kept(self, tmp_path):
        """initial_prompt.txt lives at the output_dir level; do not lose it."""
        self._layout(tmp_path)
        att = pathlib.Path(tmp_path) / "episode_027/attempt_1"
        assert (att / "initial_prompt.txt").exists()
        assert (att / "summary.txt").exists()

    def test_trial_dir_name_is_preserved(self, tmp_path):
        """It encodes the sandbox return code, which is worth keeping."""
        self._layout(tmp_path)
        note = (
            pathlib.Path(tmp_path) / "episode_027/attempt_1/trial_dir_name.txt"
        )
        assert note.exists()
        assert note.read_text().startswith("trial_01_sandboxrc_")

    def test_staging_is_removed(self, tmp_path):
        self._layout(tmp_path)
        assert not list(pathlib.Path(tmp_path).glob(".staging*"))


class TestRetriedTrials:
    """``_run_trial_with_retries`` can write several ``trial_NN_...`` dirs in one
    output_dir (one per retry, each with its own sandbox rc). The newest is the
    real outcome; earlier ones must be preserved, not left nesting.
    """

    def _stage(self, tmp_path, names):
        import time as _t

        staging = pathlib.Path(tmp_path) / ".staging_trial_001"
        staging.mkdir(parents=True)
        (staging / "initial_prompt.txt").write_text("Goal: x")
        for i, n in enumerate(names):
            d = staging / n
            d.mkdir()
            (d / "code.py").write_text(n)
            _t.sleep(0.01)                     # distinct mtimes
        return staging

    def test_newest_retry_becomes_the_attempt(self, tmp_path):
        from capx.envs.real_episode_runner import _relocate

        staging = self._stage(
            tmp_path,
            [
                "trial_01_sandboxrc_1_reward_0.000_taskcompleted_0",
                "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0",
            ],
        )
        dest = pathlib.Path(tmp_path) / "episode_028/attempt_1"
        _relocate(staging, dest)
        assert (dest / "code.py").read_text().endswith("sandboxrc_0_reward_0.000_taskcompleted_0")
        assert not [d for d in dest.iterdir() if d.is_dir() and d.name.startswith("trial_")]

    def test_earlier_retry_is_preserved(self, tmp_path):
        from capx.envs.real_episode_runner import _relocate

        staging = self._stage(
            tmp_path,
            [
                "trial_01_sandboxrc_1_reward_0.000_taskcompleted_0",
                "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0",
            ],
        )
        dest = pathlib.Path(tmp_path) / "episode_028/attempt_1"
        _relocate(staging, dest)
        kept = [d.name for d in dest.iterdir() if d.name.startswith("superseded_try_")]
        assert len(kept) == 1, f"earlier retry must be kept, got {kept}"


class TestInterruptedTrialsAreNotLost:
    """Operator report: episode 29's results were "completely missing".

    ``_relocate`` only runs after a trial RETURNS, so a process killed mid-trial
    left everything in ``.staging_trial_NNN`` -- present but hidden behind a
    dot-directory, so the episode looked empty.
    """

    def test_orphaned_staging_is_surfaced(self, tmp_path):
        from capx.envs.real_episode_runner import _adopt_orphans

        s = pathlib.Path(tmp_path) / ".staging_trial_002"
        (s / "trial_02_sandboxrc_0_reward_0.000_taskcompleted_0").mkdir(parents=True)
        (s / "trial_02_sandboxrc_0_reward_0.000_taskcompleted_0" / "code.py").write_text("x")
        _adopt_orphans(str(tmp_path))
        assert not list(pathlib.Path(tmp_path).glob(".staging*"))
        assert (
            pathlib.Path(tmp_path)
            / "orphaned/staging_trial_002"
            / "trial_02_sandboxrc_0_reward_0.000_taskcompleted_0/code.py"
        ).exists()

    def test_empty_staging_is_just_removed(self, tmp_path):
        from capx.envs.real_episode_runner import _adopt_orphans

        (pathlib.Path(tmp_path) / ".staging_trial_003").mkdir()
        _adopt_orphans(str(tmp_path))
        assert not list(pathlib.Path(tmp_path).glob(".staging*"))
        assert not (pathlib.Path(tmp_path) / "orphaned").exists()

    def test_interrupted_migration_is_folded_in(self, tmp_path):
        from capx.envs.real_episode_runner import _adopt_orphans

        att = pathlib.Path(tmp_path) / "episode_028/attempt_1"
        att.mkdir(parents=True)
        (att / "code.py").write_text("final")
        stray = att.with_name("attempt_1.migrating")
        stray.mkdir()
        (stray / "extra.txt").write_text("only here")
        _adopt_orphans(str(tmp_path))
        assert (att / "extra.txt").read_text() == "only here"
        assert not stray.exists()

    def test_conflicting_duplicate_is_kept_not_deleted(self, tmp_path):
        """trial.py rewrites all_responses.json; the older copy differs."""
        from capx.envs.real_episode_runner import _adopt_orphans

        att = pathlib.Path(tmp_path) / "episode_028/attempt_1"
        att.mkdir(parents=True)
        (att / "all_responses.json").write_text("final and longer")
        stray = att.with_name("attempt_1.migrating")
        stray.mkdir()
        (stray / "all_responses.json").write_text("older")
        _adopt_orphans(str(tmp_path))
        assert (att / "all_responses.json").read_text() == "final and longer"
        assert (att / "all_responses.json.pre_migration").read_text() == "older"


class TestNoEpisodeTrials:
    """A trial that timed out waiting for the driver is not an episode.

    Operator saw an ``episode_pending`` folder beside real episodes. Its
    contents were real (a 600s first-frame timeout, 0 code blocks) but it is a
    failed startup wait, not an episode, and it should not be listed as one.
    """

    def _run_no_frame(self, tmp_path):
        import types

        import capx.envs.real_episode_runner as R

        class Env:
            _wire_episode = None          # no frame ever arrives
            _wire_task = None

        env = Env()

        def fake(env_, trial, args, config, mtp):
            out = pathlib.Path(config["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            d = out / f"trial_{trial:02d}_sandboxrc_1_reward_0.000_taskcompleted_0"
            d.mkdir()
            (d / "summary.txt").write_text("Trial timed out after 1000 seconds.")
            return types.SimpleNamespace(trial=trial)

        orig = (
            R._run_trial_with_retries, R._setup_output_dir,
            R._print_and_save_summary, R.instantiate,
        )
        R._run_trial_with_retries = fake
        R._setup_output_dir = lambda a, c: None
        R._print_and_save_summary = lambda *a, **k: None
        R.instantiate = lambda f: env
        try:
            R.run_real_episode_trials(
                types.SimpleNamespace(model="m"), {"cfg": {}},
                {"total_trials": 1, "output_dir": str(tmp_path), "num_workers": 1},
                0.0,
            )
        finally:
            (
                R._run_trial_with_retries, R._setup_output_dir,
                R._print_and_save_summary, R.instantiate,
            ) = orig

    def test_no_episode_dir_is_created(self, tmp_path):
        self._run_no_frame(tmp_path)
        assert not list(pathlib.Path(tmp_path).glob("episode_*")), (
            "a failed startup wait must not appear as an episode"
        )
        assert (pathlib.Path(tmp_path) / "no_episode_trials/attempt_1").is_dir()

    def test_no_episode_manifest_is_written(self, tmp_path):
        self._run_no_frame(tmp_path)
        assert not list(pathlib.Path(tmp_path).rglob("episode.json"))

    def test_ledger_explains_why(self, tmp_path):
        self._run_no_frame(tmp_path)
        rows = [
            json.loads(l)
            for l in (pathlib.Path(tmp_path) / "episodes.jsonl")
            .read_text()
            .splitlines()
            if l.strip()
        ]
        assert rows[0]["episode_id"] is None
        assert "no driver frame" in rows[0]["note"]
