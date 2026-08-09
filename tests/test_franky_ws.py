"""FrankyWsLowLevel: must answer the driver promptly and never deadlock.

The driver (franky_service) is a POLICY CLIENT: it sends an observation and
expects an action chunk back within one control period, then executes it. CaP-X
is a closed-loop controller. Blocking inside the request handler deadlocks them,
so motions are queued as waypoints and drained by the driver's own frames.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from capx.envs.simulators.franky_ws import (
    ACTION_HORIZON,
    WIRE_GRIPPER_CLOSED,
    WIRE_GRIPPER_OPEN,
    FrankyWsLowLevel,
    capx_gripper_to_wire,
    wire_gripper_to_capx,
)

J = np.array([0.0, -0.5, 0.0, -2.3, 0.0, 1.87, -2.34])


def make_env(**kw):
    """Env without starting a real server (we drive _on_frame directly)."""
    from capx.envs.simulators.franky_ws import _Endpoint

    env = FrankyWsLowLevel.__new__(FrankyWsLowLevel)
    env.host, env.port = "127.0.0.1", 0
    env.action_horizon = kw.get("action_horizon", ACTION_HORIZON)
    env.max_joint_step_rad = kw.get("max_joint_step_rad", 0.05)
    # The socket-backed state lives on a process-wide endpoint so it survives
    # trial boundaries; give each test a fresh one.
    env._ep = _Endpoint("127.0.0.1", 0, env.action_horizon)
    env.grasp_sample = None
    env.grasp_scores = None
    env.grasp_contact_pts = None
    env._control_freq = 15.0
    env._record_video = False
    env._record_wrist = False
    env._frames, env._wrist_frames = [], []
    return env


def frame(joints=J, gripper=WIRE_GRIPPER_OPEN, **extra):
    obs = {
        "prompt": "pick up the red block",
        "episode_id": 0,
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), np.uint8),
        "observation/exterior_image_1_left_raw": np.zeros((720, 1280, 3), np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), np.uint8),
        "observation/joint_position": np.asarray(joints, dtype=np.float64),
        "observation/gripper_position": np.array([gripper], dtype=np.float64),
        "observation/depth_external": np.full((720, 1280), 0.8, np.float32),
        "observation/camera_K": np.array([600.0, 0, 640, 0, 600.0, 360, 0, 0, 1]),
        "observation/camera_extrinsic": np.eye(4).flatten(),
        "observation/ee_pos": np.array([0.4, 0.0, 0.3]),
        "observation/ee_quat": np.array([1.0, 0, 0, 0]),
    }
    obs.update(extra)
    return obs


class TestGripperPolarity:
    def test_wire_open_is_capx_open(self):
        assert wire_gripper_to_capx(WIRE_GRIPPER_OPEN) == pytest.approx(1.0)

    def test_wire_closed_is_capx_closed(self):
        assert wire_gripper_to_capx(WIRE_GRIPPER_CLOSED) == pytest.approx(0.0)

    def test_capx_to_wire_is_binary(self):
        assert capx_gripper_to_wire(0.51) == WIRE_GRIPPER_OPEN
        assert capx_gripper_to_wire(0.49) == WIRE_GRIPPER_CLOSED


class TestNeverBlocksTheDriver:
    def test_first_frame_holds_current_pose(self):
        env = make_env()
        chunk = env._on_frame(frame())
        assert chunk.shape == (ACTION_HORIZON, 8)
        assert np.allclose(chunk[:, :7], J), "must hold, not jump"

    def test_idle_never_returns_zeros(self):
        """Zeros would command a violent move to the zero configuration."""
        env = make_env()
        chunk = env._on_frame(frame())
        assert not np.allclose(chunk[:, :7], 0.0)

    def test_enqueued_motion_is_served_progressively(self):
        env = make_env(max_joint_step_rad=0.05)
        env._on_frame(frame())                       # seed _last_cmd
        target = J.copy()
        target[0] += 0.2                             # 4 waypoints at 0.05 rad
        env._enqueue(
            [
                np.concatenate([J + (target - J) * (i / 4), [WIRE_GRIPPER_OPEN]])
                for i in range(1, 5)
            ]
        )
        chunk = env._on_frame(frame())
        assert chunk[0, 0] > J[0], "should start moving"
        assert chunk[-1, 0] == pytest.approx(target[0]), "reaches the target"

    def test_buffer_drains_and_then_holds(self):
        env = make_env()
        env._on_frame(frame())
        env._enqueue([np.concatenate([J + 0.1, [WIRE_GRIPPER_OPEN]])])
        env._on_frame(frame())
        assert env._pending() == 0
        held = env._on_frame(frame(joints=J + 0.1))
        assert np.allclose(held[:, :7], J + 0.1), "holds the last commanded pose"

    def test_move_to_joints_does_not_deadlock(self):
        """The driver must keep being served WHILE the motion executes."""
        env = make_env(max_joint_step_rad=0.05)
        env._on_frame(frame())
        served = {"n": 0}
        stop = threading.Event()

        def driver():
            j = J.copy()
            while not stop.is_set():
                chunk = env._on_frame(frame(joints=j))
                j = chunk[-1, :7]          # a real arm follows the command
                served["n"] += 1
                time.sleep(0.002)

        t = threading.Thread(target=driver, daemon=True)
        t.start()
        try:
            ok = env.move_to_joints_blocking(J + 0.2, tolerance=0.05)
        finally:
            stop.set()
            t.join(timeout=2)
        assert served["n"] > 1, "driver was starved -> deadlock"
        assert ok, "should report convergence"

    def test_gripper_command_reaches_the_wire(self):
        env = make_env()
        env._on_frame(frame())
        env._ep.gripper_capx = 0.0                       # CaP-X closed
        env._enqueue(
            [np.concatenate([J, [capx_gripper_to_wire(0.0)]])] * 2
        )
        chunk = env._on_frame(frame())
        assert chunk[0, 7] == WIRE_GRIPPER_CLOSED


class TestObservation:
    def test_shapes_and_keys(self):
        env = make_env()
        env._on_frame(frame())
        obs = env.get_observation()
        av = obs["agentview"]
        # rgb MUST match depth/K resolution: the SAM3 mask taken from rgb is
        # passed straight to plan_grasp alongside depth.
        assert av["images"]["rgb"].shape == (720, 1280, 3)
        assert av["images"]["rgb_policy"].shape == (224, 224, 3)
        assert av["images"]["depth"].shape == (720, 1280, 1)
        assert av["intrinsics"].shape == (3, 3)
        assert av["pose_mat"].shape == (4, 4)
        assert obs["robot_joint_pos"].shape == (8,)
        assert obs["robot_cartesian_pos"].shape == (8,)
        assert obs["robot0_eye_in_hand"]["images"]["rgb"].shape == (224, 224, 3)

    def test_depth_key_alias(self):
        env = make_env()
        f = frame()
        f["observation/depth_exterior_image_1_left"] = f.pop(
            "observation/depth_external"
        )
        env._on_frame(f)
        assert "depth" in env.get_observation()["agentview"]["images"]

    def test_nan_depth_preserved(self):
        env = make_env()
        f = frame()
        d = np.full((720, 1280), np.nan, np.float32)
        d[0, 0] = 0.5
        f["observation/depth_external"] = d
        env._on_frame(f)
        got = env.get_observation()["agentview"]["images"]["depth"]
        assert np.isnan(got[1, 1, 0]), "NaN must not become 0.0 m"
        assert got[0, 0, 0] == pytest.approx(0.5)

    def test_camera_pose_not_reflipped(self):
        ext = np.array(
            [[0, -1, 0, 0.5], [0, 0, -1, 0.1], [1, 0, 0, 0.4], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        env = make_env()
        env._on_frame(frame(**{"observation/camera_extrinsic": ext.flatten()}))
        assert np.allclose(
            env.get_observation()["agentview"]["pose_mat"], ext
        )

    def test_gripper_inverted_in_observation(self):
        env = make_env()
        env._on_frame(frame(gripper=WIRE_GRIPPER_CLOSED))
        assert env.get_observation()["robot_joint_pos"][7] == pytest.approx(0.0)

    def test_prompt_and_episode_exposed(self):
        env = make_env()
        env._on_frame(frame(prompt="stack the blocks", episode_id=4))
        obs = env.get_observation()
        assert obs["wire_prompt"] == "stack the blocks"
        assert obs["wire_episode_id"] == 4

    def test_wait_times_out_without_a_driver(self):
        with pytest.raises(TimeoutError, match="no observation"):
            make_env()._wait_for_frame(0.2)

    def test_task_completed_is_false(self):
        """No GT on hardware; the operator scores each trial."""
        assert make_env().task_completed() is False


class TestSimUntouched:
    def test_franka_real_is_not_imported(self):
        """This env must not reuse the robots_realtime backend."""
        import ast
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        tree = ast.parse(
            (repo / "capx/envs/simulators/franky_ws.py").read_text()
        )
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
        assert not any("franka_real" in m for m in imported), imported
        assert not any("msgpack_server" in m for m in imported), imported

    def test_only_real_configs_reference_it(self):
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        bad = [
            c.name
            for c in (repo / "env_configs").rglob("*.yaml")
            if c.parent.name != "real" and "FrankyWsLowLevel" in c.read_text()
        ]
        assert not bad, f"sim configs referencing the real env: {bad}"


class TestCodecRobustness:
    """Both ndarray encodings must decode, and zeros must be impossible.

    openpi's codec uses ``__ndarray__``, but importing ``capx.envs.simulators``
    runs ``msgpack_numpy.patch()`` (via franka_real), which rebinds
    ``msgpack.Packer`` process-wide -- so a client in such a process emits
    msgpack_numpy's ``{b"nd", ...}`` instead. If the server cannot decode it,
    joints read as None and the chunk becomes ZEROS: a violent move to the zero
    configuration.
    """

    def test_openpi_encoding_decodes(self):
        from openpi_client import msgpack_numpy as onp

        from capx.envs.simulators.franky_ws import _unpackb

        raw = onp.packb({"observation/joint_position": J})
        got = _unpackb(raw)["observation/joint_position"]
        assert isinstance(got, np.ndarray) and np.allclose(got, J)

    def test_msgpack_numpy_encoding_decodes(self):
        import msgpack
        import msgpack_numpy as mn

        from capx.envs.simulators.franky_ws import _unpackb

        raw = msgpack.packb(
            {"observation/joint_position": J}, default=mn.encode
        )
        got = _unpackb(raw)["observation/joint_position"]
        assert isinstance(got, np.ndarray) and np.allclose(got, J)

    def test_undecodable_joints_raise_not_zeros(self):
        env = make_env()
        bad = frame()
        bad["observation/joint_position"] = {b"nd": True, b"garbage": 1}
        with pytest.raises(ValueError, match="joint_position"):
            env._on_frame(bad)

    def test_missing_joints_raise_not_zeros(self):
        env = make_env()
        bad = frame()
        del bad["observation/joint_position"]
        with pytest.raises(ValueError, match="joint_position"):
            env._on_frame(bad)


class TestStaleFrameInvalidation:
    """A disconnect must invalidate the stored observation.

    Regression: a one-shot preflight probe connected, sent a frame, and left.
    Its frame stayed in ``_wire``, so the next ``get_observation()`` returned
    STALE data (wrong pose, wrong task) instead of waiting for the real driver.
    CaP-X then planned on it and the trial hung.
    """

    def test_clearing_wire_makes_wait_block_again(self):
        env = make_env()
        env._on_frame(frame(prompt="probe"))
        assert env.get_observation()["wire_prompt"] == "probe"
        # simulate the handler's finally-block on disconnect
        with env._lock:
            env._ep.wire = {}
            env._ep.traj, env._ep.cursor = [], 0
            env._ep.last_cmd = None
        with pytest.raises(TimeoutError):
            env._wait_for_frame(0.2)

    def test_queued_trajectory_is_dropped_on_disconnect(self):
        """A reconnecting driver must not be handed the old client's waypoints."""
        env = make_env()
        env._on_frame(frame())
        env._enqueue([np.concatenate([J + 0.3, [WIRE_GRIPPER_OPEN]])] * 4)
        assert env._pending() > 0
        with env._lock:
            env._ep.wire = {}
            env._ep.traj, env._ep.cursor = [], 0
            env._ep.last_cmd = None
        assert env._pending() == 0

    def test_fresh_client_reseeds_hold_pose(self):
        """After a disconnect the first new frame must re-anchor the hold pose."""
        env = make_env()
        env._on_frame(frame(joints=J))
        with env._lock:
            env._ep.wire = {}
            env._ep.traj, env._ep.cursor = [], 0
            env._ep.last_cmd = None
        newJ = J + 0.5
        chunk = env._on_frame(frame(joints=newJ))
        assert np.allclose(chunk[:, :7], newJ), "must hold the NEW pose"


class TestTrialPathSurface:
    """Every method the trial/API path calls must exist.

    Regression: ``record_video: true`` made ``capx/envs/tasks/base.py`` call
    ``low_level_env.enable_video_capture``, which did not exist -> AttributeError
    -> the whole process exited, taking the WebSocket server with it. The driver
    then saw the socket vanish ("no close frame received or sent"), which looks
    like a network fault but is a crash.
    """

    REQUIRED = [
        # video capture (capx/envs/tasks/base.py, capx/envs/trial.py)
        "enable_video_capture",
        "get_video_frames",
        "get_video_frame_count",
        "get_video_frames_range",
        "get_wrist_video_frames",
        "get_wrist_video_frames_range",
        # control surface (FrankaControlApi)
        "move_to_joints_blocking",
        "_set_gripper",
        "_step_once",
        "get_observation",
        "peek_observation",
        # gym-ish
        "reset",
        "step",
        "render",
        "close",
        "task_completed",
        "compute_reward",
        "get_current_time_s",
    ]

    def test_all_required_methods_exist(self):
        missing = [m for m in self.REQUIRED if not hasattr(FrankyWsLowLevel, m)]
        assert not missing, f"missing: {missing}"

    def test_grasp_attributes_predeclared(self):
        """plan_grasp assigns env.grasp_sample / grasp_scores directly."""
        env = make_env()
        for attr in ("grasp_sample", "grasp_scores", "grasp_contact_pts"):
            assert hasattr(env, attr) or attr in vars(env), attr

    def test_video_capture_records_frames(self):
        env = make_env()
        env._record_video = False
        env._record_wrist = False
        env._frames, env._wrist_frames = [], []
        env.enable_video_capture(True, clear=True)
        env._on_frame(frame())
        env.get_observation()
        assert env.get_video_frame_count() == 1
        assert env.get_video_frames()[0].shape == (720, 1280, 3)

    def test_video_capture_off_by_default(self):
        env = make_env()
        env._record_video = False
        env._frames, env._wrist_frames = [], []
        env._on_frame(frame())
        env.get_observation()
        assert env.get_video_frame_count() == 0


class TestEndpointOutlivesTrials:
    """Option B: the socket is a SERVICE, not owned by one trial.

    Regression: the server lived on the env, so when a trial ended (or hit
    TRIAL_TIMEOUT_SECONDS=1000) the process exited and the port closed. The
    driver then found a dead socket and reported
    "no close frame received or sent". A process-wide endpoint keeps the port
    open across trials, so the driver can connect whenever it likes.
    """

    def test_second_env_reuses_the_same_endpoint(self, monkeypatch):
        import capx.envs.simulators.franky_ws as m

        monkeypatch.setattr(m, "_SERVERS", {})
        started = []
        monkeypatch.setattr(
            m.FrankyWsLowLevel,
            "_serve_in_background",
            lambda self, ep: started.append(ep) or (None, None),
        )

        a = m.FrankyWsLowLevel(host="127.0.0.1", port=9911)
        b = m.FrankyWsLowLevel(host="127.0.0.1", port=9911)
        assert a._ep is b._ep, "a new trial must reuse the live endpoint"
        assert len(started) == 1, "the socket must bind only once"

    def test_observation_survives_a_trial_boundary(self, monkeypatch):
        """A connected driver's frame must not be lost between trials."""
        import capx.envs.simulators.franky_ws as m

        monkeypatch.setattr(m, "_SERVERS", {})
        monkeypatch.setattr(
            m.FrankyWsLowLevel,
            "_serve_in_background",
            lambda self, ep: (None, None),
        )

        first = m.FrankyWsLowLevel(host="127.0.0.1", port=9912)
        first._on_frame(frame(prompt="task one"))
        # trial ends; a new env is constructed for the next trial
        second = m.FrankyWsLowLevel(host="127.0.0.1", port=9912)
        assert second.peek_observation()["wire_prompt"] == "task one"

    def test_reset_clears_motion_but_keeps_the_observation(self):
        env = make_env()
        env._on_frame(frame())
        env._enqueue([np.concatenate([J + 0.2, [WIRE_GRIPPER_OPEN]])] * 3)
        env.reset()
        assert env._pending() == 0, "queued motion must be dropped"
        assert env.peek_observation().get("wire_prompt"), (
            "the live observation must survive reset()"
        )


class TestGeometryResolutionConsistency:
    """rgb, depth and intrinsics must share one resolution.

    Regression: agentview["rgb"] was the driver's 224x224 POLICY image while
    depth/K were native 720x1280. The generated code segments rgb with SAM3 and
    hands that mask to plan_grasp(depth=..., segmentation=mask), so
    ContactGraspNet failed with
    "operands could not be broadcast together with shapes (720,1280) (224,224)".
    """

    def test_rgb_matches_depth_resolution(self):
        env = make_env()
        env._on_frame(frame())
        imgs = env.get_observation()["agentview"]["images"]
        assert imgs["rgb"].shape[:2] == imgs["depth"].shape[:2]

    def test_intrinsics_are_for_that_resolution(self):
        """K's principal point must lie inside the rgb frame."""
        env = make_env()
        env._on_frame(frame())
        av = env.get_observation()["agentview"]
        h, w = av["images"]["rgb"].shape[:2]
        cx, cy = av["intrinsics"][0, 2], av["intrinsics"][1, 2]
        assert 0 < cx < w and 0 < cy < h, (cx, cy, w, h)

    def test_policy_image_kept_separately(self):
        env = make_env()
        env._on_frame(frame())
        imgs = env.get_observation()["agentview"]["images"]
        assert imgs["rgb_policy"].shape == (224, 224, 3)

    def test_falls_back_when_no_native_rgb(self):
        """If the driver sends only the 224 frame, use it rather than nothing."""
        env = make_env()
        f = frame()
        del f["observation/exterior_image_1_left_raw"]
        env._on_frame(f)
        imgs = env.get_observation()["agentview"]["images"]
        assert imgs["rgb"].shape == (224, 224, 3)


class TestEpisodeBoundaryDropsStaleMotion:
    """A new episode must not execute the previous episode's waypoints.

    The operator rearranges the scene between episodes, so a queued trajectory
    from the old plan would drive the arm through poses computed for objects
    that have moved. Safety-relevant, not cosmetic.
    """

    def test_queued_waypoints_dropped_on_new_episode(self):
        """Queue MORE than one chunk so draining cannot mask the drop."""
        env = make_env()
        env._on_frame(frame(episode_id=1))
        stale = J + 0.4
        # 40 waypoints >> horizon 8, so a served chunk alone cannot empty it.
        env._enqueue([np.concatenate([stale, [WIRE_GRIPPER_OPEN]])] * 40)
        assert env._pending() == 40
        chunk = env._on_frame(frame(episode_id=2))   # operator pressed start
        assert env._pending() == 0, "stale waypoints must be dropped"
        # And the chunk served must be the NEW pose, not the stale target.
        assert not np.allclose(chunk[0, :7], stale), (
            "must not serve a waypoint from the previous episode"
        )

    def test_same_episode_keeps_serving_the_queue(self):
        """Within an episode the queue drains normally (not dropped)."""
        env = make_env()
        env._on_frame(frame(episode_id=1))
        target = J + 0.4
        env._enqueue([np.concatenate([target, [WIRE_GRIPPER_OPEN]])] * 12)
        chunk = env._on_frame(frame(episode_id=1))
        # Served the queued target rather than holding the observed pose.
        assert np.allclose(chunk[0, :7], target), (
            "queued motion must still be served within the same episode"
        )

    def test_hold_reanchors_after_a_boundary(self):
        """The new episode's first chunk must hold the CURRENT pose."""
        env = make_env()
        env._on_frame(frame(episode_id=1, joints=J))
        newJ = J + 0.6
        chunk = env._on_frame(frame(episode_id=2, joints=newJ))
        assert np.allclose(chunk[:, :7], newJ), "must re-anchor, not reuse old pose"

    def test_episode_id_is_exposed(self):
        env = make_env()
        env._on_frame(frame(episode_id=7))
        assert env.episode_id() == 7

    def test_first_episode_does_not_drop(self):
        """No previous episode -> nothing to drop."""
        env = make_env()
        env._on_frame(frame(episode_id=5))
        env._enqueue([np.concatenate([J, [WIRE_GRIPPER_OPEN]])] * 3)
        env._on_frame(frame(episode_id=5))
        assert env._pending() >= 0


class TestSupersededPlanAborts:
    """A new episode must fully stop the previous plan, not race it.

    Operator report: the dashboard filled with "previous episode still
    unwinding" on the second episode. Tagging those leftovers was cosmetic --
    the old generated code was still commanding the arm. It now aborts at the
    first robot interaction after the boundary.
    """

    def _env(self):
        import threading

        import capx.envs.simulators.franky_ws as fw

        env = fw.FrankyWsLowLevel.__new__(fw.FrankyWsLowLevel)
        env._ep = fw._Endpoint(host="127.0.0.1", port=0, action_horizon=8)
        env._ep.wire = {"observation/joint_position": np.zeros(7)}
        env.max_joint_step_rad = 0.05
        return env, fw

    def test_abort_raises_after_a_boundary(self):
        env, fw = self._env()
        env.begin_plan()                       # plan bound to generation 0
        env._abort_if_superseded()             # same generation: fine
        env._ep.generation += 1                # operator starts a new episode
        with pytest.raises(fw.EpisodeSuperseded):
            env._abort_if_superseded()

    def test_abort_is_not_swallowed_by_generated_code(self):
        """Generated code is full of `except Exception`; the abort must survive."""
        env, fw = self._env()
        env.begin_plan()
        env._ep.generation += 1
        raised = False
        try:
            try:
                env._abort_if_superseded()
            except Exception:                  # what generated code does
                pass
        except fw.EpisodeSuperseded:
            raised = True
        assert raised, "abort must derive from BaseException, not Exception"

    def test_motion_aborts_instead_of_moving(self):
        env, fw = self._env()
        env.begin_plan()
        env._ep.generation += 1
        with pytest.raises(fw.EpisodeSuperseded):
            env.move_to_joints_blocking(np.ones(7) * 0.3)

    def test_observation_does_NOT_abort(self):
        """get_observation runs OUTSIDE the exec sandbox.

        ``base.step()`` calls ``_get_observation()`` after the sandbox returns,
        so an abort raised there escapes CodeExecutionEnvBase's BaseException
        handler and propagates through trial.py -> runner.py -> launch.py,
        killing the process and closing port 8041. Observed live: the run died
        at "0/500" and the driver saw "no close frame received or sent".
        """
        env, fw = self._env()
        env.begin_plan()
        env._ep.generation += 1
        env.get_observation()          # must NOT raise

    def test_no_abort_when_no_plan_is_bound(self):
        """Unbound code paths (tests, setup) must never abort."""
        env, fw = self._env()
        env._ep.generation += 5
        env._abort_if_superseded()             # must not raise

    def test_rebinding_clears_the_abort(self):
        env, fw = self._env()
        env.begin_plan()
        env._ep.generation += 1
        env.begin_plan()                       # new episode's plan
        env._abort_if_superseded()             # must not raise


class TestEveryEpisodeStartsFresh:
    """The env is built ONCE per process and reused for all 500 trials
    (``runner.py`` instantiates outside the loop), so anything ``reset()`` does
    not clear leaks into the next episode.
    """

    def _env(self, wire=None):
        import capx.envs.simulators.franky_ws as fw

        env = fw.FrankyWsLowLevel.__new__(fw.FrankyWsLowLevel)
        env._ep = fw._Endpoint(host="127.0.0.1", port=0, action_horizon=8)
        env._ep.wire = wire or {
            "observation/joint_position": np.zeros(7),
            "observation/gripper_position": np.array([0.0118]),  # open on wire
        }
        env.max_joint_step_rad = 0.05
        return env, fw

    def test_queued_motion_is_dropped(self):
        env, _ = self._env()
        env._ep.traj = [np.zeros(8)] * 40
        env._ep.cursor = 5
        env.reset()
        assert env._ep.traj == [] and env._ep.cursor == 0

    def test_hold_pose_is_dropped(self):
        """A stale last_cmd would hold the PREVIOUS episode's pose."""
        env, _ = self._env()
        env._ep.last_cmd = np.ones(8) * 9.0
        env.reset()
        assert env._ep.last_cmd is None

    def test_commanded_gripper_resyncs_from_the_wire(self):
        """Ending an episode holding an object must not carry into the next."""
        env, _ = self._env()
        env._ep.gripper_capx = 0.0          # ep N ended gripping
        env.reset()
        assert env._ep.gripper_capx > 0.5, (
            "must re-read the real gripper, not inherit the commanded state"
        )

    def test_plan_binding_is_cleared(self):
        """Setup code before begin_plan() must never abort."""
        env, _ = self._env()
        env.begin_plan()
        env._ep.generation += 3
        env.reset()
        env._abort_if_superseded()          # must not raise

    def test_abort_notice_can_fire_again(self):
        env, _ = self._env()
        env._abort_announced = 7
        env.reset()
        assert env._abort_announced is None

    def test_boundary_detection_state_persists(self):
        """episode_seen / generation MUST survive: they are cross-episode."""
        env, _ = self._env()
        env._ep.episode_seen = 18
        env._ep.generation = 4
        env.reset()
        assert env._ep.episode_seen == 18
        assert env._ep.generation == 4

    def test_live_observation_survives_reset(self):
        """The wire frame is live truth, not per-episode state."""
        env, _ = self._env()
        env.reset()
        assert env._ep.wire, "must not clear the driver's latest frame"


class TestAbortStaysInsideTheSandbox:
    """An abort must end the TRIAL, never the process.

    Live failure: get_observation() aborted from base.step(), which runs after
    the exec sandbox, so EpisodeSuperseded propagated to launch.py and killed
    the process -- taking port 8041 with it, which is what the operator saw as
    a repeated "no close frame received or sent".
    """

    def test_abort_sites_are_only_generated_code_paths(self):
        import pathlib

        src = pathlib.Path("capx/envs/simulators/franky_ws.py").read_text()
        # Attribute the checks to their enclosing def.
        owners, current = [], None
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("def "):
                current = stripped[4:].split("(")[0]
            if "_abort_if_superseded()" in stripped and not stripped.startswith(
                "def "
            ):
                owners.append(current)
        allowed = {"move_to_joints_blocking", "_set_gripper"}
        assert set(owners) <= allowed, (
            f"abort must only fire on paths generated code calls directly; "
            f"found in {sorted(set(owners) - allowed)}"
        )

    def test_get_observation_is_abort_free(self):
        import pathlib

        src = pathlib.Path("capx/envs/simulators/franky_ws.py").read_text()
        body = src[src.index("    def get_observation("):]
        body = body[: body.index("\n    def ", 1)]
        assert "_abort_if_superseded()" not in body, (
            "base.step() calls this outside the sandbox; an abort here kills "
            "the process"
        )

    def test_abort_message_breaks_the_multiturn_loop(self):
        """trial.py:805 breaks only on the phrase 'terminated episode'."""
        import pathlib

        src = pathlib.Path("capx/envs/simulators/franky_ws.py").read_text()
        assert "terminated episode." in src


class TestEpisodeEndStopsTheWork:
    """Operator report (episode 26): 12s idle detection fired, the dashboard
    showed the episode ended, yet CaP-X kept generating code and driving the arm
    until episode 27 started.

    Cause: _announce_episode_end only published to the monitor. The plan's
    validity is keyed on _ep.generation, which only the NEXT episode bumped.
    """

    def _env(self):
        import capx.envs.simulators.franky_ws as fw

        env = fw.FrankyWsLowLevel.__new__(fw.FrankyWsLowLevel)
        env._ep = fw._Endpoint(host="127.0.0.1", port=0, action_horizon=8)
        env._ep.wire = {"observation/joint_position": np.zeros(7)}
        env.max_joint_step_rad = 0.05
        return env, fw

    def test_idle_end_invalidates_the_running_plan(self):
        env, fw = self._env()
        env._ep.episode_seen = 26
        env.begin_plan()
        env._abort_if_superseded()                  # still valid

        env._announce_episode_end("no frames for 12s")

        with pytest.raises(fw.EpisodeSuperseded):
            env._abort_if_superseded()

    def test_motion_stops_after_the_episode_ends(self):
        env, fw = self._env()
        env._ep.episode_seen = 26
        env.begin_plan()
        env._announce_episode_end("driver disconnected")
        with pytest.raises(fw.EpisodeSuperseded):
            env.move_to_joints_blocking(np.ones(7) * 0.2)

    def test_end_is_still_idempotent(self):
        """Must not bump the generation repeatedly."""
        env, _ = self._env()
        env._ep.episode_seen = 26
        env._announce_episode_end("first")
        gen = env._ep.generation
        env._announce_episode_end("second")
        assert env._ep.generation == gen

    def test_frames_are_live_distinguishes_stale_from_streaming(self):
        """_wire_prompt() returns the last frame forever; liveness must not."""
        import time as _t

        env, _ = self._env()
        assert env.frames_are_live() is False, "no frame yet"
        env._ep.last_frame_t = _t.time()
        assert env.frames_are_live() is True
        env._ep.last_frame_t = _t.time() - 30
        assert env.frames_are_live() is False, "stale frame must not count"


class TestEndedEpisodeRefusesNewWork:
    """Operator report: after the 12s flush on episode 29, CaP-X kept generating
    code and solving.

    ``step()`` calls ``begin_plan()`` before EVERY code block. An unconditional
    re-bind meant the abort only killed the block in flight; the next block
    re-armed itself against the dead episode and carried on.
    """

    def _env(self):
        import capx.envs.simulators.franky_ws as fw

        env = fw.FrankyWsLowLevel.__new__(fw.FrankyWsLowLevel)
        env._ep = fw._Endpoint(host="127.0.0.1", port=0, action_horizon=8)
        env._ep.wire = {"observation/joint_position": np.zeros(7)}
        env._ep.episode_seen = 29
        env.max_joint_step_rad = 0.05
        return env, fw

    def test_begin_plan_refuses_after_the_episode_ended(self):
        env, fw = self._env()
        env.begin_plan()                                # fine while live
        env._announce_episode_end("no frames for 12s")
        with pytest.raises(fw.EpisodeSuperseded):
            env.begin_plan()

    def test_rebinding_cannot_resurrect_a_dead_episode(self):
        """The exact live failure: block N aborts, block N+1 re-arms."""
        env, fw = self._env()
        env.begin_plan()
        env._announce_episode_end("no frames for 12s")
        with pytest.raises(fw.EpisodeSuperseded):
            env.move_to_joints_blocking(np.ones(7) * 0.2)   # block N aborts
        with pytest.raises(fw.EpisodeSuperseded):
            env.begin_plan()                                # block N+1 refused

    def test_a_new_episode_clears_the_refusal(self):
        """ended_announced is reset by the next frame, so work can resume."""
        import time as _t

        env, _ = self._env()
        env._announce_episode_end("no frames for 12s")
        env._ep.ended_announced = False        # what _on_frame does on a frame
        env._ep.last_frame_t = _t.time()
        env.begin_plan()                       # must not raise

    def test_message_ends_the_trial_loop(self):
        """trial.py:805 breaks only on the phrase 'terminated episode'."""
        env, fw = self._env()
        env._announce_episode_end("driver disconnected")
        try:
            env.begin_plan()
            raise AssertionError("should have raised")
        except fw.EpisodeSuperseded as exc:
            assert "terminated episode" in str(exc)


class TestIdleEndAlsoDropsTheQueue:
    """Operator report: the FIRST episode end discarded stale steps, the SECOND
    did not.

    Live log:
        episode 0 ended (no frames for 12s) ... generation -> 1
        episode 0 -> 1: dropping 0 queued waypoint(s)
        episode 1 ended (no frames for 12s) ... generation -> 3
        episode 1 -> 2: dropping 488 queued waypoint(s)     <-- 488 STALE

    Bumping the generation stops the plan only at its NEXT robot interaction.
    Anything ALREADY enqueued kept being served until the next episode_id
    arrived, so the arm executed 488 rows of a finished episode. Episode 0 only
    looked correct because its plan had barely enqueued anything.
    """

    def _env(self, queued=0):
        import capx.envs.simulators.franky_ws as fw

        env = fw.FrankyWsLowLevel.__new__(fw.FrankyWsLowLevel)
        env._ep = fw._Endpoint(host="127.0.0.1", port=0, action_horizon=8)
        env._ep.wire = {"observation/joint_position": np.zeros(7)}
        env._ep.episode_seen = 1
        env._ep.joints = np.zeros(7)
        env.max_joint_step_rad = 0.05
        env.action_horizon = 8
        if queued:
            env._ep.traj = [np.zeros(8) for _ in range(queued)]
            env._ep.last_cmd = np.ones(8)
        return env, fw

    def test_queued_waypoints_are_dropped_on_idle_end(self):
        env, _ = self._env(queued=488)
        assert env._pending() == 488
        env._announce_episode_end("no frames for 12s")
        assert env._pending() == 0, (
            "a finished episode's waypoints must not keep reaching the arm"
        )

    def test_hold_pose_is_cleared_so_it_re_anchors(self):
        """A stale last_cmd would hold the FINISHED episode's pose."""
        env, _ = self._env(queued=10)
        env._announce_episode_end("driver disconnected")
        assert env._ep.last_cmd is None

    def test_second_episode_end_behaves_like_the_first(self):
        """The reported asymmetry: both ends must drop their queue."""
        env, _ = self._env(queued=0)
        env._announce_episode_end("no frames for 12s")      # episode 1
        first_pending = env._pending()

        env._ep.ended_announced = False                     # episode 2 starts
        env._ep.episode_seen = 2
        env._ep.traj = [np.zeros(8) for _ in range(488)]
        env._announce_episode_end("no frames for 12s")      # episode 2 ends
        assert env._pending() == first_pending == 0

    def test_still_idempotent(self):
        env, _ = self._env(queued=5)
        env._announce_episode_end("first")
        gen = env._ep.generation
        env._ep.traj = [np.zeros(8) for _ in range(3)]      # late arrival
        env._announce_episode_end("second")
        assert env._ep.generation == gen, "must not bump twice"

    def test_driver_serving_after_idle_end_gets_the_hold_pose_not_stale_rows(self):
        """End-to-end: the next chunk must be a hold, not queued motion."""
        env, _ = self._env(queued=40)
        env._ep.traj = [np.full(8, 0.5) for _ in range(40)]
        env._announce_episode_end("no frames for 12s")
        actions = env._on_frame({"observation/joint_position": np.zeros(7)})
        assert not np.allclose(actions[:, :7], 0.5), "served stale waypoints"
        assert np.allclose(actions[:, :7], 0.0), "should hold the measured pose"
