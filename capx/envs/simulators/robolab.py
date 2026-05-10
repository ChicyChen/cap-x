"""Franka Robolab environment.

Wraps a robolab Isaac-Sim long-horizon task as a CaP-X BaseEnv so the
existing CaP-Agent0 code-execution loop can drive it.

robolab is imported lazily (inside ``__init__``) because importing it
bootstraps Isaac Sim, which we don't want triggered just by loading
this module.

Action convention (robolab DroidJointPositionActionCfg):
- 7 absolute panda arm joint positions, ``use_default_offset=False``
- 1 binary gripper command, 0.0 = open, 1.0 = close (BinaryJointPositionZeroToOne)

CaP-X gripper convention is the inverse: ``_gripper_fraction = 0.0`` (closed)
to ``1.0`` (open). The adapter inverts on the way out.

Observation shape exposed to the API matches FrankaLiberoEnv so the
existing privileged-API plumbing can stay agnostic of the underlying
simulator.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv


class FrankaRobolabEnv(BaseEnv):
    """Franka Robolab environment (Isaac Sim, long-horizon tasks)."""

    def __init__(
        self,
        scene_name: str,
        max_steps: int = 4000,
        seed: int | None = None,
        device: str = "cuda:0",
        instruction_type: str = "default",
        gt_state_dump_dir: str | None = None,
        gt_state_dump_path: str | None = None,  # back-compat
        record_video: bool = True,
        video_subsample: int = 4,
    ) -> None:
        super().__init__()
        # ``CAPX_SCENE_NAME`` env-var overrides the YAML's scene_name so a
        # single config file can be reused across tasks (the wrapper
        # script's --scene-name flag sets this). Falls back to whatever
        # the YAML / kwargs provided.
        env_scene = os.environ.get("CAPX_SCENE_NAME")
        if env_scene:
            scene_name = env_scene
        self.scene_name = scene_name
        self.max_steps = max_steps
        self.seed = seed
        self.privileged = True
        self._device = device
        self._instruction_type = instruction_type
        # Per-trial gt_state files land here as ``trial_<seed>.jsonl``.
        # ``gt_state_dump_path`` is honoured for back-compat (dirname is used).
        # ``CAPX_GT_STATE_DUMP_DIR`` env var overrides the YAML value so the
        # wrapper script can keep gt_state staging under the same per-task
        # output directory as the cap-x trial artifacts.
        env_override = os.environ.get("CAPX_GT_STATE_DUMP_DIR")
        if env_override:
            gt_state_dump_dir = env_override
        elif gt_state_dump_dir is None and gt_state_dump_path is not None:
            gt_state_dump_dir = os.path.dirname(gt_state_dump_path)
        self._gt_state_dump_dir = gt_state_dump_dir
        self._gt_state_dump_fp = None
        self._gt_state_dump_path_resolved: str | None = None
        self._gt_state_trial_id: int | None = None

        # Per-trial video buffer.  Each step appends a downsampled RGB
        # frame; cap-x's ``CodeExecutionEnvBase.get_video_frames`` reads
        # the buffer and writes an MP4 inside the per-trial output dir.
        self._record_video = bool(record_video)
        self._video_subsample = max(1, int(video_subsample))
        # Native-fps streaming MP4 disabled by default (cap-x's per-turn
        # videos are sufficient). Re-enable per-trial via the
        # ``CAPX_RECORD_NATIVE_VIDEO`` env var if needed.
        self._record_native_video = (
            os.environ.get("CAPX_RECORD_NATIVE_VIDEO", "0") == "1"
        )
        self._native_video_writer = None
        self._native_video_path: str | None = None
        self._native_video_fps = 15

        # Lazy imports — these pull in Isaac Sim.
        import torch  # noqa: F401  (imported here to fail fast if missing)
        # Importing robolab triggers `import isaaclab.utils`, which only
        # works once the Isaac Sim AppLauncher has been started.  CaP-X's
        # launcher doesn't know about Isaac Sim — bootstrap it here if
        # nothing else has already.  Idempotent: re-using AppLauncher in
        # the same process is a no-op.
        _bootstrap_isaac_sim()
        _patch_trial_video_dir()

        # Enable subtask progress checking before constructing the env.
        # Without this, robolab's BaseRoboEnvCfg skips creating the
        # SubtaskCompletionRecorderTerm, GTStateExporter has no source
        # for ``object_completed`` / ``conditions`` / ``score``, and
        # the success termination never has the data it needs to fire.
        # This is the same flip that ``run_eval.py --enable-subtask``
        # performs at module-init time.
        import robolab.constants as _robolab_constants
        _robolab_constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True

        # Force robolab's recorder to write its data.hdf5 to a pod-local
        # path (default lives under the robolab package dir, which on
        # osmo points at the SHARED Lustre clone — 10 parallel pods all
        # writing to the same data.hdf5 collide on HDF5 file locks with
        # `errno=11 Resource temporarily unavailable` and create_env
        # raises before any trial runs. /tmp is pod-local; cap-x doesn't
        # consume data.hdf5 for its scoring (gt_state.jsonl is the
        # primary signal) so losing it on pod teardown is fine.
        _capx_data_dir = os.environ.get(
            "CAPX_ROBOLAB_OUTPUT_DIR", f"/tmp/robolab_output_{os.getpid()}"
        )
        os.makedirs(_capx_data_dir, exist_ok=True)
        _robolab_constants.set_output_dir(_capx_data_dir)

        # Inject ``wrist_cam_depth`` into env_cfg.observations.image_obs
        # so cap-x's S2/S3/S4 perception tiers (which expect both
        # cameras to expose depth for multiview point-cloud fusion)
        # work without modifying robolab. We do this by wrapping
        # ``parse_env_cfg`` — modifying ``ImageObsCfg`` at class level
        # doesn't take effect because ``@configclass`` froze its fields
        # at definition time.
        _patch_parse_env_cfg_for_wrist_depth()

        from robolab.core.environments.runtime import create_env
        from robolab.core.events.gt_state_exporter import GTStateExporter

        self._torch = torch
        self._create_env = create_env
        self._GTStateExporter = GTStateExporter

        self.handle, self.env_cfg = create_env(
            scene=scene_name,
            device=device,
            seed=seed if seed is not None else 0,
            num_envs=1,
            instruction_type=instruction_type,
            policy="capx",
        )
        # CaP-X's _patch_libero_goal substitutes `{libero_environment_goal}`
        # in the prompt by reading `handle.task_language`. Robolab's
        # ManagerBasedRLEnv doesn't expose that, so we monkey-patch a
        # task_language attr that mirrors the resolved instruction —
        # keeps the prompt template portable between LIBERO and robolab.
        try:
            self.handle.task_language = self.env_cfg.instruction
        except (AttributeError, TypeError):
            pass

        # Per-task wallclock budget for cap-x's runner. Robolab defines
        # ``episode_length_s`` on each LH task class (90 / 120 / 180 s
        # sim time across the LH common-sense suite). The 10× multiplier
        # covers Isaac Sim render time + cap-x agent code-gen latency
        # (measured ~9× on InferClearTableTask in run-8). Override via
        # CAPX_TRIAL_TIMEOUT_MULTIPLIER for sweeps that hit the cap.
        _eps_len = float(getattr(self.env_cfg, "episode_length_s", 120))
        _mult = float(os.environ.get("CAPX_TRIAL_TIMEOUT_MULTIPLIER", "10"))
        self.trial_timeout_s = int(_eps_len * _mult)
        print(
            f"[capx-robolab] scene={scene_name} episode_length_s={_eps_len:.0f} "
            f"× {_mult:.1f} → trial_timeout_s={self.trial_timeout_s}",
            flush=True,
        )

        # GT state exporter — single source of truth for object poses
        # in this adapter; also used to populate the per-step dump that
        # the offline Aspect-1 scorer consumes.
        self._gt_exporter = GTStateExporter(self.handle, self.env_cfg)

        # Latest cached env-step results
        self._latest_obs: dict[str, Any] | None = None
        self._latest_reward: float = 0.0
        self._latest_term: bool = False
        self._latest_trunc: bool = False
        self._latest_info: dict[str, Any] = {}
        self._latest_gt_state: dict[str, Any] | None = None

        # Control state
        self._control_freq = 20  # Hz; matches robolab default
        self._step_count = 0
        self._sim_step_count = 0
        # 0.0 = closed, 1.0 = open in CaP-X convention
        self._gripper_fraction = 1.0
        self._current_joints: np.ndarray = np.zeros(7, dtype=np.float64)

        # Used by FrankaLiberoEnv's _get_object_pose contract
        self.segmentation_level = "instance"

        # Cap-X's S3/S4 ``goto_home_joint_position`` reads
        # ``self._env.home_joint_position``. Cached at first reset.
        self.home_joint_position: np.ndarray | None = None

        # Cap-X's CodeExecutionEnvBase reads these on the low-level env
        # for viser overlay + video capture. We don't drive viser, and
        # video output comes from cap-x's higher-level recorder, not
        # from per-step renders pushed by this adapter.  Keep them as
        # no-op surfaces so cap-x's hasattr/inspect.signature checks
        # don't blow up.
        self.viser_debug = False
        self._frame_buffer: list = []
        self._wrist_frame_buffer: list = []

        if self._gt_state_dump_dir is not None:
            os.makedirs(self._gt_state_dump_dir, exist_ok=True)

        self.reset()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # CaP-X reuses one env across trials and calls reset(seed=trial_id)
        # per trial.  Rotate the per-trial gt_state file and frame buffer
        # at the start of each reset so the resulting artifacts can be
        # post-moved into the corresponding cap-x trial dir.
        trial_id = self._extract_trial_id(seed, options)
        self._open_gt_state_file(trial_id)
        self._open_native_video_writer(trial_id)
        self._frame_buffer.clear()
        self._wrist_frame_buffer.clear()

        # robolab's eval loop calls reset twice (see episode.py); replicate
        # to avoid initial-state quirks where the first reset returns
        # a partially-stepped state.
        obs, _ = self.handle.reset()
        obs, _ = self.handle.reset()

        self._latest_obs = obs
        self._gt_exporter.reset()

        self._step_count = 0
        self._sim_step_count = 0
        self._gripper_fraction = 1.0
        self._current_joints = self._read_arm_joints(obs)
        # Cache the reset configuration as the "home" pose so cap-x's
        # ``goto_home_joint_position`` works (LIBERO does the same in
        # FrankaLiberoEnv.reset).
        if self.home_joint_position is None:
            self.home_joint_position = self._current_joints.copy()

        # Post-reset settling so physics stabilises.
        for _ in range(10):
            self._step_once()

        gt_state = self._refresh_gt_state()
        self._dump_gt_state(gt_state)
        self._record_frame()

        info = {"task_prompt": self.env_cfg.instruction}
        return self.get_observation(), info

    def _extract_trial_id(
        self, seed: int | None, options: dict[str, Any] | None
    ) -> int | None:
        if options and "trial" in options:
            return int(options["trial"])
        if seed is not None:
            return int(seed)
        return None

    def _open_gt_state_file(self, trial_id: int | None) -> None:
        """(Re)open the per-trial gt_state.jsonl file, closing any prior one."""
        if self._gt_state_dump_fp is not None:
            try:
                self._gt_state_dump_fp.close()
            finally:
                self._gt_state_dump_fp = None
        if self._gt_state_dump_dir is None:
            self._gt_state_dump_path_resolved = None
            self._gt_state_trial_id = None
            return
        # ``trial_<NN>.jsonl`` matches the trial-NN convention cap-x uses
        # for its per-trial output dirs (``trial_NN_sandboxrc_*_reward_*_*``).
        # The post-process step uses this filename to land the file in the
        # right cap-x trial dir.
        if trial_id is None:
            # Fall back to a unique-per-call name so we don't overwrite.
            import time
            stem = f"trial_unknown.{os.getpid()}.{int(time.time())}.jsonl"
        else:
            stem = f"trial_{int(trial_id):02d}.jsonl"
        path = os.path.join(self._gt_state_dump_dir, stem)
        os.makedirs(self._gt_state_dump_dir, exist_ok=True)
        self._gt_state_dump_fp = open(path, "w")
        self._gt_state_dump_path_resolved = path
        self._gt_state_trial_id = trial_id

    def _open_native_video_writer(self, trial_id: int | None) -> None:
        """Open a per-trial streaming MP4 writer at robolab's native env-step
        rate (15 Hz). Closes any prior writer first.

        The output file is ``trial_<NN>_native.mp4`` next to the gt_state
        dump; the wrapper's post-process moves it into the per-trial
        cap-x output directory alongside ``video_combined.mp4``.
        """
        # Close prior writer (previous trial within this worker process).
        self._close_native_video_writer()
        if (
            not self._record_native_video
            or not self._record_video
            or self._gt_state_dump_dir is None
        ):
            self._native_video_path = None
            return
        try:
            import imageio
        except ImportError:
            self._native_video_writer = None
            self._native_video_path = None
            return
        os.makedirs(self._gt_state_dump_dir, exist_ok=True)
        if trial_id is None:
            import time
            stem = f"trial_unknown.{os.getpid()}.{int(time.time())}_native.mp4"
        else:
            stem = f"trial_{int(trial_id):02d}_native.mp4"
        path = os.path.join(self._gt_state_dump_dir, stem)
        try:
            self._native_video_writer = imageio.get_writer(
                path, fps=self._native_video_fps, codec="libx264", quality=8
            )
            self._native_video_path = path
        except Exception:
            # Don't break the trial if the writer can't open (e.g. ffmpeg
            # missing). Just skip native-fps output.
            self._native_video_writer = None
            self._native_video_path = None

    def _close_native_video_writer(self) -> None:
        if self._native_video_writer is None:
            return
        try:
            self._native_video_writer.close()
        except Exception:
            pass
        finally:
            self._native_video_writer = None

    def _record_native_frame(self) -> None:
        """Append every env.step's agentview RGB to the native-fps writer."""
        if self._native_video_writer is None:
            return
        rgb = _maybe_first_env(
            (self._latest_obs or {}).get("image_obs", {}).get("external_cam")
        )
        if rgb is None:
            return
        try:
            self._native_video_writer.append_data(np.asarray(rgb, dtype=np.uint8))
        except Exception:
            # Best-effort; never let video failure tank the trial.
            self._close_native_video_writer()

    def _record_frame(self) -> None:
        """Append a downsampled agentview RGB frame to the video buffer."""
        # Native-fps stream captures every step regardless of subsample.
        self._record_native_frame()
        if not self._record_video:
            return
        if self._sim_step_count % self._video_subsample != 0:
            return
        rgb = _maybe_first_env(
            (self._latest_obs or {}).get("image_obs", {}).get("external_cam")
        )
        if rgb is None:
            return
        # Buffered as uint8 (H, W, 3); cap-x's video writer feeds these
        # directly into imageio. Keep memory bounded.
        if len(self._frame_buffer) < 4096:
            self._frame_buffer.append(np.asarray(rgb, dtype=np.uint8).copy())

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step — not typically called directly under code-execution.

        Provided so this class is a valid gymnasium.Env. The CaP-Agent0
        loop drives the env through ``move_to_joints_blocking`` instead.
        """
        action_t = self._format_action(action)
        obs, reward, term, trunc, info = self.handle.step(action_t)
        self._cache_step(obs, reward, term, trunc, info)
        self._step_count += 1
        gt_state = self._refresh_gt_state()
        self._dump_gt_state(gt_state)
        truncated = self._latest_trunc or self._step_count >= self.max_steps
        return self.get_observation(), self._latest_reward, self._latest_term, truncated, info

    # ------------------------------------------------------------------
    # FrankaControlApi interface (called by FrankaRobolabPrivilegedApi)
    # ------------------------------------------------------------------

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.01,
        max_steps: int = 120,
    ) -> None:
        """Move to absolute target joint positions (radians).

        Robolab's action is absolute joint position (use_default_offset=False),
        so we pass ``target`` straight through — no delta scaling like LIBERO.
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        steps = 0
        while steps < max_steps:
            current = self._read_arm_joints(self._latest_obs)
            error = float(np.linalg.norm(current - target))
            if error < tolerance and steps > 0:
                break

            action = self._build_action(target, self._gripper_fraction)
            obs, reward, term, trunc, info = self.handle.step(action)
            self._cache_step(obs, reward, term, trunc, info)
            self._sim_step_count += 1
            self._refresh_gt_state()
            self._dump_gt_state(self._latest_gt_state)
            self._record_frame()
            self._check_robolab_terminations()
            steps += 1

    def _set_gripper(self, fraction: float) -> None:
        """Set gripper opening fraction (0.0 closed → 1.0 open, CaP-X convention)."""
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        """One env tick with the current control state held."""
        action = self._build_action(self._current_joints, self._gripper_fraction)
        obs, reward, term, trunc, info = self.handle.step(action)
        self._cache_step(obs, reward, term, trunc, info)
        self._sim_step_count += 1
        self._refresh_gt_state()
        self._dump_gt_state(self._latest_gt_state)
        self._record_frame()
        self._check_robolab_terminations()

    # ------------------------------------------------------------------
    # Action / observation plumbing
    # ------------------------------------------------------------------

    def _build_action(self, joints: np.ndarray, gripper_fraction: float):
        """Pack [7 abs joints + 1 binary gripper] into a [1, 8] torch tensor.

        CaP-X gripper_fraction: 0.0 = closed, 1.0 = open.
        Robolab BinaryJointPositionZeroToOne: 0.0 = open, 1.0 = closed.
        Threshold at 0.5 to keep it binary.
        """
        gripper_cmd = 0.0 if gripper_fraction > 0.5 else 1.0
        action_np = np.concatenate(
            [np.asarray(joints, dtype=np.float32).reshape(7), np.array([gripper_cmd], dtype=np.float32)]
        )
        return self._torch.from_numpy(action_np).to(self._device).unsqueeze(0)

    def _format_action(self, action: Any):
        """Accept either a [7+1] numpy/list or a pre-built torch tensor."""
        if hasattr(action, "to") and hasattr(action, "unsqueeze"):
            t = action
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t.to(self._device).float()
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        return self._torch.from_numpy(arr).to(self._device).float().unsqueeze(0)

    def _cache_step(self, obs, reward, term, trunc, info) -> None:
        self._latest_obs = obs
        self._latest_reward = float(_to_scalar(reward))
        self._latest_term = bool(_to_scalar(term))
        self._latest_trunc = bool(_to_scalar(trunc))
        self._latest_info = info or {}

    def _check_robolab_terminations(self) -> None:
        """End the trial cleanly when robolab fires ``terminated`` or
        ``truncated``.

        IsaacLab's ``ManagerBasedRLEnv.step()`` auto-resets the scene
        on either flag. Without surfacing these to cap-x's trial loop:
        - **Success** (``terminated=True``): the agent doesn't know the
          task is done and may keep generating code on a re-spawned scene.
        - **Time_out** (``truncated=True``): same hazard — the agent's
          next code block silently runs against fresh state.

        Cap-x's multi-turn loop already has a magic-string check
        (``trial.py:805``: ``if "terminated episode" in info_step["stderr"]:
        break``). We propagate to it by raising ``RuntimeError`` whose
        message contains exactly that substring. The exec wrapper catches
        the exception, writes ``repr(exc)`` to ``info_step["stderr"]``,
        and cap-x's loop breaks the trial cleanly.
        """
        if not (self._latest_term or self._latest_trunc):
            return
        if self._latest_term:
            kind = "success termination"
        else:
            kind = "time_out (episode_length_s reached)"
        raise RuntimeError(
            f"Robolab {kind} at sim_step={self._sim_step_count}; "
            f"terminated episode."
        )

    def _read_arm_joints(self, obs: dict[str, Any] | None) -> np.ndarray:
        """Read 7-dim arm joint positions from a robolab obs dict."""
        if obs is None:
            return self._current_joints.copy()
        proprio = obs.get("proprio_obs") or {}
        joints = proprio.get("arm_joint_pos")
        if joints is None:
            return self._current_joints.copy()
        return joints[0].detach().cpu().numpy().astype(np.float64).reshape(7)

    def _read_gripper_pos(self, obs: dict[str, Any] | None) -> float:
        if obs is None:
            return 0.0
        proprio = obs.get("proprio_obs") or {}
        gp = proprio.get("gripper_pos")
        if gp is None:
            return 0.0
        return float(gp[0].detach().cpu().numpy().reshape(-1)[0])

    def _read_ee_pose(self, obs: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
        if obs is None:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        proprio = obs.get("proprio_obs") or {}
        ee_pos = proprio.get("ee_pos")
        ee_quat = proprio.get("ee_quat")
        if ee_pos is None or ee_quat is None:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        return (
            ee_pos[0].detach().cpu().numpy().astype(np.float64).reshape(3),
            ee_quat[0].detach().cpu().numpy().astype(np.float64).reshape(4),
        )

    def _refresh_gt_state(self) -> dict[str, Any]:
        gt_state = self._gt_exporter.export(self._latest_obs)
        self._latest_gt_state = gt_state
        return gt_state

    def _dump_gt_state(self, gt_state: dict[str, Any] | None) -> None:
        if self._gt_state_dump_fp is None or gt_state is None:
            return
        record = {
            "step": self._sim_step_count,
            "instruction": self.env_cfg.instruction,
            "task_completed": bool(self._latest_term),
            "gt_state": _jsonable_gt_state(gt_state),
        }
        self._gt_state_dump_fp.write(json.dumps(record) + "\n")
        self._gt_state_dump_fp.flush()

    # ------------------------------------------------------------------
    # Object-pose hooks (called by FrankaRobolabPrivilegedApi)
    # ------------------------------------------------------------------

    def _get_object_pose(self, obj_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (position (3,), quat_wxyz (4,)) for a named scene object.

        Pulled from the GT state exporter, which already gives env-relative
        positions and Isaac-Lab-convention (wxyz) quaternions.
        """
        gt = self._latest_gt_state or self._refresh_gt_state()
        objects = gt.get("objects", {})
        if obj_name in objects:
            entry = objects[obj_name]
        else:
            # Fuzzy match — strip suffixes / case
            query = obj_name.lower().replace(" ", "_")
            match = next(
                (k for k in objects if query in k.lower() or k.lower() in query),
                None,
            )
            if match is None:
                raise KeyError(
                    f"Object '{obj_name}' not found. Available: {sorted(objects.keys())}"
                )
            entry = objects[match]
        pos = np.asarray(entry["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(entry["quat"], dtype=np.float64).reshape(4)
        return pos, quat

    def _get_all_object_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        gt = self._latest_gt_state or self._refresh_gt_state()
        objects = gt.get("objects", {})
        return {
            name: (
                np.asarray(entry["pos"], dtype=np.float64).reshape(3),
                np.asarray(entry["quat"], dtype=np.float64).reshape(4),
            )
            for name, entry in objects.items()
        }

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """Build a CaP-X-style observation dict from the latest robolab obs.

        Keys mirror FrankaLiberoEnv so the privileged API stays portable.
        """
        obs: dict[str, Any] = {}
        latest = self._latest_obs or {}
        image_obs = latest.get("image_obs") or {}
        viewport = latest.get("viewport_cam") or {}

        # Primary external camera → "agentview"
        ext_rgb = _maybe_first_env(image_obs.get("external_cam"))
        ext_depth = _maybe_first_env(image_obs.get("external_cam_depth"))
        agentview: dict[str, Any] = {"images": {}}
        if ext_rgb is not None:
            agentview["images"]["rgb"] = ext_rgb
        if ext_depth is not None:
            agentview["images"]["depth"] = ext_depth
        intrinsics, pose_mat = self._read_camera_calib("external_cam")
        if intrinsics is not None:
            agentview["intrinsics"] = intrinsics
        if pose_mat is not None:
            agentview["pose_mat"] = pose_mat
        obs["agentview"] = agentview

        # Wrist camera → "robot0_eye_in_hand"
        wrist_rgb = _maybe_first_env(image_obs.get("wrist_cam"))
        # Wrist depth is added at runtime by ``_bootstrap_isaac_sim``.
        wrist_depth = _maybe_first_env(image_obs.get("wrist_cam_depth"))
        wrist: dict[str, Any] = {"images": {}}
        if wrist_rgb is not None:
            wrist["images"]["rgb"] = wrist_rgb
        if wrist_depth is not None:
            wrist["images"]["depth"] = wrist_depth
        intrinsics_w, pose_mat_w = self._read_camera_calib("wrist_cam")
        if intrinsics_w is not None:
            wrist["intrinsics"] = intrinsics_w
        if pose_mat_w is not None:
            wrist["pose_mat"] = pose_mat_w
        obs["robot0_eye_in_hand"] = wrist

        # Front (top-down) camera if available — exposed under the same
        # key the orchestrator's --use-front-camera flag uses.
        front_rgb = _maybe_first_env(viewport.get("egocentric_mirrored_camera"))
        if front_rgb is not None:
            front: dict[str, Any] = {"images": {"rgb": front_rgb}}
            front_depth = _maybe_first_env(
                viewport.get("egocentric_mirrored_camera_depth")
            )
            if front_depth is not None:
                front["images"]["depth"] = front_depth
            obs["egocentric_mirrored_camera"] = front

        # Robot proprioception
        ee_pos, ee_quat = self._read_ee_pose(latest)
        gripper_pos = self._read_gripper_pos(latest)
        joints = self._read_arm_joints(latest)
        # 0=open, 1=closed in robolab; CaP-X expects 0=closed, 1=open.
        gripper_normalised_capx = 1.0 - float(np.clip(gripper_pos, 0.0, 1.0))
        obs["robot_joint_pos"] = np.concatenate([joints, [gripper_normalised_capx]])
        obs["robot_cartesian_pos"] = np.concatenate(
            [ee_pos, ee_quat, [gripper_normalised_capx]]
        )
        return obs

    def _read_camera_calib(self, sensor_name: str):
        """Best-effort intrinsics (3,3) and pose_mat (4,4) for a scene camera.

        ``pose_mat`` is returned in **OpenCV convention** (X right, Y down,
        Z forward into the image) so it can be applied directly to points
        deprojected via ``(x_cam = (u - cx) * z / fx, y_cam = (v - cy) * z / fy,
        z_cam = z)``. Robolab cameras report ``quat_w_opengl`` (OpenGL: Y up,
        Z backward), so we right-multiply by a 180-deg flip of Y and Z to
        convert. Matches LIBERO's ``cam_robot_tf`` chain
        (``rpy(0, pi, 0) @ rpy(0, 0, pi) == diag(1, -1, -1)``).

        Returns (None, None) if the camera doesn't exist or its data isn't
        ready yet (e.g. before the first physics step).
        """
        try:
            cam = self.handle.scene[sensor_name]
        except (KeyError, AttributeError):
            return None, None
        intrinsics = None
        pose_mat = None
        try:
            if hasattr(cam.data, "intrinsic_matrices"):
                intrinsics = (
                    cam.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
                )
        except Exception:
            intrinsics = None
        try:
            pos = cam.data.pos_w[0].detach().cpu().numpy().astype(np.float64)
            quat = cam.data.quat_w_opengl[0].detach().cpu().numpy().astype(np.float64)
            pose_mat_gl = _wxyz_pos_to_matrix(quat, pos)
            pose_mat = pose_mat_gl @ _OPENGL_TO_OPENCV_LOCAL
        except Exception:
            pose_mat = None
        return intrinsics, pose_mat

    # ------------------------------------------------------------------
    # BaseEnv abstracts
    # ------------------------------------------------------------------

    def compute_reward(self) -> float:
        return float(self._latest_reward)

    def task_completed(self) -> bool:
        return bool(self._latest_term)

    def get_current_time_s(self) -> float:
        return self._sim_step_count / self._control_freq

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        rgb = _maybe_first_env(
            (self._latest_obs or {}).get("image_obs", {}).get("external_cam")
        )
        if rgb is None:
            raise RuntimeError("No external camera image available to render")
        return rgb

    def render_wrist(self) -> np.ndarray:
        rgb = _maybe_first_env(
            (self._latest_obs or {}).get("image_obs", {}).get("wrist_cam")
        )
        if rgb is None:
            raise RuntimeError("No wrist camera image available to render")
        return rgb

    # --- Video API (cap-x trial.py probes these via hasattr) ---
    # Frames are appended to ``_frame_buffer`` during steps in ``reset``,
    # ``_step_once``, and ``move_to_joints_blocking`` (subsampled by
    # ``video_subsample``). cap-x's ``CodeExecutionEnvBase.get_video_frames``
    # surfaces them; ``_save_trial_video`` then writes an MP4 inside the
    # per-trial output dir alongside code.py / summary.txt etc.

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        self._record_video = bool(enabled)
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()

    def get_video_frames(self, *, clear: bool = False) -> list:
        frames = [f.copy() for f in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list:
        return [f.copy() for f in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list:
        frames = [f.copy() for f in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list:
        return [f.copy() for f in self._wrist_frame_buffer[start:end]]

    def close(self) -> None:
        if self._gt_state_dump_fp is not None:
            try:
                self._gt_state_dump_fp.close()
            finally:
                self._gt_state_dump_fp = None
        self._close_native_video_writer()
        if hasattr(self.handle, "close"):
            try:
                self.handle.close()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


# OpenGL → OpenCV camera-frame conversion: flip Y and Z axes of the
# local camera frame. Right-multiply onto an OpenGL cam-to-world matrix
# to get an OpenCV cam-to-world matrix that the standard pinhole
# deprojection (x_cam = (u-cx)z/fx, y_cam = (v-cy)z/fy, z_cam = z) can
# round-trip cleanly.
_OPENGL_TO_OPENCV_LOCAL = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

_ISAAC_SIM_APP = None
_ROBOLAB_TASKS_REGISTERED: set[str] = set()
_DEFAULT_ROBOLAB_TASK_DIRS = (
    "long_horizon/common_sense",
)
_PARSE_ENV_CFG_PATCHED = False
_TRIAL_VIDEO_DIR_PATCHED = False


def _patch_trial_video_dir() -> None:
    """Make cap-x write each trial's artifacts to ONE stable dir.

    Two functions in cap-x build per-trial dir paths from
    ``(sandbox_rc, reward, taskcompleted)`` — all mutable across code
    blocks. Since ``_save_trial_artifacts`` is invoked after every
    block, each state change creates a NEW directory, leaving 2–3
    duplicate folders per trial.

    We replace both with stable ``trial_<NN>`` names. Outcome
    information stays available in the per-trial ``metadata.json``
    (written by the offline scorer), the ``summary.txt`` log lines,
    and ``all_responses.json``.

    Idempotent.
    """
    global _TRIAL_VIDEO_DIR_PATCHED
    if _TRIAL_VIDEO_DIR_PATCHED:
        return
    from pathlib import Path
    import json as _json
    from capx.envs import trial as _capx_trial
    from capx.utils import launch_utils as _capx_launch_utils

    def _stable_trial_video_dir(config, trial, info_step, reward):
        return os.path.join(config["output_dir"], f"trial_{int(trial):02d}")

    _capx_trial._trial_video_dir = _stable_trial_video_dir

    # Re-implement ``_save_trial_artifacts`` — same behaviour as the
    # upstream version but with a stable per-trial dir name. We keep
    # this in lockstep with the original (capx/utils/launch_utils.py).
    def _stable_save_trial_artifacts(
        config,
        trial,
        sandbox_rc,
        reward,
        task_completed,
        final_code,
        raw_code,
        all_responses,
        log_lines,
        visual_feedback_imgs,
        ensemble_data=None,
        multiturn_ensemble_data=None,
    ):
        if not config["output_dir"]:
            return None
        trial_dir = Path(config["output_dir"]) / f"trial_{int(trial):02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        code_path_obj = trial_dir / "code.py"
        code_path_obj.write_text(final_code)
        code_path = str(code_path_obj)

        if raw_code:
            (trial_dir / "raw_response.sh").write_text(raw_code)

        (trial_dir / "all_responses.json").write_text(
            _json.dumps(all_responses, indent=2)
        )
        (trial_dir / "summary.txt").write_text("\n".join(log_lines))

        if ensemble_data:
            if ensemble_data.get("ensemble_candidates_txt"):
                (trial_dir / "ensemble_candidates.txt").write_text(
                    ensemble_data["ensemble_candidates_txt"]
                )
            if ensemble_data.get("ensemble_synthesis_txt"):
                (trial_dir / "ensemble_synthesis.txt").write_text(
                    ensemble_data["ensemble_synthesis_txt"]
                )

        if multiturn_ensemble_data:
            for entry in multiturn_ensemble_data:
                regen_num = entry.get("regeneration", 0)
                if entry.get("ensemble_candidates_txt"):
                    (trial_dir / f"ensemble_candidates_regen_{regen_num:02d}.txt").write_text(
                        entry["ensemble_candidates_txt"]
                    )
                if entry.get("ensemble_synthesis_txt"):
                    (trial_dir / f"ensemble_synthesis_regen_{regen_num:02d}.txt").write_text(
                        entry["ensemble_synthesis_txt"]
                    )

        i = 0
        (trial_dir / "prompts_and_responses").mkdir(parents=True, exist_ok=True)
        for response in all_responses:
            if "task_seg_description" in response:
                (trial_dir / "prompts_and_responses" / "task_seg_description.txt").write_text(
                    response["task_seg_description"]
                )
            if "task_seg_prompt" in response:
                (trial_dir / "prompts_and_responses" / "task_seg_prompt.txt").write_text(
                    str(response["task_seg_prompt"])
                )
            if "initial_prompt" in response:
                try:
                    initial_prompt_content = response["initial_prompt"][-1]["content"][0]["text"]
                    if isinstance(initial_prompt_content, list):
                        initial_prompt_content = "\n".join(map(str, initial_prompt_content))
                    (trial_dir / "prompts_and_responses" / "initial_prompt.txt").write_text(
                        str(initial_prompt_content)
                    )
                except Exception as e:
                    print(f"Error saving initial_prompt: {e}")
            if "multi_turn_prompt" in response and response["multi_turn_prompt"] is not None:
                try:
                    multi_turn_prompt_content = response["multi_turn_prompt"][-1]["content"][0]["text"]
                    if isinstance(multi_turn_prompt_content, list):
                        multi_turn_prompt_content = "\n".join(map(str, multi_turn_prompt_content))
                    (trial_dir / "prompts_and_responses" / f"multi_turn_prompt_{i:02d}.txt").write_text(
                        str(multi_turn_prompt_content)
                    )
                except Exception as e:
                    print(f"Error saving multi_turn_prompt_{i}: {e}")
                i += 1
        for i, img in enumerate(visual_feedback_imgs):
            img.save(trial_dir / f"visual_feedback_{i:02d}.png")

        return code_path

    _capx_launch_utils._save_trial_artifacts = _stable_save_trial_artifacts
    # The trial module pulled the original at import time — patch its
    # local binding so cap-x calls our stable version.
    _capx_trial._save_trial_artifacts = _stable_save_trial_artifacts

    _TRIAL_VIDEO_DIR_PATCHED = True


def _patch_parse_env_cfg_for_wrist_depth() -> None:
    """Wrap ``robolab.core.environments.config.parse_env_cfg`` so that
    every parsed env_cfg gets a ``wrist_cam_depth`` ObsTerm injected.

    Robolab ships ``external_cam_depth`` only; cap-x's S2/S3/S4
    perception tiers want depth on both cameras for multiview pointcloud
    fusion. We can't add the field at class level (``@configclass``
    freezes fields), but the dataclass instance accepts ``setattr`` so
    we add it after parsing, before the env is constructed.

    Idempotent.
    """
    global _PARSE_ENV_CFG_PATCHED
    if _PARSE_ENV_CFG_PATCHED:
        return

    from robolab.core.environments import config as _robolab_cfg
    from robolab.policies.droid_jointpos.observations import image_safe
    from isaaclab.managers import ObservationTermCfg as _ObsTerm
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg

    original_parse = _robolab_cfg.parse_env_cfg

    def patched(*args, **kwargs):
        env_cfg = original_parse(*args, **kwargs)
        try:
            image_obs = getattr(env_cfg.observations, "image_obs", None)
            if image_obs is not None and not hasattr(image_obs, "wrist_cam_depth"):
                image_obs.wrist_cam_depth = _ObsTerm(
                    func=image_safe,
                    params={
                        "sensor_cfg": _SceneEntityCfg("wrist_cam"),
                        "data_type": "depth",
                        "normalize": False,
                    },
                )
        except Exception:
            # Don't let our patch break any env that doesn't have an
            # ImageObsCfg-shaped observations group.
            pass

        return env_cfg

    _robolab_cfg.parse_env_cfg = patched
    # ``runtime.py`` already imported ``parse_env_cfg`` at module load,
    # binding the name in its own namespace. Update that binding too so
    # ``create_env`` calls our patched version.
    try:
        from robolab.core.environments import runtime as _robolab_runtime
        _robolab_runtime.parse_env_cfg = patched
    except ImportError:
        pass
    _PARSE_ENV_CFG_PATCHED = True


def _bootstrap_isaac_sim(
    task_dirs: tuple[str, ...] = _DEFAULT_ROBOLAB_TASK_DIRS,
) -> None:
    """Start Isaac Sim AppLauncher and auto-register robolab LH tasks.

    Robolab's modules import ``isaaclab.utils`` at import time, which only
    succeeds after AppLauncher has booted Isaac Sim. CaP-X's main process
    doesn't do this — without it, simply importing the env adapter fails
    with ``ModuleNotFoundError: isaaclab.utils``.

    LH common-sense tasks aren't in robolab's ``DEFAULT_TASK_SUBFOLDERS``,
    so we also explicitly register them so ``create_env(scene=<id>)``
    succeeds. Idempotent: AppLauncher reuse and per-subdir registration
    are both no-ops on second call.
    """
    global _ISAAC_SIM_APP
    if _ISAAC_SIM_APP is None:
        from isaaclab.app import AppLauncher

        _ISAAC_SIM_APP = AppLauncher(headless=True, enable_cameras=True).app

    # Register requested robolab task subdirs once each.
    todo = [d for d in task_dirs if d not in _ROBOLAB_TASKS_REGISTERED]
    if todo:
        from robolab.policies.droid_jointpos.auto_env_registrations import (
            auto_register_droid_envs,
        )

        auto_register_droid_envs(task_dirs=list(todo))
        _ROBOLAB_TASKS_REGISTERED.update(todo)


def _to_scalar(x: Any) -> Any:
    """Reduce a 1-element tensor / array / scalar to a python scalar."""
    if x is None:
        return 0
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(x, "shape") and getattr(x, "size", 1) == 1:
        return x.reshape(-1)[0]
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def _maybe_first_env(t: Any) -> np.ndarray | None:
    """Take env_id=0 from a [num_envs, ...] tensor / array, returning numpy."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    arr = np.asarray(t)
    if arr.ndim == 0:
        return None
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _wxyz_pos_to_matrix(quat_wxyz: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Build a 4×4 SE(3) matrix from a wxyz quaternion + position."""
    w, x, y, z = quat_wxyz
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        rot = np.eye(3)
    else:
        s = 2.0 / n
        wx, wy, wz = s * w * x, s * w * y, s * w * z
        xx, xy, xz = s * x * x, s * x * y, s * x * z
        yy, yz, zz = s * y * y, s * y * z, s * z * z
        rot = np.array(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ]
        )
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat


def _jsonable_gt_state(gt: dict[str, Any]) -> dict[str, Any]:
    """Convert the gt_state dict (with numpy arrays) to a JSON-serialisable dict."""

    def _conv(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {k: _conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_conv(x) for x in v]
        return v

    return _conv(gt)


class FrankaRobolabTask(FrankaRobolabEnv):
    """Generic robolab LH task — pick any scene by name from a YAML config.

    Example::

        env:
          _target_: capx.envs.simulators.robolab.FrankaRobolabTask
          scene_name: SortFoodVsNonFoodTask
          instruction_type: default
    """

    def __init__(
        self,
        scene_name: str = "SortFoodVsNonFoodTask",
        max_steps: int = 4000,
        seed: int | None = None,
        device: str = "cuda:0",
        instruction_type: str = "default",
        gt_state_dump_dir: str | None = None,
        gt_state_dump_path: str | None = None,  # back-compat
        record_video: bool = True,
        video_subsample: int = 4,
        # Accept (and ignore) flags the launcher passes for parity with
        # FrankaLiberoTask, even though they don't apply here.
        privileged: bool = True,
        enable_render: bool = False,
        viser_debug: bool = False,
    ) -> None:
        super().__init__(
            scene_name=scene_name,
            max_steps=max_steps,
            seed=seed,
            device=device,
            instruction_type=instruction_type,
            gt_state_dump_dir=gt_state_dump_dir,
            gt_state_dump_path=gt_state_dump_path,
            record_video=record_video,
            video_subsample=video_subsample,
        )


__all__ = ["FrankaRobolabEnv", "FrankaRobolabTask"]
