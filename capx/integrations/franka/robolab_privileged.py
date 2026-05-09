"""Privileged Franka API backed by robolab's GT state exporter.

Same callable surface as ``FrankaLiberoPrivilegedApi`` — the only
behavioural difference is where object poses come from (robolab's
``GTStateExporter`` instead of MuJoCo ``sim.data``).  All other methods
delegate to the env adapter, so the agent-facing API docstrings the
LLM sees stay identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.integrations.franka.libero_privileged import FrankaLiberoPrivilegedApi


class FrankaRobolabPrivilegedApi(FrankaLiberoPrivilegedApi):
    """Privileged-perception API for the robolab adapter."""

    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["agentview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["agentview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W), dtype float32 (only present when the camera was configured with depth).
                - ["agentview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["agentview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot0_eye_in_hand"]["images"]["rgb"]: Current wrist camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_eye_in_hand"]["intrinsics"]: Wrist camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_eye_in_hand"]["pose_mat"]: Wrist camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot_cartesian_pos"]: Current end-effector (panda_hand) pose in the robot/world frame as a numpy array of shape (8,), dtype float64. The first 3 elements are the robot's end-effector XYZ, the next 4 elements are the quaternion wxyz, and the last element is the gripper position normalized, 0 (closed) to 1 (open).
                - ["robot_joint_pos"]: Current joint positions as a numpy array of shape (8,), dtype float64. The last element is the gripper position normalized, 0 (closed) to 1 (open).
        """
        obs = self._env.get_observation()
        # The LIBERO privileged API does an unconditional ``squeeze(-1)`` on
        # depth (assumes (H,W,1) shape from robosuite). Robolab returns
        # depth as (H,W) already, and only on cameras configured with
        # depth. Normalize defensively without crashing on missing keys.
        for cam in (self.camera_name, self.wrist_camera_name):
            cam_obs = obs.get(cam, {})
            images = cam_obs.get("images", {})
            depth = images.get("depth")
            if depth is None:
                continue
            arr = np.asarray(depth)
            if arr.ndim >= 1 and arr.shape[-1] == 1:
                images["depth"] = arr.squeeze(-1)
            else:
                images["depth"] = arr
        return obs

    def sample_grasp_pose(self, object_name: str):
        """Top-down grasp at the object's GT centre.

        robolab scenes don't ship a ground-truth grasp annotation, so we
        return a generic gripper-down quaternion (wxyz = [0, 1, 0, 0])
        the same way the LIBERO privileged API does. The agent is
        expected to refine z-approach via ``goto_pose(z_approach=...)``.
        """
        pos, _ = self._env._get_object_pose(object_name)
        return pos, np.array([0.0, 1.0, 0.0, 0.0])
