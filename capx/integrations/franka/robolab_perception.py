"""S2/S3/S4 perception API tiers for the robolab adapter.

The LIBERO classes already encapsulate all the sim-agnostic logic
(SAM3 / Molmo / ContactGraspNet / PyRoKi / depth helpers) — the only
LIBERO-specific assumption is the camera key naming convention
(``"agentview"`` / ``"robot0_eye_in_hand"``), which our env adapter
already mirrors. So the robolab variants are straight subclasses with
the same registered names but qualified for clarity.

These tiers do NOT use GT object poses — they exercise the actual
perception pipeline (SAM3 segmentation, depth-to-pointcloud,
ContactGraspNet, IK), which is what the CaP-Agent0 paper headline
benchmarks.

Server requirements (started by the YAML's ``api_servers`` block):
- SAM3 server on :8114  (S2/S3/S4)
- ContactGraspNet server on :8115  (S3/S4)
- PyRoKi server on :8116  (S1/S2/S3/S4)
"""

from __future__ import annotations

from capx.integrations.franka.libero import FrankaLiberoApi
from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
from capx.integrations.franka.libero_reduced_skill_library import (
    FrankaLiberoApiReducedSkillLibrary,
)


class FrankaRobolabApi(FrankaLiberoApi):
    """S2: high-level perception API (SAM3 + depth-to-3D + IK + gripper)."""


class FrankaRobolabApiReduced(FrankaLiberoApiReduced):
    """S3: atomic perception primitives (SAM3 prompts, ContactGraspNet,
    IK solver, point-cloud manipulation)."""


class FrankaRobolabApiReducedSkillLibrary(FrankaLiberoApiReducedSkillLibrary):
    """S4: S3 + LLM-synthesized utility skill library.

    Matches the strongest CaP-Agent0 baseline in the paper.
    """


__all__ = [
    "FrankaRobolabApi",
    "FrankaRobolabApiReduced",
    "FrankaRobolabApiReducedSkillLibrary",
]
