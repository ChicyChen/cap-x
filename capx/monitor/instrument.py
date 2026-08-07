"""Annotated tool views for the CaP-X dashboard.

WHY THIS FILE NO LONGER PATCHES TOOL METHODS
--------------------------------------------
The obvious approach -- wrap ``plan_grasp`` / ``detect_object_owlvit`` to render
their outputs -- is **unsafe here and was reverted twice on the real robot**.

``ApiBase.functions()`` captures **bound methods** at env-construction time and
hands them to the generated code::

    fns["plan_grasp"] = self.plan_grasp            # control_reduced.py:104
    fns["detect_object_owlvit"] = self.detect_object_owlvit
    fns["select_top_down_grasp"] = self.select_top_down_grasp

If instrumentation wraps such a method, the captured object is the wrapper
*already bound to self*. Calling it re-passes ``self`` as the first positional
argument and every argument shifts by one::

    TypeError: select_top_down_grasp() missing 2 required positional
               arguments: 'scores' and 'cam_to_world'
    TypeError: plan_grasp() missing 2 required positional
               arguments: 'intrinsics' and 'segmentation'

Both were live failures. Note also that the exposing ``functions()`` lives in the
PARENT class (``FrankaControlApiReduced``), so checking only the skill-library
subclass is not enough -- that mistake caused the second failure.

WHAT WE DO INSTEAD
------------------
``capx.monitor.hooks.install_tool_hook`` already wraps ``ApiBase._log_step`` and
``ApiBase._log_step_update``, which are internal (never handed to generated code)
and which every tool already calls. ``annotate_step`` below is invoked from that
hook: given a tool name plus the images CaP-X was already logging, it adds the
annotations CaP-X does not draw itself -- detection boxes and grasp candidates --
using state read from the env.

Nothing here changes any shared CaP-X file, and nothing patches a method that the
generated code can reach.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Set by hooks when an api instance logs a step, so renderers can read env state
#: (grasp poses/scores, intrinsics) without patching any tool method.
_LAST_API: dict[str, Any] = {}


def instrument_tools() -> None:
    """Kept for call-site compatibility; deliberately a no-op.

    See the module docstring: patching tool methods breaks the generated code
    because ``functions()`` hands out bound methods. Annotation happens in
    ``annotate_step``, driven by the safe ``_log_step`` hook.
    """
    print(
        "[monitor] annotated views via the _log_step hook "
        "(tool methods are NOT patched -- functions() hands them to generated "
        "code as bound methods)",
        flush=True,
    )


def note_api(api: Any) -> None:
    """Remember the api instance that is currently logging."""
    _LAST_API["api"] = api


def annotate_step(tool_name: str, images: Any) -> tuple[str | None, str | None]:
    """Return ``(annotated_b64, caption)`` for tools CaP-X does not draw itself.

    ``images`` is whatever the tool passed to ``_log_step``. Returns
    ``(None, None)`` when there is nothing to add, so the caller falls back to the
    image CaP-X supplied.
    """
    try:
        from capx.monitor import viz

        api = _LAST_API.get("api")
        if api is None:
            return None, None

        if "GraspNet" in tool_name or "grasp" in tool_name.lower():
            return _annotate_grasps(api, images, viz)
        if "OWL-ViT" in tool_name or "Detection" in tool_name:
            return _annotate_detections(api, images, viz)
    except Exception:
        logger.debug("annotate_step failed", exc_info=True)
    return None, None


def _annotate_grasps(api, images, viz):
    """Draw Contact GraspNet candidates + the chosen one on the current RGB."""
    env = getattr(api, "_env", None)
    poses = getattr(env, "grasp_sample", None) if env is not None else None
    if poses is None:
        return None, None
    P = np.asarray(poses, dtype=np.float64)
    if P.ndim == 2:
        P = P[None]
    if P.ndim != 3 or len(P) == 0:
        return None, "NO GRASP CANDIDATES returned — the masked point cloud was "\
                     "probably empty. Check the segmentation mask."
    scores = getattr(env, "grasp_scores", None)
    sc = np.asarray(scores).reshape(-1) if scores is not None else None
    chosen = int(np.argmax(sc)) if sc is not None and len(sc) == len(P) else None

    rgb, K = _rgb_and_K(api, images)
    if rgb is None or K is None:
        return None, None
    img = viz.grasp_candidates(rgb, P, K, sc, chosen)
    best = f", best score {float(sc[chosen]):.3f}" if chosen is not None else ""
    return img, f"{len(P)} grasp candidate(s){best}"


def _annotate_detections(api, images, viz):
    """Draw the most recent detection boxes, if the env kept them."""
    env = getattr(api, "_env", None)
    dets = None
    for attr in ("last_detections", "detections", "owlvit_results"):
        dets = getattr(env, attr, None) if env is not None else None
        if dets:
            break
    if not dets:
        return None, None
    rgb, _ = _rgb_and_K(api, images)
    if rgb is None:
        return None, None
    return viz.detection_boxes(rgb, dets), f"{len(dets)} detection(s)"


def _rgb_and_K(api, images):
    """Best-effort RGB + intrinsics: prefer the logged image, else the live obs."""
    rgb = None
    for cand in (images if isinstance(images, (list, tuple)) else [images]):
        try:
            a = np.asarray(cand)
            if a.ndim == 3 and a.shape[-1] == 3:
                rgb = a
                break
        except Exception:
            continue
    K = None
    try:
        obs = api._env.get_observation()
        cam = obs["robot0_robotview"]
        rgb = rgb if rgb is not None else cam["images"]["rgb"]
        K = np.asarray(cam["intrinsics"])
    except Exception:
        pass
    return rgb, K
