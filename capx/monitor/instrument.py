"""Render annotated tool outputs into the CaP-X dashboard by runtime patching.

``capx/integrations/franka/control_reduced.py`` is shared with the sim/robolab
path, so it is not edited. Instead the real-robot service calls
``instrument_tools()`` at startup, which wraps the two tools whose visual output
is missing:

``detect_objects``  (GDino / OWL-ViT) logs the RAW rgb; the predicted boxes are
                    never drawn, so a box on the wrong object is invisible.
``plan_grasp``      (Contact GraspNet) logs no image at all, so there is no way
                    to see where the candidates landed or which was chosen.

Everything is best-effort: a rendering failure must never affect the robot.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

_installed = False


def instrument_tools() -> None:
    """Wrap detection + grasp planning so they publish annotated images."""
    global _installed
    if _installed:
        return
    # The real-robot config uses FrankaRealReducedSkillLibrary, which is
    # FrankaControlApiReducedSkillLibrary(real=True) — the class lives in
    # control_reduced_skill_library, NOT control_reduced. Getting this wrong
    # makes every wrapper silently no-op.
    try:
        from capx.integrations.franka.control_reduced_skill_library import (
            FrankaControlApiReducedSkillLibrary as Api,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[monitor] tool instrumentation unavailable: {exc}", flush=True)
        return

    wrapped = _wrap_detection(Api) + _wrap_grasp(Api)
    if not wrapped:
        print(
            "[monitor] WARNING: no tool methods were wrapped — the API surface "
            "changed; dashboard tool images will be missing",
            flush=True,
        )
    else:
        print(f"[monitor] annotated tool views: {', '.join(wrapped)}", flush=True)
    _installed = True


def _wrap_detection(Api) -> list[str]:
    """Draw GDino / OWL-ViT boxes."""
    done = []
    for name in ("detect_object_owlvit", "detect_objects", "detect_gdino"):
        fn = getattr(Api, name, None)
        if fn is None:
            continue
        orig = fn

        def wrapped(self, *a, _orig=orig, **kw):
            out = _orig(self, *a, **kw)
            try:
                rgb = _first_image(a, kw)
                if rgb is not None and isinstance(out, (list, tuple)):
                    from capx.monitor import hooks as mon
                    from capx.monitor import viz

                    img = viz.detection_boxes(rgb, out)
                    if img:
                        mon.tool_image(
                            "Detection boxes",
                            f"{len(out)} detection(s)"
                            + (
                                f", best score {max(d.get('score', 0) for d in out):.3f}"
                                if out
                                else " — NOTHING DETECTED"
                            ),
                            img,
                            is_error=not out,
                        )
            except Exception:
                pass
            return out

        setattr(Api, name, wrapped)
        done.append(name)
    return done


def _wrap_grasp(Api) -> list[str]:
    """Draw Contact GraspNet candidates and the chosen grasp."""
    done = []
    for name in ("plan_grasp", "select_top_down_grasp"):
        fn = getattr(Api, name, None)
        if fn is None:
            continue
        orig = fn

        def wrapped(self, *a, _orig=orig, **kw):
            t0 = time.perf_counter()
            out = _orig(self, *a, **kw)
            try:
                _report_grasps(self, a, kw, time.perf_counter() - t0)
            except Exception:
                pass
            return out

        setattr(Api, name, wrapped)
        done.append(name)
    return done


def _report_grasps(api, args, kwargs, seconds: float) -> None:
    """Publish grasp candidates + the chosen one, drawn on the RGB."""
    from capx.monitor import hooks as mon
    from capx.monitor import viz

    env = getattr(api, "_env", None)
    poses = getattr(env, "grasp_sample", None) if env is not None else None
    scores = getattr(env, "grasp_scores", None) if env is not None else None

    if poses is None or not len(np.atleast_3d(np.asarray(poses))):
        mon.tool_image(
            "Contact GraspNet",
            "NO GRASP CANDIDATES returned — the masked point cloud was probably "
            "empty or too small. Check the segmentation mask.",
            None,
            is_error=True,
        )
        return

    P = np.asarray(poses, dtype=np.float64)
    if P.ndim == 2:
        P = P[None]
    sc = np.asarray(scores).reshape(-1) if scores is not None else None
    chosen = int(np.argmax(sc)) if sc is not None and len(sc) == len(P) else None

    rgb, K = _rgb_and_intrinsics(api, args, kwargs)
    img = (
        viz.grasp_candidates(rgb, P, K, sc, chosen)
        if rgb is not None and K is not None
        else None
    )
    best = f", best score {float(sc[chosen]):.3f}" if chosen is not None else ""
    mon.tool_image(
        "Contact GraspNet",
        f"{len(P)} grasp candidate(s) in {seconds:.2f}s{best}",
        img,
    )


def _first_image(args, kwargs):
    for v in list(args) + list(kwargs.values()):
        a = np.asarray(v) if not isinstance(v, (str, bytes)) else None
        if a is not None and a.ndim == 3 and a.shape[-1] == 3:
            return a
    return None


def _rgb_and_intrinsics(api, args, kwargs):
    """Best-effort: the live observation carries both."""
    rgb = _first_image(args, kwargs)
    K = None
    for v in list(args) + list(kwargs.values()):
        try:
            a = np.asarray(v)
            if a.shape == (3, 3):
                K = a
        except Exception:
            continue
    if rgb is None or K is None:
        try:
            obs = api._env.get_observation()
            cam = obs["robot0_robotview"]
            rgb = rgb if rgb is not None else cam["images"]["rgb"]
            K = K if K is not None else np.asarray(cam["intrinsics"])
        except Exception:
            pass
    return rgb, K
