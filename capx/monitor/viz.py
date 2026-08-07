"""Annotated tool-output images for the CaP-X dashboard.

CaP-X already overlays some tool results before logging them
(``overlay_segmentation_masks`` for SAM2/SAM3, ``draw_molmo_point`` for Molmo).
Two of the most failure-prone tools log only the RAW rgb, or no image at all:

``OWL-ViT / GDino detection``  logs the raw rgb; the predicted boxes are never
                              drawn, so a box on the wrong object is invisible.
``Contact GraspNet``          logs no image at all, so there is no way to see
                              WHERE the candidate grasps landed or which one was
                              chosen — exactly the failure we spent hours
                              diagnosing by hand on the TiPToP side.

This module renders those, plus a generic depth heat-map, and is wired in by
``capx.monitor.instrument`` at runtime so no shared CaP-X file changes.

Every function returns a base64 PNG and never raises.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_PALETTE = [
    (255, 64, 64),
    (64, 255, 64),
    (64, 160, 255),
    (255, 220, 0),
    (255, 64, 255),
    (0, 255, 220),
    (255, 150, 60),
    (170, 120, 255),
]


def _b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _as_pil(rgb):
    from PIL import Image

    a = np.asarray(rgb)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.dtype != np.uint8:
        a = (
            (a * 255).clip(0, 255).astype(np.uint8)
            if float(np.nanmax(a)) <= 1.0
            else a.astype(np.uint8)
        )
    return Image.fromarray(a[:, :, :3])


def _fit(img, max_w: int = 900):
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    return img


def detection_boxes(rgb, detections: Sequence[dict]) -> str | None:
    """Draw GDino / OWL-ViT boxes with labels and scores.

    ``detections`` entries carry ``box`` as [x1, y1, x2, y2] in PIXELS (unlike
    TiPToP's normalised 0-1000 ``box_2d``), plus ``label`` and ``score``.
    """
    try:
        from PIL import ImageDraw

        img = _as_pil(rgb)
        d = ImageDraw.Draw(img)
        n = 0
        for i, det in enumerate(detections or []):
            box = det.get("box") or det.get("bbox")
            if box is None or len(box) != 4:
                continue
            col = _PALETTE[i % len(_PALETTE)]
            x1, y1, x2, y2 = [float(v) for v in box]
            d.rectangle([x1, y1, x2, y2], outline=col, width=3)
            tag = str(det.get("label", "?"))
            if det.get("score") is not None:
                tag += f" {float(det['score']):.2f}"
            d.text((x1 + 4, max(0, y1 - 14)), tag, fill=col)
            n += 1
        d.text((6, 6), f"detections: {n}", fill=(255, 255, 255))
        return _b64(_fit(img))
    except Exception:
        logger.debug("detection_boxes failed", exc_info=True)
        return None


def grasp_candidates(
    rgb,
    grasp_poses,
    intrinsics,
    scores=None,
    chosen_index: int | None = None,
    cam_from_world=None,
) -> str | None:
    """Project ContactGraspNet candidates into the image.

    ``grasp_poses`` is (N, 4, 4). They come out of ContactGraspNet in the CAMERA
    frame, so no extrinsics are needed unless ``cam_from_world`` is given (i.e.
    the poses have already been transformed to world).

    Colour = confidence (dim -> bright). The chosen grasp gets a white ring and
    its approach axis, which is what actually gets sent to IK.
    """
    try:
        from PIL import ImageDraw

        T = np.asarray(grasp_poses, dtype=np.float64)
        if T.ndim == 2 and T.shape == (4, 4):
            T = T[None]
        if T.ndim != 3 or T.shape[1:] != (4, 4) or len(T) == 0:
            return None
        K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        pts = T[:, :3, 3]
        if cam_from_world is not None:
            hom = np.c_[pts, np.ones(len(pts))]
            pts = (np.asarray(cam_from_world) @ hom.T).T[:, :3]
        front = pts[:, 2] > 1e-6
        if not front.any():
            return None
        uv = (K @ pts[front].T).T
        uv = uv[:, :2] / uv[:, 2:3]

        sc = None
        if scores is not None and len(np.asarray(scores).reshape(-1)) == len(T):
            sc = np.asarray(scores, dtype=np.float64).reshape(-1)[front]

        img = _as_pil(rgb)
        d = ImageDraw.Draw(img)
        if sc is not None and len(sc) and float(sc.max() - sc.min()) > 1e-9:
            norm = (sc - sc.min()) / (sc.max() - sc.min())
        else:
            norm = np.ones(len(uv))
        for (u, v), t in zip(uv, norm):
            g = int(70 + 185 * float(t))
            d.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(g, g, 40))

        if chosen_index is not None and 0 <= chosen_index < len(T):
            Tc = T[chosen_index]
            origin = Tc[:3, 3]
            if cam_from_world is not None:
                origin = (np.asarray(cam_from_world) @ np.r_[origin, 1.0])[:3]
            if origin[2] > 1e-6:
                o = (K @ origin)[:2] / (K @ origin)[2]
                d.ellipse(
                    [o[0] - 8, o[1] - 8, o[0] + 8, o[1] + 8],
                    outline=(255, 255, 255),
                    width=3,
                )
                approach = Tc[:3, 2]
                tip = Tc[:3, 3] + 0.05 * approach
                if cam_from_world is not None:
                    tip = (np.asarray(cam_from_world) @ np.r_[tip, 1.0])[:3]
                if tip[2] > 1e-6:
                    tp = (K @ tip)[:2] / (K @ tip)[2]
                    d.line([o[0], o[1], tp[0], tp[1]], fill=(80, 160, 255), width=3)
                d.text((o[0] + 12, o[1] - 6), "CHOSEN", fill=(255, 255, 255))

        d.text((6, 6), f"grasp candidates: {len(uv)}", fill=(255, 255, 255))
        if sc is not None and len(sc):
            d.text(
                (6, 24),
                f"score {sc.min():.3f}..{sc.max():.3f} (brighter = higher)",
                fill=(255, 255, 255),
            )
        if chosen_index is not None:
            d.text((6, 42), "white ring = chosen; blue = approach axis", fill=(80, 160, 255))
        return _b64(_fit(img))
    except Exception:
        logger.debug("grasp_candidates failed", exc_info=True)
        return None


def depth_heatmap(depth) -> str | None:
    """Near = bright, far = dark, invalid = black, with the valid fraction."""
    try:
        from PIL import Image, ImageDraw

        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 3:
            d = d[:, :, 0]
        if d.ndim != 2:
            return None
        finite = np.isfinite(d) & (d > 0)
        norm = np.zeros_like(d)
        if finite.any():
            lo, hi = float(d[finite].min()), float(d[finite].max())
            if hi > lo:
                norm[finite] = 1.0 - (d[finite] - lo) / (hi - lo)
        heat = np.stack(
            [norm * 255, norm * 160, np.where(finite, 40, 0)], axis=-1
        ).astype(np.uint8)
        img = Image.fromarray(heat)
        dr = ImageDraw.Draw(img)
        txt = f"valid depth: {100.0 * finite.mean():.1f}%"
        if finite.any():
            txt += f"   {float(d[finite].min()):.2f}-{float(d[finite].max()):.2f} m"
        dr.text((6, 6), txt, fill=(255, 255, 255))
        return _b64(_fit(img))
    except Exception:
        logger.debug("depth_heatmap failed", exc_info=True)
        return None
