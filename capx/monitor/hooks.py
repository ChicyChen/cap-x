"""Bridges CaP-X's existing instrumentation into the monitor bus.

Nothing in CaP-X is modified: tool steps arrive through the callback hook that
``capx.utils.execution_logger`` already exposes, and the APIs already call
``log_step`` for every tool (SAM3, Contact GraspNet, Molmo, IK, gripper, ...).
"""

from __future__ import annotations

import logging
from typing import Any

from .events import BUS, Event, EventKind, publish

logger = logging.getLogger(__name__)

_installed = False
_update_warned = False


def install_tool_hook() -> None:
    """Forward every tool step to the bus. Idempotent.

    Two obstacles in CaP-X, neither of which we modify:

    1. ``ApiBase._log_step`` does ``from capx.utils.execution_logger import
       log_step`` *inside the method*, so patching the module attribute is
       bypassed on every call. We therefore wrap ``ApiBase._log_step`` itself.
    2. ``_log_step`` early-returns unless ``_webui_enabled``, which only CaP-X's
       own web UI sets. Headless runs would emit nothing, so the wrapper
       publishes regardless of that flag (without changing it, so CaP-X's own
       logging behaviour is untouched).
    """
    global _installed
    if _installed:
        return
    try:
        from capx.integrations.base_api import ApiBase
        from capx.utils import execution_logger as el
    except Exception as exc:  # pragma: no cover
        print(f"[monitor] tool hook unavailable: {exc}", flush=True)
        return

    # Unwind any previous install so re-installing cannot stack wrappers (which
    # would publish the same event several times).
    for _name in ("_log_step", "_log_step_update"):
        _saved = getattr(ApiBase, f"__monitor_orig{_name}", None)
        if _saved is not None:
            setattr(ApiBase, _name, _saved)
    ApiBase.__monitor_orig_log_step = ApiBase._log_step
    ApiBase.__monitor_orig_log_step_update = ApiBase._log_step_update

    original = ApiBase._log_step
    last_tool = {"name": "tool result"}

    def _log_step(self, tool_name, text="", images=None, highlight=False):  # type: ignore[no-untyped-def]
        last_tool["name"] = tool_name
        try:
            from capx.monitor import instrument

            instrument.note_api(self)
        except Exception:
            pass
        try:
            original(self, tool_name, text, images=images, highlight=highlight)
        except Exception:
            pass
        try:
            # This is the PRE-RUN log: the raw input frame, before the tool
            # has produced anything. Label it, otherwise it is indistinguishable
            # from the annotated result that follows.
            publish(
                EventKind.TOOL,
                (text or "") + ("  [input frame]" if images is not None else ""),
                data={"tool": tool_name, "annotated": False},
                images=_encode_images(images),
            )
            # Some tools (Contact GraspNet, OWL-ViT) never draw their own
            # result. Render those here rather than by patching the tool
            # methods -- functions() hands those out as BOUND methods, so
            # wrapping them shifts every argument and breaks generated code.
            from capx.monitor import instrument

            extra, caption = instrument.annotate_step(tool_name, images)
            if extra or caption:
                publish(
                    EventKind.TOOL if extra else EventKind.ERROR,
                    (caption or "") + ("  [annotated]" if extra else ""),
                    data={"tool": tool_name, "annotated": bool(extra)},
                    images=[extra] if extra else [],
                )
        except Exception:
            pass

    ApiBase._log_step = _log_step  # type: ignore[assignment]

    # CaP-X's tools log TWICE: _log_step(...) with the RAW rgb before running,
    # then _log_step_update(...) with the ANNOTATED result (Molmo points, SAM2/
    # SAM3 mask overlays). Wrapping only _log_step therefore captured the
    # un-annotated image every time -- which is exactly what the dashboard
    # showed. Wrap the update too and publish it as its own event so the
    # annotated view is what the operator sees.
    original_update = ApiBase._log_step_update

    # Signature must match ApiBase._log_step_update exactly: (text, images).
    # It has no `highlight` parameter, unlike _log_step.
    def _log_step_update(self, text=None, images=None):  # type: ignore[no-untyped-def]
        try:
            original_update(self, text=text, images=images)
        except Exception:
            pass
        try:
            if images is not None:
                # Headless runs have no CaP-X history, so the tool name comes
                # from the _log_step that immediately preceded this update.
                hist = el.get_current_history()
                steps = getattr(hist, "steps", None) if hist else None
                tool = (
                    getattr(steps[-1], "tool_name", None)
                    if steps
                    else None
                ) or last_tool["name"]
                publish(
                    EventKind.TOOL,
                    (text or "") + "  [annotated]",
                    data={"tool": tool, "annotated": True},
                    images=_encode_images(images),
                )
        except Exception:
            # Print once rather than silently dropping every annotated image --
            # a NameError here is exactly how the annotated views went missing.
            global _update_warned
            if not _update_warned:
                _update_warned = True
                import traceback

                print("[monitor] annotated-image publish failed:", flush=True)
                traceback.print_exc()

    ApiBase._log_step_update = _log_step_update  # type: ignore[assignment]
    _installed = True


def _encode_images(images: Any) -> list[str]:
    """Best-effort base64 PNG list. Never raises, never blocks long."""
    if images is None:
        return []
    if not isinstance(images, (list, tuple)):
        images = [images]
    out: list[str] = []
    for im in list(images)[:4]:            # cap: the feed is for humans
        try:
            if isinstance(im, str):
                out.append(im if len(im) > 512 else "")
                continue
            import base64
            import io

            import numpy as np
            from PIL import Image

            arr = np.asarray(im)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, -1)
            if arr.dtype != np.uint8:
                a = arr.astype("float32")
                rng = float(a.max() - a.min()) or 1.0
                arr = (((a - a.min()) / rng) * 255).astype("uint8")
            img = Image.fromarray(arr[:, :, :3])
            img.thumbnail((480, 480))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode())
        except Exception:
            continue
    return [s for s in out if s]


def tool_image(
    tool: str, text: str, image: str | None, is_error: bool = False
) -> None:
    """Publish an annotated tool output. ``is_error`` makes it a red row.

    Used for the tools whose visual result CaP-X does not already overlay:
    detection boxes and Contact GraspNet candidates.
    """
    publish(
        EventKind.ERROR if is_error else EventKind.TOOL,
        text,
        data={"tool": tool},
        images=[image] if image else [],
    )


# ── convenience publishers used by the env / code env ──────────────────

def episode_start(episode: Any, task: str | None = None) -> None:
    publish(EventKind.EPISODE_START, f"episode {episode} started", episode=episode)
    if task:
        # A task means a FRESH plan is starting, so anything still arriving from
        # the previous episode's code is no longer expected: end the superseded
        # window (see EventBus.mark_superseded).
        from .events import BUS

        BUS.clear_superseded(episode)
        publish(EventKind.TASK, task, episode=episode)


def episode_end(episode: Any, reason: str = "") -> None:
    publish(EventKind.EPISODE_END, reason or f"episode {episode} ended", episode=episode)


def attempt(episode: Any, n: int) -> None:
    """A new plan for an episode already running (CaP-X started a new trial)."""
    publish(
        EventKind.NOTE,
        f"re-planning episode {episode} (attempt {n})",
        episode=episode,
        data={"attempt": n},
    )


def task_adopted(task: str, episode: Any = None) -> None:
    publish(EventKind.TASK, task, episode=episode)


def scene(*, rgb=None, depth=None, wrist=None, joints=None) -> None:
    """Publish what the robot sees: RGB, depth heat-map, wrist view."""
    import numpy as np

    images: list[str] = []
    data: dict[str, Any] = {}
    if rgb is not None:
        a = np.asarray(rgb)
        data["rgb_shape"] = list(a.shape)
        images += _encode_images([a])
    if depth is not None:
        d = np.asarray(depth, dtype="float32")
        if d.ndim == 3:
            d = d[:, :, 0]
        finite = np.isfinite(d)
        data["depth_shape"] = list(d.shape)
        data["depth_valid_frac"] = round(float(finite.mean()), 3)
        if finite.any():
            data["depth_min_m"] = round(float(d[finite].min()), 3)
            data["depth_max_m"] = round(float(d[finite].max()), 3)
            # Colourise: near = bright, far = dark, invalid = black.
            lo, hi = float(d[finite].min()), float(d[finite].max())
            norm = np.zeros_like(d)
            if hi > lo:
                norm[finite] = 1.0 - (d[finite] - lo) / (hi - lo)
            heat = np.stack(
                [
                    (norm * 255),
                    (norm * 160),
                    (np.where(finite, 40, 0)),
                ],
                axis=-1,
            ).astype("uint8")
            images += _encode_images([heat])
    if wrist is not None:
        images += _encode_images([np.asarray(wrist)])
    if joints is not None:
        data["joints"] = [round(float(x), 3) for x in np.asarray(joints).reshape(-1)]
    publish(EventKind.OBSERVATION, "scene", data=data, images=images)


def observation(summary: dict[str, Any], images: list[str] | None = None) -> None:
    publish(EventKind.OBSERVATION, "", data=summary, images=images or [])


def model_query(phase: str, seconds: float | None = None) -> None:
    publish(
        EventKind.MODEL_QUERY,
        f"LLM {phase}" + (f" ({seconds:.2f}s)" if seconds else ""),
        data={"phase": phase, "seconds": seconds},
    )


def code_block(code: str, index: int = 0) -> None:
    publish(EventKind.CODE, f"generated code block {index}", data={"code": code})


def motion_done(
    *,
    n_waypoints: int,
    start: list[float] | None = None,
    target: list[float] | None = None,
    residual: float | None = None,
    converged: bool | None = None,
) -> None:
    """One row per completed motion, not per enqueue batch."""
    status = "reached" if converged else "did NOT reach"
    publish(
        EventKind.MOTION,
        f"move {status} target · {n_waypoints} waypoints · residual "
        f"{residual if residual is not None else float('nan'):.4f} rad",
        data={
            "waypoints": n_waypoints,
            "start": start,
            "target": target,
            "residual": residual,
            "converged": converged,
        },
    )


def motion(waypoints: int, target: list[float] | None = None) -> None:
    publish(
        EventKind.MOTION,
        f"enqueued {waypoints} waypoint(s)",
        data={"waypoints": waypoints, "target": target},
    )


def error(text: str, **data: Any) -> None:
    publish(EventKind.ERROR, text, data=data)


def note(text: str, **data: Any) -> None:
    publish(EventKind.NOTE, text, data=data)
