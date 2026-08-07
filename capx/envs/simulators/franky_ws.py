"""CaP-X low-level env for a ``franky_service`` real Franka cell.

Written for that driver specifically. It deliberately does **not** reuse
``franka_real.py`` (which targets ``robots_realtime``: raw-TCP msgpack on :9000,
different observation keys, and a blocking convergence loop that stalls a
policy-style client).

Nothing in this file is imported by any simulator env, so sim/robolab behaviour
is unaffected.

The transport problem, and the fix
----------------------------------
``franky_service`` is a **policy client**: it sends one observation and expects
an action chunk back promptly, then executes that chunk at a fixed control rate
(``droid_plus/eval/episode_runner.py`` -> ``policy.infer_chunk``). It never waits
for us.

CaP-X is a **closed-loop controller**: its generated Python calls
``move_to_joints_blocking(...)`` and expects the arm to arrive.

Blocking inside the request handler deadlocks the two. So we use the same
trajectory-cursor pattern vlm-orchestrator uses for its grasp/place primitives
(``grasp/tool.py::_step_trajectory``): a motion is interpolated into waypoints
and appended to a buffer; each observation pops the next ``action_horizon``
waypoints; when the buffer runs dry we repeat the last commanded pose so the arm
holds still. The driver is therefore always answered immediately, and CaP-X's
code still observes motions completing.

Wire format (verified against
``franky_service/droid_plus/policies/pi05_with_depth.py``, branch
``hugohadfield/siyi_project``):

===========================================  ==================================
``prompt``                                   task, sent on EVERY frame
``episode_id``                               int (NOT ``__episode_id``)
``observation/exterior_image_1_left``        224x224x3 uint8
``observation/exterior_image_1_left_raw``    native ZED res uint8
``observation/wrist_image_left``             224x224x3 uint8
``observation/joint_position``               (7,) float64 rad
``observation/gripper_position``             (1,) float64, 0=open 1=closed
``observation/depth_external``               native res float32 METRES, NaN-holed
``observation/camera_K``                     (9,) row-major, native res
``observation/camera_extrinsic``             (16,) row-major, cam->BASE, OpenCV
``observation/ee_pos``                       (3,) raw flange position, base frame
``observation/ee_quat``                      (4,) (w, x, y, z)
===========================================  ==================================

Action reply: ``{"actions": (horizon, 8)}``, rows ``[j0..j6, gripper]``,
gripper 0=open 1=closed.

Gripper polarity is inverted at this boundary: the wire uses 0=open/1=closed,
CaP-X uses 0=closed/1=open (matching ``robolab.py``).
"""

from __future__ import annotations

import asyncio
import http
import os
import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from capx.envs.base import BaseEnv

logger = logging.getLogger(__name__)

# ── wire conventions ────────────────────────────────────────────────────
WIRE_GRIPPER_OPEN = 0.0
WIRE_GRIPPER_CLOSED = 1.0

#: Rows per action chunk. Matches the pi0.5/robolab chunk length the driver
#: expects; it does not hard-code a horizon, so this only sets our granularity.
class EpisodeSuperseded(BaseException):
    """The operator started a new episode; the running plan is void.

    Derives from BaseException (not Exception) on purpose: generated code is
    full of ``try/except Exception`` blocks that would swallow an abort and keep
    driving the arm from a plan describing a scene that no longer exists.
    CodeExecutionEnvBase catches BaseException, so the abort is still contained.
    """


IDLE_EPISODE_END_S = float(os.environ.get("CAPX_IDLE_EPISODE_END_S", 12.0))

ACTION_HORIZON = 8

#: Radians per waypoint when interpolating a joint motion. Smaller = smoother
#: and slower. 0.01 rad at ~15 Hz is a conservative real-robot speed.
MAX_JOINT_STEP_RAD = 0.01

_DEPTH_KEYS = (
    "observation/depth_external",                # what the driver sends
    "observation/depth_exterior_image_1_left",   # contract-doc spelling
)
_EPISODE_KEYS = ("episode_id", "__episode_id")

# Mirror the metadata the driver already accepts from the reference servers
# (both send ``orchestrator: True``, plus openpi-style fields). Keep the shape
# identical so nothing on the client side can key off a missing field.
_METADATA = {
    "model": "capx-franky-ws",
    "action_horizon": ACTION_HORIZON,
    "action_dim": 8,
    "control_frequency_hz": 15.0,
    "orchestrator": True,
    "notice": (
        "CaP-X real-robot endpoint. Actions come from CaP-X's generated code, "
        "not a learned policy."
    ),
}

# ``openpi_client`` encodes ndarrays with its own msgpack extension. Bind the C
# implementation of msgpack so a global ``msgpack_numpy.patch()`` elsewhere in
# the process cannot change what we put on the wire.
try:  # pragma: no cover - depends on the deployment env
    import msgpack

    from msgpack import _cmsgpack as _raw_msgpack
    from openpi_client import msgpack_numpy as _openpi_codec

    _PACKER = _raw_msgpack.Packer
    _UNPACKB = _raw_msgpack.unpackb
    _PACK_HOOK = _openpi_codec.pack_array
    _UNPACK_HOOK = _openpi_codec.unpack_array
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "franky_ws needs msgpack + openpi_client (the driver's codec)"
    ) from exc


def _packb(obj) -> bytes:
    return _PACKER(default=_PACK_HOOK, use_bin_type=True).pack(obj)


def _decode_hook(obj):
    """Decode an ndarray written by EITHER encoding.

    openpi's own codec uses ``{b"__ndarray__", b"data", b"dtype", b"shape"}``.
    But importing ``capx.envs.simulators`` pulls in ``franka_real`` ->
    ``msgpack_server_client_utils``, which calls ``msgpack_numpy.patch()`` at
    import time. That REBINDS ``msgpack.Packer`` process-wide, and since openpi's
    packer is ``partial(msgpack.Packer, default=pack_array)``, a client in such a
    process emits msgpack_numpy's ``{b"nd", b"type", b"kind", b"shape", b"data"}``
    instead. We do not control the client's import order, so accept both --
    otherwise arrays arrive as raw dicts, joints read as None, and the chunk
    silently becomes ZEROS (a violent move to the zero configuration).
    """
    obj = _UNPACK_HOOK(obj)
    if isinstance(obj, dict) and (b"nd" in obj or "nd" in obj):
        try:
            import msgpack_numpy as _mn

            return _mn.decode(obj)
        except Exception:  # pragma: no cover
            return obj
    return obj


def _unpackb(raw):
    return _UNPACKB(raw, object_hook=_decode_hook, raw=False)


def wire_gripper_to_capx(v: float) -> float:
    """wire 0=open,1=closed  ->  CaP-X 0=closed,1=open."""
    return float(1.0 - np.clip(float(v), 0.0, 1.0))


def capx_gripper_to_wire(v: float) -> float:
    """CaP-X 0=closed,1=open  ->  wire 0=open,1=closed (binarised at 0.5)."""
    return WIRE_GRIPPER_OPEN if float(v) > 0.5 else WIRE_GRIPPER_CLOSED


def _arr(v, dtype=np.float64) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        return np.asarray(v, dtype=dtype)
    except (TypeError, ValueError):
        return None


def _mon_first_episode(ep_id: Any) -> None:
    """Announce the first episode of a session. Never raises."""
    try:
        from capx.monitor import hooks as _h

        _h.episode_start(ep_id)
    except Exception:
        pass


def _mon_episode(prev: Any, new: Any, dropped: int) -> None:
    """Tell the monitor an episode boundary happened. Never raises."""
    try:
        from capx.monitor import hooks as _h

        if prev is not None:
            _h.episode_end(prev, f"operator started episode {new}")
        _h.episode_start(new)
        if dropped:
            _h.note(f"dropped {dropped} stale waypoint(s) from episode {prev}")
    except Exception:
        pass


def _first(wire: dict, keys):
    for k in keys:
        if wire.get(k) is not None:
            return wire[k]
    return None


#: One server per (host, port) for the lifetime of the PROCESS. A trial owning
#: the socket would close it on exit, so the driver would find a dead port
#: between episodes ("no close frame received or sent"). Sharing the endpoint
#: makes this a service: it binds once and stays up across trials.
_SERVERS: Dict[tuple, "_Endpoint"] = {}
_SERVERS_LOCK = threading.Lock()


class _Endpoint:
    """The socket + shared state, independent of any single trial."""

    def __init__(self, host: str, port: int, action_horizon: int) -> None:
        self.host = host
        self.port = port
        self.action_horizon = action_horizon
        self.lock = threading.Lock()
        self.wire: Dict[str, Any] = {}
        self.traj: list[np.ndarray] = []
        self.cursor = 0
        self.last_cmd: Optional[np.ndarray] = None
        self.gripper_capx = 1.0
        self.frames_served = 0
        self.connected = False
        # The driver tags every frame with an incrementing ``episode_id``
        # ("so the server can detect boundaries" -- episode_runner.py:103).
        # ``episode_seen`` is what we have acted on; a mismatch with the live
        # frame means the operator started/stopped an episode.
        self.episode_seen: Any = None
        # Wall-clock of the last frame, for idle/episode-end detection. The
        # driver never sends an explicit "episode over": episode_runner calls
        # policy.reset(), which only clears LOCAL chunk state and leaves the
        # socket open (eval/base_client.py:63). So the only observable end of an
        # episode is that frames stop arriving.
        self.last_frame_t: float = 0.0
        self.ended_announced: bool = False
        # Bumped on every episode boundary. Tool calls captured a generation at
        # entry; a mismatch means their plan is stale and must abort.
        self.generation: int = 0


class FrankyWsLowLevel(BaseEnv):
    """Serves the driver's policy contract; exposes CaP-X's control surface."""

    #: Last plan generation whose abort was announced (de-dupes the notice).
    _abort_announced: Optional[int] = None

    #: Generation the running plan was bound to; None = unbound, never aborts.
    #: Class-level so the attribute always exists, including for instances built
    #: with __new__ (tests) and for code paths that never call begin_plan.
    _plan_generation: Optional[int] = None

    def __init__(
        self,
        seed: int | None = None,
        *,
        host: str = "0.0.0.0",
        port: int = 8041,
        action_horizon: int = ACTION_HORIZON,
        max_joint_step_rad: float = MAX_JOINT_STEP_RAD,
        privileged: bool = False,
        enable_render: bool = False,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = int(port)
        self.action_horizon = int(action_horizon)
        self.max_joint_step_rad = float(max_joint_step_rad)

        # ``plan_grasp`` assigns these on the env (control_reduced.py:420).
        self.grasp_sample = None
        self.grasp_scores = None
        self.grasp_contact_pts = None
        self._control_freq = 15.0
        self._record_video = False
        self._record_wrist = False
        self._frames: list[np.ndarray] = []
        self._wrist_frames: list[np.ndarray] = []

        # Reuse the process-wide endpoint so the socket outlives this trial.
        key = (host, self.port)
        with _SERVERS_LOCK:
            ep = _SERVERS.get(key)
            if ep is None:
                ep = _Endpoint(host, self.port, self.action_horizon)
                _SERVERS[key] = ep
                self._serve_in_background(ep)
                self._start_idle_watchdog()
                try:
                    from capx.monitor import hooks as _mh
                    from capx.monitor.server import start_monitor

                    _mh.install_tool_hook()
                    try:
                        from capx.monitor.instrument import instrument_tools

                        instrument_tools()
                    except Exception as _iexc:
                        print(
                            f"[monitor] tool instrumentation off: {_iexc}",
                            flush=True,
                        )
                    start_monitor(
                        port=int(os.environ.get("CAPX_MONITOR_PORT", 8300)),
                        endpoint=f"ws://{host}:{self.port}/",
                        model=os.environ.get("CAPX_MONITOR_MODEL"),
                    )
                except Exception as _exc:
                    # Monitoring is optional, but a silent failure would leave
                    # the operator staring at an unreachable dashboard.
                    print(
                        f"[monitor] NOT started: {type(_exc).__name__}: {_exc}",
                        flush=True,
                    )
                print(
                    f"[franky-ws] endpoint created on ws://{host}:{self.port}/",
                    flush=True,
                )
            else:
                print(
                    f"[franky-ws] reusing the live endpoint on "
                    f"ws://{host}:{self.port}/ (driver stays connected)",
                    flush=True,
                )
        self._ep = ep

    # ── server ─────────────────────────────────────────────────────────
    # -- endpoint-backed state (shared across trials) -------------------
    @property
    def _lock(self):
        return self._ep.lock

    @property
    def frames_served(self) -> int:
        return self._ep.frames_served

    def _serve_in_background(self, ep: "_Endpoint"):
        loop = asyncio.new_event_loop()

        def run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._serve(ep))

        th = threading.Thread(target=run, daemon=True)
        th.start()
        return loop, th

    async def _serve(self, ep: "_Endpoint") -> None:
        import websockets.asyncio.server as ws_server

        def health(connection, request):
            # Log EVERY inbound request. Without this a rejected handshake is
            # invisible on our side and the client only sees
            # "no close frame received or sent".
            peer = getattr(connection, "remote_address", None)
            print(
                f"[franky-ws] << request path={request.path!r} from {peer} "
                f"upgrade={request.headers.get('Upgrade')!r}",
                flush=True,
            )
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        async with ws_server.serve(
            self._handler,
            ep.host,
            ep.port,
            compression=None,
            max_size=None,
            process_request=health,
            # Match the driver's server: a short default ping timeout would
            # drop a client that pauses between episodes.
            ping_interval=60,
            ping_timeout=120,
        ):
            print(
                f"[franky-ws] serving ws://{self.host}:{self.port}/ "
                f"(horizon={self.action_horizon})",
                flush=True,
            )
            await asyncio.Future()

    async def _handler(self, ws) -> None:
        await ws.send(_packb(_METADATA))
        print("[franky-ws] driver connected", flush=True)
        n = 0
        try:
            async for raw in ws:
                try:
                    wire = _unpackb(raw)
                except Exception as exc:
                    # Print, don't log: capx.envs.launch configures no logging,
                    # so a logger call here would be invisible and the driver
                    # would just see the socket close.
                    print(
                        f"[franky-ws] !! cannot decode a {len(raw)}-byte frame: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise
                if wire.get("__finalize_only"):
                    continue
                if n == 0:
                    self._report_first_frame(wire)
                try:
                    actions = self._on_frame(wire)
                except Exception as exc:
                    print(
                        f"[franky-ws] !! failed to build a chunk: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise
                await ws.send(_packb({"actions": actions}))
                n += 1
        except Exception as exc:
            print(
                f"[franky-ws] session ended after {n} frame(s): "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            # Drop the stored observation. A disconnected client's last frame is
            # STALE: if a probe (or a crashed driver) leaves one behind, the next
            # get_observation() would return it and CaP-X would plan on data
            # that no longer describes the robot. Also clear any queued
            # trajectory so a reconnecting driver is not handed old waypoints.
            with self._lock:
                self._ep.wire = {}
                self._ep.traj, self._ep.cursor = [], 0
                self._ep.last_cmd = None
            print(
                "[franky-ws] driver disconnected; cleared observation + "
                "trajectory (waiting for a fresh connection)",
                flush=True,
            )
            self._announce_episode_end("driver disconnected")

    def _announce_episode_end(self, reason: str) -> None:
        """Close out the live episode on the dashboard. Idempotent, never raises.

        Needed because the driver sends no end-of-episode message: it calls
        policy.reset() (local chunk state only) and keeps the socket open, so a
        stopped episode is indistinguishable from a slow one except by the
        absence of frames. Without this the dashboard shows every episode as
        still running forever.
        """
        with self._lock:
            if self._ep.ended_announced or self._ep.episode_seen is None:
                return
            ep = self._ep.episode_seen
            self._ep.ended_announced = True
            # Invalidate the running plan as well. Announcing only to the
            # dashboard meant end-of-episode had two different meanings: the UI
            # knew the episode was over while the plan kept calling the LLM and
            # commanding the arm, until the NEXT episode finally bumped the
            # generation. Bumping it here makes the end of an episode abort its
            # plan at the next robot interaction.
            self._ep.generation += 1
            gen = self._ep.generation
        print(
            f"[franky-ws] episode {ep} ended ({reason}); invalidating the "
            f"running plan (generation -> {gen})",
            flush=True,
        )
        try:
            from capx.monitor import hooks as _h

            _h.episode_end(ep, f"episode {ep} ended ({reason})")
        except Exception:
            pass

    # ── aborting a superseded plan ─────────────────────────────────────
    def current_generation(self) -> int:
        with self._lock:
            return self._ep.generation

    def _announce_abort(self, planned: int, gen: int) -> None:
        # Once per superseded plan: the checkpoint fires on every tool call, and
        # generated code may catch-and-retry, which would flood the feed.
        if getattr(self, "_abort_announced", None) == planned:
            return
        self._abort_announced = planned
        try:
            from capx.monitor import hooks as _h

            _h.note(
                f"aborted the episode-{planned} plan: operator started a new "
                f"episode (generation {planned} -> {gen})"
            )
        except Exception:
            pass

    def _abort_if_superseded(self) -> None:
        """Raise EpisodeSuperseded if the operator moved on.

        Called at every point where generated code touches the robot, so a new
        episode really does start fresh instead of racing the previous plan.
        """
        with self._lock:
            gen = self._ep.generation
            planned = self._plan_generation
        if planned is not None and gen != planned:
            self._announce_abort(planned, gen)
            raise EpisodeSuperseded(
                f"episode advanced (plan generation {planned} -> {gen}); "
                "aborting the stale plan; terminated episode."
            )

    def begin_plan(self) -> None:
        """Bind the code about to run to the CURRENT episode generation.

        Raises EpisodeSuperseded if the episode has already ENDED. ``step()``
        re-binds before every code block, so an unconditional re-bind let each
        new block re-arm itself against a dead episode: the abort killed only
        the block that was in flight, and the trial carried on solving the task.
        """
        with self._lock:
            if self._ep.ended_announced:
                ep = self._ep.episode_seen
                raise EpisodeSuperseded(
                    f"episode {ep} has ended; refusing to start new work on it; "
                    "terminated episode."
                )
            self._plan_generation = self._ep.generation

    def frames_are_live(self, max_age_s: float = 3.0) -> bool:
        """True if a driver frame arrived within *max_age_s*.

        ``_wire_prompt()`` returns the LAST frame forever, so a trial starting
        after an episode ended would re-plan against a remembered observation.
        Callers use this to wait for the robot to actually be driven again.
        """
        with self._lock:
            last = self._ep.last_frame_t
        return bool(last) and (time.time() - last) <= max_age_s

    def _idle_seconds(self) -> float:
        with self._lock:
            last = self._ep.last_frame_t
        return (time.time() - last) if last else 0.0

    def _start_idle_watchdog(self, idle_s: float = IDLE_EPISODE_END_S) -> None:
        """Mark the episode ended once frames stop for ``idle_s``."""

        def _loop() -> None:
            while True:
                time.sleep(1.0)
                try:
                    if 0 < idle_s <= self._idle_seconds():
                        self._announce_episode_end(
                            f"no frames for {idle_s:.0f}s"
                        )
                except Exception:
                    pass

        threading.Thread(target=_loop, daemon=True).start()

    @staticmethod
    def _report_first_frame(wire: dict) -> None:
        """Print the first frame's schema so a mismatch is immediately visible."""
        print("\n[franky-ws] === first observation ===", flush=True)
        for k in sorted(wire, key=str):
            v = wire[k]
            if isinstance(v, np.ndarray):
                d = f"ndarray {v.shape} {v.dtype}"
                if np.issubdtype(v.dtype, np.floating):
                    f = v[np.isfinite(v)]
                    if f.size:
                        d += f" min={f.min():.4f} max={f.max():.4f}"
                        if f.size < v.size:
                            d += f" ({v.size - f.size} NaN)"
                print(f"[franky-ws]   {str(k):46s} {d}", flush=True)
            else:
                print(f"[franky-ws]   {str(k):46s} {type(v).__name__} {v!r}", flush=True)
        missing = [
            k
            for k in (
                "prompt",
                "observation/joint_position",
                "observation/gripper_position",
                "observation/exterior_image_1_left",
            )
            if k not in wire
        ]
        if missing:
            print(f"[franky-ws]   !! MISSING REQUIRED: {missing}", flush=True)
        print("[franky-ws] === end first observation ===\n", flush=True)

    # ── the cursor: never block the driver ─────────────────────────────
    def _on_frame(self, wire: dict) -> np.ndarray:
        """Store the observation and answer with the next chunk."""
        ep_id = _first(wire, _EPISODE_KEYS)
        with self._lock:
            prev = self._ep.episode_seen
            if ep_id is not None and prev is None:
                # FIRST episode of the session: no boundary to detect, but the
                # monitor still needs an episode_start or every later event is
                # filed under "no episode" and the sidebar stays empty.
                _mon_first_episode(ep_id)
            if ep_id is not None and prev is not None and ep_id != prev:
                # New episode from the operator: the previous plan describes a
                # scene that no longer exists. Drop it rather than driving the
                # arm through stale waypoints.
                print(
                    f"[franky-ws] episode {prev} -> {ep_id}: dropping "
                    f"{max(0, len(self._ep.traj) - self._ep.cursor)} queued "
                    "waypoint(s)",
                    flush=True,
                )
                dropped = max(0, len(self._ep.traj) - self._ep.cursor)
                self._ep.traj, self._ep.cursor = [], 0
                self._ep.last_cmd = None
                self._ep.generation += 1
                _mon_episode(prev, ep_id, dropped)
            if ep_id is not None:
                self._ep.episode_seen = ep_id
            self._ep.last_frame_t = time.time()
            self._ep.ended_announced = False
            self._ep.wire = wire
            if self._ep.last_cmd is None:
                # First frame: hold wherever the arm currently is.
                j = _arr(wire.get("observation/joint_position"))
                if j is None or j.size < 7:
                    # Never fabricate a pose: zeros would command the arm to its
                    # zero configuration. Fail loudly instead.
                    raise ValueError(
                        "observation/joint_position missing or undecodable "
                        f"(got {type(wire.get('observation/joint_position')).__name__}); "
                        "refusing to synthesize an action chunk"
                    )
                j = j.reshape(-1)[:7]
                self._ep.last_cmd = np.concatenate(
                    [j, [capx_gripper_to_wire(self._ep.gripper_capx)]]
                )
            rows = []
            for _ in range(self.action_horizon):
                if self._ep.cursor < len(self._ep.traj):
                    self._ep.last_cmd = self._ep.traj[self._ep.cursor]
                    self._ep.cursor += 1
                rows.append(self._ep.last_cmd)
            if self._ep.cursor >= len(self._ep.traj) and self._ep.traj:
                self._ep.traj = []
                self._ep.cursor = 0
            self._ep.frames_served += 1
            return np.asarray(rows, dtype=np.float64)

    def _enqueue(self, targets: list[np.ndarray]) -> None:
        with self._lock:
            self._ep.traj.extend(targets)
        # Deliberately NOT published here. move_to_joints_blocking enqueues in
        # small batches, so publishing per call floods the feed with dozens of
        # near-identical "enqueued N waypoint(s)" rows that tell the operator
        # nothing. One summary per completed motion is emitted instead (see
        # move_to_joints_blocking).
        return

    def _pending(self) -> int:
        with self._lock:
            return max(0, len(self._ep.traj) - self._ep.cursor)

    # ── observation ────────────────────────────────────────────────────
    def _wait_for_frame(self, timeout_s: float = 600.0) -> dict:
        deadline = time.time() + timeout_s
        announced = False
        while time.time() < deadline:
            with self._lock:
                if self._ep.wire:
                    return dict(self._ep.wire)
            if not announced:
                print(
                    "[franky-ws] waiting for the driver's first observation...",
                    flush=True,
                )
                announced = True
            time.sleep(0.05)
        raise TimeoutError(
            f"no observation within {timeout_s:.0f}s — is the driver pointed at "
            f"ws://{self.host}:{self.port}/ ?"
        )

    def peek_observation(self) -> dict:
        """Non-blocking view of the latest frame's task fields.

        Used by the real code env to learn the instruction without waiting. Empty
        dict when no driver frame has arrived yet (or after a disconnect).
        """
        with self._lock:
            wire = dict(self._ep.wire)
        if not wire:
            return {}
        out: dict[str, Any] = {}
        if isinstance(wire.get("prompt"), str):
            out["wire_prompt"] = wire["prompt"]
        ep = _first(wire, _EPISODE_KEYS)
        if ep is not None:
            out["wire_episode_id"] = ep
        return out

    def episode_id(self) -> Any:
        """The episode id the driver is currently sending (None before any frame)."""
        with self._lock:
            return self._ep.episode_seen

    def current_joints(self) -> np.ndarray:
        j = _arr(self._wait_for_frame().get("observation/joint_position"))
        return np.zeros(7) if j is None else j.reshape(-1)[:7]

    def get_observation(self) -> dict[str, Any]:
        # NO abort check here. base.step() calls _get_observation() AFTER the
        # exec sandbox returns, so raising here escapes CodeExecutionEnvBase's
        # BaseException handler and propagates through trial.py -> runner.py ->
        # launch.py, killing the whole process (and port 8041 with it).
        # Aborts belong on the paths generated code calls DIRECTLY: motion and
        # gripper commands.
        """CaP-X-shaped observation.

        Key names mirror ``robolab.py::get_observation`` so the same API tier
        works unchanged: ``agentview`` for the scene camera,
        ``robot0_eye_in_hand`` for the wrist, plus proprioception.
        """
        wire = self._wait_for_frame()
        obs: dict[str, Any] = {}

        agentview: dict[str, Any] = {"images": {}}
        # ``rgb`` MUST share the resolution of depth + intrinsics. The generated
        # code segments ``rgb`` with SAM3 and passes that mask straight to
        # plan_grasp(depth=..., intrinsics=..., segmentation=mask); a 224x224
        # mask against 720x1280 depth raises
        # "operands could not be broadcast together with shapes (720,1280) (224,224)".
        # The 224x224 frame the driver also sends is a VLA policy input and is
        # deliberately NOT used here (CaP-X is not a VLA). robolab.py keeps
        # rgb/depth/K at one resolution for the same reason.
        rgb_native = wire.get("observation/exterior_image_1_left_raw")
        rgb_small = wire.get("observation/exterior_image_1_left")
        rgb = rgb_native if rgb_native is not None else rgb_small
        if rgb is not None:
            agentview["images"]["rgb"] = np.asarray(rgb, dtype=np.uint8)
        if rgb_small is not None:
            # Kept for logging / visual differencing, never for geometry.
            agentview["images"]["rgb_policy"] = np.asarray(rgb_small, dtype=np.uint8)
        depth = _first(wire, _DEPTH_KEYS)
        if depth is not None:
            d = np.asarray(depth, dtype=np.float32)
            # Metres, NaN where the stereo match failed. Downstream code masks
            # NaN explicitly (control.py: ``~np.isnan(depth)``); do not fill it.
            agentview["images"]["depth"] = d[:, :, None] if d.ndim == 2 else d
        k = _arr(wire.get("observation/camera_K"))
        if k is not None:
            agentview["intrinsics"] = k.reshape(3, 3)
        ext = _arr(wire.get("observation/camera_extrinsic"))
        if ext is not None:
            # cam->base, already OpenCV. No conversion (the sim path needs an
            # OpenGL flip; applying it here would corrupt every grasp pose).
            agentview["pose_mat"] = ext.reshape(4, 4)
        obs["agentview"] = agentview
        # Same camera under the name the real API's helpers expect.
        obs["robot0_robotview"] = agentview

        wrist = wire.get("observation/wrist_image_left")
        if wrist is not None:
            obs["robot0_eye_in_hand"] = {
                "images": {"rgb": np.asarray(wrist, dtype=np.uint8)}
            }

        joints = _arr(wire.get("observation/joint_position"))
        joints = np.zeros(7) if joints is None else joints.reshape(-1)[:7]
        grip = _arr(wire.get("observation/gripper_position"))
        grip_capx = (
            self._ep.gripper_capx
            if grip is None or grip.size == 0
            else wire_gripper_to_capx(grip.reshape(-1)[0])
        )
        # [7 joints + gripper], matching the sim adapters.
        obs["robot_joint_pos"] = np.concatenate([joints, [grip_capx]])

        ee_pos = _arr(wire.get("observation/ee_pos"))
        ee_quat = _arr(wire.get("observation/ee_quat"))
        if ee_pos is not None:
            q = (
                np.array([1.0, 0.0, 0.0, 0.0])
                if ee_quat is None
                else ee_quat.reshape(-1)[:4]
            )
            obs["robot_cartesian_pos"] = np.concatenate(
                [ee_pos.reshape(-1)[:3], q, [grip_capx]]
            )

        # The driver sends the task every frame; expose it so the code env can
        # take the instruction from the wire.
        if isinstance(wire.get("prompt"), str):
            obs["wire_prompt"] = wire["prompt"]
        ep = _first(wire, _EPISODE_KEYS)
        if ep is not None:
            obs["wire_episode_id"] = ep
        self._capture_frame(obs)
        self._maybe_publish_scene(obs)
        return obs

    def _maybe_publish_scene(self, obs: dict) -> None:
        """Send RGB + colourised depth to the monitor, throttled.

        The operator needs to see WHAT THE ROBOT SEES -- an rgb thumbnail plus a
        depth heat-map with its valid-pixel fraction, which is the fastest way to
        spot a bad camera or an all-NaN depth frame. Throttled to once every
        ``_scene_period_s`` (and always on a new episode) so the feed stays
        readable and encoding never sits on the action path for long.
        """
        import time as _t

        now = _t.time()
        ep = self._ep.episode_seen
        first_of_episode = ep != getattr(self, "_scene_last_ep", object())
        if not first_of_episode and now - getattr(self, "_scene_last_t", 0.0) < getattr(
            self, "_scene_period_s", 20.0
        ):
            return
        self._scene_last_t = now
        self._scene_last_ep = ep
        try:
            from capx.monitor import hooks as _h

            imgs = obs.get("agentview", {}).get("images", {})
            rgb = imgs.get("rgb")
            depth = imgs.get("depth")
            _h.scene(rgb=rgb, depth=depth, wrist=(
                obs.get("robot0_eye_in_hand", {}).get("images", {}).get("rgb")
            ), joints=obs.get("robot_joint_pos"))
        except Exception:
            pass

    # ── control surface used by FrankaControlApi ────────────────────────
    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.01,
        max_steps: int = 2000,
    ) -> bool:
        """Interpolate to *joints*, then wait for the driver to execute it.

        Enqueues waypoints and waits for the BUFFER to drain rather than
        blocking inside the request handler, so the driver keeps receiving
        chunks throughout. Returns True if the arm reached the target.
        """
        self._abort_if_superseded()
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        start = self.current_joints()
        span = float(np.abs(target - start).max())
        n = max(1, int(np.ceil(span / max(self.max_joint_step_rad, 1e-6))))
        grip_wire = capx_gripper_to_wire(self._ep.gripper_capx)
        self._enqueue(
            [
                np.concatenate(
                    [start + (target - start) * (i / n), [grip_wire]]
                )
                for i in range(1, n + 1)
            ]
        )

        for _ in range(max_steps):
            # Check DURING the wait: a boundary can land mid-motion, and this
            # loop is where the plan spends most of its time.
            self._abort_if_superseded()
            if self._pending() == 0:
                break
            time.sleep(0.01)

        err = float(np.linalg.norm(self.current_joints() - target))
        try:
            from capx.monitor import hooks as _h

            _h.motion_done(
                n_waypoints=n,
                start=[round(float(x), 3) for x in start],
                target=[round(float(x), 3) for x in target],
                residual=round(err, 4),
                converged=err <= tolerance,
            )
        except Exception:
            pass
        if err > tolerance:
            logger.warning(
                "move_to_joints_blocking: residual %.4f rad > tol %.4f "
                "(arm may still be settling)", err, tolerance,
            )
            return False
        return True

    def _set_gripper(self, fraction: float) -> None:
        """CaP-X convention: 0 = closed, 1 = open."""
        self._abort_if_superseded()
        self._ep.gripper_capx = float(np.clip(fraction, 0.0, 1.0))
        # Hold the current pose while the gripper actuates.
        j = self.current_joints()
        wire_g = capx_gripper_to_wire(self._ep.gripper_capx)
        self._enqueue([np.concatenate([j, [wire_g]])] * self.action_horizon)
        for _ in range(200):
            if self._pending() == 0:
                break
            time.sleep(0.01)

    def open_gripper(self) -> None:
        self._set_gripper(1.0)

    def close_gripper(self) -> None:
        self._set_gripper(0.0)

    def _step_once(self) -> None:
        """Hold the current control state for one chunk."""
        j = self.current_joints()
        self._enqueue(
            [
                np.concatenate([j, [capx_gripper_to_wire(self._ep.gripper_capx)]])
            ]
            * self.action_horizon
        )

    def _read_arm_joints(self, obs: dict | None = None) -> np.ndarray:
        if obs and obs.get("robot_joint_pos") is not None:
            return np.asarray(obs["robot_joint_pos"][:-1], dtype=np.float64)
        return self.current_joints()

    # ── gym-ish surface ────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        """Start a fresh episode: drop stale motion and re-sync held state.

        The env object is created ONCE per process and reused for every trial
        (runner.py instantiates outside the loop), so anything not cleared here
        leaks into the next episode.
        """
        with self._lock:
            self._ep.traj = []
            self._ep.cursor = 0
            self._ep.last_cmd = None
            # Re-sync the COMMANDED gripper from the wire. Without this a new
            # episode inherits the previous episode's commanded state, so a plan
            # that ended holding an object starts the next episode believing the
            # gripper is still closed.
            grip = _arr(self._ep.wire.get("observation/gripper_position"))
            if grip is not None and grip.size:
                self._ep.gripper_capx = wire_gripper_to_capx(
                    grip.reshape(-1)[0]
                )
        # Monitor bookkeeping: allow the next superseded plan to be announced.
        self._abort_announced = None
        # No plan is bound until the caller binds one, so setup code that runs
        # before begin_plan() can never abort.
        self._plan_generation = None
        return self.get_observation(), {}

    def step(self, action: Any):
        self._step_once()
        return self.get_observation(), 0.0, False, False, {}

    def task_completed(self) -> bool:
        """No ground truth on hardware; the operator scores each trial."""
        return False

    def compute_reward(self) -> float:
        return 0.0

    def get_current_time_s(self) -> float:
        return time.time()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        rgb = self.get_observation()["agentview"]["images"].get("rgb")
        return (
            np.zeros((224, 224, 3), dtype=np.uint8) if rgb is None else rgb
        )

    # ── video capture (used when record_video is on) ────────────────────
    # capx.envs.tasks.base / trial call these on the low-level env; without them
    # the trial crashes with AttributeError and the whole process exits, taking
    # the WebSocket server with it (the driver then sees the socket vanish).
    # Frames come from the driver's scene camera, sampled per observation.
    def enable_video_capture(
        self, enabled: bool = True, *, clear: bool = False, wrist_camera: bool = False
    ) -> None:
        self._record_video = bool(enabled)
        self._record_wrist = bool(wrist_camera)
        if clear:
            self._frames = []
            self._wrist_frames = []

    def _capture_frame(self, obs: dict) -> None:
        if not getattr(self, "_record_video", False):
            return
        rgb = obs.get("agentview", {}).get("images", {}).get("rgb")
        if rgb is not None:
            self._frames.append(np.asarray(rgb, dtype=np.uint8))
        if getattr(self, "_record_wrist", False):
            w = (
                obs.get("robot0_eye_in_hand", {})
                .get("images", {})
                .get("rgb")
            )
            if w is not None:
                self._wrist_frames.append(np.asarray(w, dtype=np.uint8))

    def get_video_frames(self, *, clear: bool = False) -> list:
        out = list(getattr(self, "_frames", []))
        if clear:
            self._frames = []
        return out

    def get_video_frame_count(self) -> int:
        return len(getattr(self, "_frames", []))

    def get_video_frames_range(self, start: int, end: int) -> list:
        return list(getattr(self, "_frames", [])[start:end])

    def get_wrist_video_frames(self, *, clear: bool = False) -> list:
        out = list(getattr(self, "_wrist_frames", []))
        if clear:
            self._wrist_frames = []
        return out

    def get_wrist_video_frames_range(self, start: int, end: int) -> list:
        return list(getattr(self, "_wrist_frames", [])[start:end])

    def close(self) -> None:
        return None


__all__ = ["FrankyWsLowLevel", "wire_gripper_to_capx", "capx_gripper_to_wire"]
