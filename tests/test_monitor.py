"""Monitor: episode-aware, non-blocking, and never able to break a run."""

from __future__ import annotations

import json
import queue
import urllib.request

import pytest

from capx.monitor.events import BUS, Event, EventBus, EventKind


@pytest.fixture(autouse=True)
def fresh_bus():
    BUS._history.clear()
    BUS.episodes.clear()
    BUS._current_episode = None
    yield


class TestMultiEpisode:
    """A real session is MANY episodes; history must be per-episode."""

    def test_episodes_are_tracked_separately(self):
        from capx.monitor import hooks as h

        h.episode_start(1, task="pick up the red block")
        h.motion(12)
        h.episode_end(1)
        h.episode_start(2, task="pick up the green container")
        h.motion(30)
        assert set(BUS.episodes) == {1, 2}
        assert BUS.episodes[1]["task"] == "pick up the red block"
        assert BUS.episodes[2]["task"] == "pick up the green container"
        assert BUS.episodes[1]["waypoints"] == 12
        assert BUS.episodes[2]["waypoints"] == 30
        assert BUS.episodes[1]["ended"] is not None
        assert BUS.episodes[2]["ended"] is None, "episode 2 still running"

    def test_events_inherit_the_current_episode(self):
        from capx.monitor import hooks as h

        h.episode_start(5)
        h.note("something happened")
        assert BUS._history[-1].episode == 5

    def test_tool_counts_per_episode(self):
        from capx.monitor import hooks as h
        from capx.monitor.events import publish

        h.episode_start(1)
        for _ in range(3):
            publish(EventKind.TOOL, "seg", data={"tool": "SAM3 Segmentation"})
        publish(EventKind.TOOL, "grasp", data={"tool": "Contact GraspNet"})
        h.episode_start(2)
        publish(EventKind.TOOL, "seg", data={"tool": "SAM3 Segmentation"})
        assert BUS.episodes[1]["tools"] == {
            "SAM3 Segmentation": 3,
            "Contact GraspNet": 1,
        }
        assert BUS.episodes[2]["tools"] == {"SAM3 Segmentation": 1}

    def test_snapshot_exposes_every_episode(self):
        from capx.monitor import hooks as h

        for i in (1, 2, 3):
            h.episode_start(i, task=f"task {i}")
        snap = BUS.snapshot()
        assert len(snap["episodes"]) == 3
        assert snap["current_episode"] == 3


class TestNeverBreaksTheRun:
    """Observability must not be able to affect the robot."""

    def test_publish_survives_a_bad_payload(self):
        class Boom:
            def __repr__(self):
                raise RuntimeError("nope")

        BUS.publish(Event(kind=EventKind.NOTE, data={"x": Boom()}))  # must not raise

    def test_slow_subscriber_does_not_block(self):
        bus = EventBus(queue_size=4)
        q = bus.subscribe()
        for i in range(50):                      # far more than the queue holds
            bus.publish(Event(kind=EventKind.NOTE, text=str(i)))
        assert q.qsize() <= 4, "queue must stay bounded"

    def test_hooks_are_safe_without_a_server(self):
        from capx.monitor import hooks as h

        h.episode_start(1)
        h.model_query("start")
        h.code_block("print(1)")
        h.error("boom")
        h.note("fine")            # none of these may raise

    def test_history_is_bounded(self):
        bus = EventBus(history=10)
        for i in range(100):
            bus.publish(Event(kind=EventKind.NOTE, text=str(i)))
        assert len(bus._history) == 10


class TestHttp:
    def _serve(self):
        import socket

        from capx.monitor.server import MonitorServer

        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        return MonitorServer("127.0.0.1", port).start(), port

    def test_dashboard_and_snapshot(self):
        from capx.monitor import hooks as h

        srv, port = self._serve()
        h.episode_start(1, task="pick up the object")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert r.status == 200
            assert b"CaP-X real robot" in r.read()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/snapshot", timeout=5
        ) as r:
            snap = json.loads(r.read())
        assert snap["episodes"][0]["task"] == "pick up the object"

    def test_healthz(self):
        srv, port = self._serve()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as r:
            assert r.status == 200


class TestToolHookRealPath:
    """The hook must fire through ApiBase._log_step, headless included.

    Two regressions this pins:
    * ``_log_step`` imports ``log_step`` INSIDE the method, so patching the
      module attribute never took effect.
    * ``_log_step`` early-returns unless ``_webui_enabled``, which only CaP-X's
      own web UI sets -- so a headless real-robot run emitted no tool events.
    """

    def test_hook_fires_with_webui_disabled(self):
        import numpy as np

        from capx.integrations.base_api import ApiBase
        from capx.monitor import hooks as h

        h._installed = False
        h.install_tool_hook()

        class Api(ApiBase):
            def __init__(self):
                self._webui_enabled = False      # headless: CaP-X logs nothing

            def functions(self):
                return {}

        Api()._log_step("SAM3 Segmentation", "segmenting red block",
                        images=np.zeros((32, 32, 3), dtype=np.uint8))
        tools = [e for e in BUS._history if e.kind == EventKind.TOOL]
        assert tools, "tool step must reach the monitor even when webui is off"
        assert tools[-1].data["tool"] == "SAM3 Segmentation"
        assert tools[-1].images, "image should be encoded"

    def test_install_is_idempotent(self):
        from capx.monitor import hooks as h

        h._installed = False
        h.install_tool_hook()
        first = __import__(
            "capx.integrations.base_api", fromlist=["ApiBase"]
        ).ApiBase._log_step
        h.install_tool_hook()
        second = __import__(
            "capx.integrations.base_api", fromlist=["ApiBase"]
        ).ApiBase._log_step
        assert first is second, "must not wrap twice"

    def test_image_encoding_is_defensive(self):
        from capx.monitor.hooks import _encode_images

        assert _encode_images(None) == []
        assert _encode_images(["not-base64"]) == []
        assert _encode_images([object()]) == []      # must not raise


class TestOperatorFeedback:
    """The three problems the operator reported after the first live session."""

    def test_task_is_filed_under_its_episode(self):
        """'episode 17 shows a task but no episode 17 tab'."""
        from capx.monitor import hooks as h

        h.task_adopted("pick up the larger yellow block", episode=17)
        assert 17 in BUS.episodes, "a task must create/attach to its episode row"
        assert BUS.episodes[17]["task"] == "pick up the larger yellow block"

    def test_episode_row_exists_even_without_an_explicit_start(self):
        """A trial restarting mid-episode emits no episode_start."""
        from capx.monitor.events import publish

        publish(EventKind.TOOL, "seg", episode=17, data={"tool": "SAM3"})
        assert 17 in BUS.episodes
        assert BUS.episodes[17]["tools"] == {"SAM3": 1}

    def test_one_motion_row_per_move_not_per_enqueue(self):
        """'a whole long list of not useful motion enqueue'."""
        from capx.monitor import hooks as h

        h.episode_start(1)
        h.motion_done(
            n_waypoints=120, start=[0.0] * 7, target=[0.1] * 7,
            residual=0.004, converged=True,
        )
        motions = [e for e in BUS._history if e.kind == EventKind.MOTION]
        assert len(motions) == 1
        assert "reached" in motions[0].text
        assert motions[0].data["waypoints"] == 120
        assert BUS.episodes[1]["waypoints"] == 120

    def test_non_convergence_is_visible(self):
        from capx.monitor import hooks as h

        h.motion_done(n_waypoints=10, residual=1.29, converged=False)
        assert "did NOT reach" in BUS._history[-1].text

    def test_scene_publishes_rgb_and_depth_heatmap(self):
        """'there is no visualization of the scene, depth, etc'."""
        import numpy as np

        from capx.monitor import hooks as h

        depth = np.full((60, 80), np.nan, dtype=np.float32)
        depth[10:50, 10:70] = 0.8
        h.scene(
            rgb=np.zeros((60, 80, 3), np.uint8),
            depth=depth,
            wrist=np.zeros((32, 32, 3), np.uint8),
            joints=np.zeros(8),
        )
        e = BUS._history[-1]
        assert e.kind == EventKind.OBSERVATION
        assert len(e.images) == 3, "rgb + depth heat-map + wrist"
        assert 0 < e.data["depth_valid_frac"] < 1, "NaN fraction must be reported"
        assert e.data["depth_min_m"] == 0.8

    def test_scene_survives_all_nan_depth(self):
        """An all-NaN depth frame is a real failure mode; must not raise."""
        import numpy as np

        from capx.monitor import hooks as h

        h.scene(depth=np.full((20, 20), np.nan, dtype=np.float32))
        assert BUS._history[-1].data["depth_valid_frac"] == 0.0


class TestEpisodeEndDetection:
    """The driver sends NO end-of-episode message.

    ``episode_runner`` calls ``policy.reset()``, which clears only local chunk
    state and leaves the socket open (``eval/base_client.py:63``). So a stopped
    episode looks exactly like a slow one apart from frames ceasing -- and the
    dashboard used to show every episode as running forever.
    """

    def test_end_is_announced_once(self):
        from capx.monitor import hooks as h

        h.episode_start(3)
        h.episode_end(3, "driver disconnected")
        assert BUS.episodes[3]["ended"] is not None

    def test_last_event_tracks_activity(self):
        from capx.monitor import hooks as h
        from capx.monitor.events import publish

        h.episode_start(4)
        started = BUS.episodes[4]["started"]
        publish(EventKind.TOOL, "x", data={"tool": "SAM3"})
        assert BUS.episodes[4]["last_event"] >= started

    def test_reopening_an_episode_keeps_its_task(self):
        """episode_start after a task event must not wipe the task."""
        from capx.monitor import hooks as h

        h.task_adopted("pick up the block", episode=9)
        h.episode_start(9)
        assert BUS.episodes[9]["task"] == "pick up the block"


class TestIdleWatchdog:
    def test_disconnect_and_idle_both_close_the_episode(self):
        import capx.envs.simulators.franky_ws as fw

        env = fw.FrankyWsLowLevel.__new__(fw.FrankyWsLowLevel)
        env._ep = fw._Endpoint(host="127.0.0.1", port=0, action_horizon=8)
        env._ep.episode_seen = 21
        env._ep.last_frame_t = 0.0

        env._announce_episode_end("driver disconnected")
        assert BUS.episodes[21]["ended"] is not None, "must close the episode"

        n = len([e for e in BUS._history if e.kind == EventKind.EPISODE_END])
        env._announce_episode_end("no frames for 12s")
        assert (
            len([e for e in BUS._history if e.kind == EventKind.EPISODE_END]) == n
        ), "must not announce the same end twice"

    def test_idle_seconds_reports_zero_before_any_frame(self):
        import capx.envs.simulators.franky_ws as fw

        env = fw.FrankyWsLowLevel.__new__(fw.FrankyWsLowLevel)
        env._ep = fw._Endpoint(host="127.0.0.1", port=0, action_horizon=8)
        assert env._idle_seconds() == 0.0


class TestNewEpisodeStartsClean:
    """Operator report: 'episode 19 has no task and opens with tool calls'."""

    def test_same_task_still_announced_for_a_new_episode(self):
        """The env is reused across trials, so the task string is unchanged.

        ``_adopt_wire_task`` early-returns in that case, so publishing the task
        from there left the new episode blank.
        """
        from capx.monitor import hooks as h

        h.episode_start(18, task="pick up the larger yellow block")
        h.episode_start(19, task="pick up the larger yellow block")
        assert BUS.episodes[19]["task"] == "pick up the larger yellow block"

    def test_leftover_tool_calls_are_marked_stale_not_attributed(self):
        """The old episode's code is synchronous and cannot be interrupted."""
        from capx.monitor import hooks as h
        from capx.monitor.events import publish

        h.episode_start(18, task="task A")
        BUS.mark_superseded(19)
        publish(EventKind.TOOL, "IK", episode=19, data={"tool": "IK Solver"})
        assert BUS._history[-1].data.get("stale") is True
        assert BUS.episodes.get(19, {}).get("tools", {}) == {}, (
            "leftovers must not count toward the new episode"
        )

    def test_new_task_ends_the_superseded_window(self):
        from capx.monitor import hooks as h
        from capx.monitor.events import publish

        BUS.mark_superseded(19)
        h.episode_start(19, task="task B")          # fresh plan begins
        publish(EventKind.TOOL, "SAM3", episode=19, data={"tool": "SAM3"})
        assert not BUS._history[-1].data.get("stale")
        assert BUS.episodes[19]["tools"] == {"SAM3": 1}
