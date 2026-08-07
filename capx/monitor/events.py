"""Event model + bus for the real-robot monitor.

The bus is deliberately dumb and non-blocking: producers are on the robot's
critical path, so an event must never block them. Each subscriber gets a bounded
queue and loses old events if it cannot keep up.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class EventKind(str, Enum):
    # session / episode lifecycle
    SESSION = "session"            # server up, config, endpoint
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    # what the driver is sending
    OBSERVATION = "observation"    # summary of a wire frame (no images by default)
    TASK = "task"                  # instruction adopted from the wire
    # what the agent is doing
    MODEL_QUERY = "model_query"    # LLM call started / finished
    CODE = "code"                  # generated code block
    TOOL = "tool"                  # a tool step (SAM3, GraspNet, IK, gripper...)
    # what the robot is told to do
    MOTION = "motion"              # trajectory enqueued / drained
    CHUNK = "chunk"                # action chunk served (throttled)
    # outcomes
    ERROR = "error"
    NOTE = "note"


@dataclass
class Event:
    kind: EventKind
    episode: Optional[Any] = None
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)  # base64 PNG/JPEG
    ts: float = field(default_factory=time.time)
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value if isinstance(self.kind, EventKind) else self.kind
        return d


class EventBus:
    """Fan-out with bounded per-subscriber queues and a replay buffer.

    ``history`` lets a browser opened mid-episode still render what happened,
    which matters when a run has been going for an hour.
    """

    def __init__(self, history: int = 4000, queue_size: int = 1000) -> None:
        self._history_max = history
        self._queue_size = queue_size
        self._lock = threading.Lock()
        self._history: list[Event] = []
        self._subs: list[queue.Queue] = []
        self._seq = 0
        #: Per-episode roll-up so the UI can show a summary table without
        #: replaying every event.
        self.episodes: dict[Any, dict[str, Any]] = {}
        self._current_episode: Any = None
        # Episodes whose fresh plan has not started yet (see mark_superseded).
        self._awaiting_plan: set = set()

    # -- publish -------------------------------------------------------
    def publish(self, event: Event) -> None:
        """Never raises, never blocks. Safe to call from the robot loop."""
        try:
            with self._lock:
                self._seq += 1
                event.seq = self._seq
                if event.episode is None:
                    event.episode = self._current_episode
                if (
                    event.episode in self._awaiting_plan
                    and event.kind
                    not in (
                        EventKind.EPISODE_START,
                        EventKind.EPISODE_END,
                        EventKind.TASK,
                        EventKind.SESSION,
                    )
                ):
                    # Leftovers from the previous episode's still-running code.
                    event.data = dict(event.data or {})
                    event.data["stale"] = True
                self._history.append(event)
                if len(self._history) > self._history_max:
                    del self._history[: len(self._history) - self._history_max]
                self._roll_up(event)
                subs = list(self._subs)
            for q in subs:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    # Drop the oldest so a slow browser degrades instead of
                    # blocking the producer.
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except Exception:
                        pass
        except Exception:
            # Observability must never break the run.
            pass

    def _roll_up(self, e: Event) -> None:
        if e.kind == EventKind.EPISODE_START:
            self._current_episode = e.episode
            existing = self.episodes.get(e.episode)
            if existing is not None:
                # A row can already exist because a task/tool event arrived
                # first (auto-created below). Re-opening the episode must not
                # wipe what we already know about it.
                existing["ended"] = None
                existing["last_event"] = e.ts
                return
            self.episodes[e.episode] = {
                "episode": e.episode,
                "started": e.ts,
                "ended": None,
                "task": None,
                "tools": {},
                "model_calls": 0,
                "code_blocks": 0,
                "errors": 0,
                "waypoints": 0,
                "chunks": 0,
                "last_event": e.ts,
            }
            return
        ep = self.episodes.get(e.episode)
        if ep is None:
            if e.episode is None:
                return
            # Events can arrive for an episode we never saw start -- e.g. the
            # trial restarted mid-episode, so the endpoint had already recorded
            # that id and emitted no new episode_start. Create the row rather
            # than dropping the data (that produced "a task with no episode
            # tab").
            self._current_episode = e.episode
            ep = self.episodes[e.episode] = {
                "episode": e.episode,
                "started": e.ts,
                "ended": None,
                "task": None,
                "tools": {},
                "model_calls": 0,
                "code_blocks": 0,
                "errors": 0,
                "waypoints": 0,
                "chunks": 0,
                "last_event": e.ts,
            }
        if (e.data or {}).get("stale"):
            return                       # leftovers must not count for this episode
        ep["last_event"] = e.ts
        if e.kind == EventKind.EPISODE_END:
            ep["ended"] = e.ts
        elif e.kind == EventKind.TASK:
            ep["task"] = e.text
        elif e.kind == EventKind.TOOL:
            name = e.data.get("tool", "?")
            ep["tools"][name] = ep["tools"].get(name, 0) + 1
        elif e.kind == EventKind.MODEL_QUERY and e.data.get("phase") == "end":
            ep["model_calls"] += 1
        elif e.kind == EventKind.CODE:
            ep["code_blocks"] += 1
        elif e.kind == EventKind.ERROR:
            ep["errors"] += 1
        elif e.kind == EventKind.MOTION:
            ep["waypoints"] += int(e.data.get("waypoints", 0) or 0)
        elif e.kind == EventKind.CHUNK:
            ep["chunks"] += 1

    # -- subscribe -----------------------------------------------------
    def mark_superseded(self, episode: Any) -> None:
        """A new episode arrived; the OLD plan may still be unwinding.

        The generated code block for the previous episode is plain synchronous
        Python: a boundary cannot interrupt it, so its remaining tool calls keep
        arriving and would be filed under the NEW episode, making a fresh
        episode look like it began mid-plan. Events in this window are tagged
        ``stale`` until the new episode announces its own task.
        """
        with self._lock:
            self._awaiting_plan.add(episode)

    def clear_superseded(self, episode: Any) -> None:
        with self._lock:
            self._awaiting_plan.discard(episode)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._queue_size)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def snapshot(self, limit: int = 1500) -> dict[str, Any]:
        with self._lock:
            hist = self._history[-limit:]
            return {
                "events": [e.to_dict() for e in hist],
                "episodes": list(self.episodes.values()),
                "current_episode": self._current_episode,
                "seq": self._seq,
            }


#: Process-wide bus. The env, the code env and the logger bridge all publish
#: here; the HTTP server reads it.
BUS = EventBus()


def publish(
    kind: EventKind,
    text: str = "",
    *,
    episode: Any = None,
    data: dict[str, Any] | None = None,
    images: list[str] | None = None,
) -> None:
    BUS.publish(
        Event(
            kind=kind,
            episode=episode,
            text=text,
            data=data or {},
            images=images or [],
        )
    )
