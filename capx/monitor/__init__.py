"""Real-robot experiment monitor for CaP-X.

A read-only dashboard for watching a live real-robot session: which episode the
operator is on, what the driver is sending, what the LLM generated, which tools
ran, and what the arm was commanded to do.

Design constraints:

* **Episode-aware.** A real session is many episodes -- the operator stops,
  rearranges the scene, and starts again. The driver signals this by
  incrementing ``episode_id``. The monitor keeps per-episode history rather than
  assuming a single run, so you can compare episode 3 with episode 7.
* **Observability only.** Nothing here can affect the robot. Event delivery is
  best-effort behind a bounded queue; a closed browser or a slow client cannot
  delay an action chunk.
* **No CaP-X internals changed.** Tool steps arrive through the existing
  ``capx.utils.execution_logger`` callback hook, which the APIs already call.
"""

from .events import Event, EventBus, EventKind
from .server import MonitorServer, start_monitor

__all__ = ["Event", "EventBus", "EventKind", "MonitorServer", "start_monitor"]
