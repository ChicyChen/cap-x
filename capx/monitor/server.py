"""HTTP + SSE server for the real-robot monitor.

Stdlib only (``http.server``) so it adds no dependency and cannot conflict with
vLLM/torch pins in this env. Server-Sent Events rather than WebSocket: the feed
is one-directional, SSE reconnects automatically in the browser, and it needs no
extra library.

Runs on a daemon thread. Every handler is wrapped so a browser cannot raise into
the robot's process.
"""

from __future__ import annotations

import http.server
import json
import logging
import threading
import time
from typing import Any, Optional

from .events import BUS, EventKind
from .ui import DASHBOARD_HTML

logger = logging.getLogger(__name__)

_SERVER: Optional["MonitorServer"] = None


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the robot log readable
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        try:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/healthz":
                self._send(200, b"OK\n", "text/plain")
            elif path == "/api/snapshot":
                body = json.dumps(BUS.snapshot()).encode()
                self._send(200, body, "application/json")
            elif path == "/api/stream":
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logger.debug("monitor request failed", exc_info=True)

    def _stream(self) -> None:
        """Server-Sent Events feed."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = BUS.subscribe()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                    payload = json.dumps(ev.to_dict())
                    self.wfile.write(f"data: {payload}\n\n".encode())
                except Exception as exc:  # timeout -> keepalive
                    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                        raise
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            BUS.unsubscribe(q)


class _ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MonitorServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8300) -> None:
        self.host = host
        self.port = port
        self._httpd = _ThreadingHTTPServer((host, port), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> "MonitorServer":
        self._thread.start()
        return self


def start_monitor(
    host: str = "0.0.0.0",
    port: int = 8300,
    *,
    endpoint: str | None = None,
    model: str | None = None,
) -> Optional[MonitorServer]:
    """Start the monitor once per process; safe to call repeatedly.

    Returns None (and logs) if the port is taken -- monitoring must never stop a
    robot run from starting.
    """
    global _SERVER
    if _SERVER is not None:
        return _SERVER
    try:
        _SERVER = MonitorServer(host, port).start()
    except OSError as exc:
        print(f"[monitor] not started ({exc}); the run continues", flush=True)
        return None
    BUS.publish_session = True  # marker for tests
    from .events import Event

    BUS.publish(
        Event(
            kind=EventKind.SESSION,
            text="monitor started",
            data={
                "endpoint": endpoint,
                "model": model,
                "started": time.time(),
            },
        )
    )
    print(f"[monitor] dashboard on http://{host}:{port}/", flush=True)
    return _SERVER
