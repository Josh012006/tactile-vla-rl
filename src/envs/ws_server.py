"""Synchronous WebSocket server that exposes a lockstep vector-environment backend (protocol.py).

Knows nothing about Isaac Lab: it dispatches to a `Backend` object (backend.ManagerBackend). Two properties that matter for correctness:

  * Threading. Isaac Sim / Kit must be driven from the thread that created AppLauncher; calling env.step() from a
    connection thread does not raise, it hangs. Network I/O therefore stays on the connection threads, and every
    Backend call is executed on the thread that runs serve_forever() (the main thread in server.py), through a queue.
  * ONE client at a time. The backend holds the episodes in flight; two clients interleaving their requests would corrupt
    them, so a second connection is refused with an error message instead of being served. A new client must call `reset`
    first (the previous client may have left episodes half-way).
"""
from __future__ import annotations

import inspect
import logging
import queue
import threading
import traceback
from typing import Any, Dict, Optional, Protocol

import websockets.exceptions
import websockets.sync.server

from . import wire
from .protocol import PROTOCOL_VERSION

log = logging.getLogger("env_server")


class Backend(Protocol):
    """Every method runs on the thread that called WsEnvServer.serve_forever()."""

    def spec(self) -> dict: ...                                   # protocol.EnvSpec.to_wire()
    def reset(self, request: dict) -> dict: ...                   # request: env_ids, bank_ids, seed -> StepResult in wire form
    def step(self, request: dict) -> dict: ...                    # request: actions, auto_reset -> StepResult in wire form
    def observe(self) -> dict: ...                                # {"obs": ...}
    def debug(self, request: dict) -> Any: ...                    # optional diagnostics, simulation-only


class WsEnvServer:
    def __init__(self, backend: Backend, host: str = "0.0.0.0", port: int = 8102):
        self.backend = backend
        self.host = host
        self.port = port                       # 0 = pick a free port; the real one is set when `ready` is
        self.ready = threading.Event()
        self._work: "queue.Queue[tuple[dict, queue.Queue]]" = queue.Queue()
        self._stop = threading.Event()
        self._client_lock = threading.Lock()
        self._ws: Optional[Any] = None

    # -- main thread ---------------------------------------------------------------------------------------------
    def _dispatch(self, request: dict) -> Dict[str, Any]:
        op = request.get("op")
        try:
            if op == "spec":
                return {"status": "ok", "spec": self.backend.spec()}
            if op == "reset":
                return {"status": "ok", **self.backend.reset(request)}
            if op == "step":
                return {"status": "ok", **self.backend.step(request)}
            if op == "observe":
                return {"status": "ok", **self.backend.observe()}
            if op == "debug":
                fn = getattr(self.backend, "debug", None)
                if fn is None:
                    return {"status": "error", "message": "backend has no debug()"}
                return {"status": "ok", "result": fn(request)}
            return {"status": "error", "message": f"unknown op: {op!r} (protocol version {PROTOCOL_VERSION})"}
        except Exception as e:  # noqa: BLE001 -- must always answer, never kill the loop
            log.error("op %s failed:\n%s", op, traceback.format_exc())
            return {"status": "error", "message": f"{type(e).__name__}: {e}"}

    def serve_forever(self):
        """Start the accept loop on a background thread, then pump the work queue on THIS thread until stop()."""
        threading.Thread(target=self._run_ws, daemon=True, name="ws-accept").start()
        while not self._stop.is_set():
            try:
                request, result = self._work.get(timeout=0.2)
            except queue.Empty:
                continue
            result.put(self._dispatch(request))

    def stop(self):
        self._stop.set()
        if self._ws is not None:
            self._ws.shutdown()

    # -- connection threads --------------------------------------------------------------------------------------
    def _handle(self, request: dict) -> dict:
        result: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        self._work.put((request, result))
        return result.get()

    def _connection(self, ws):
        if not self._client_lock.acquire(blocking=False):
            ws.send(wire.pack({"status": "error", "message": "another client is already connected to this env server"}))
            return
        log.info("client connected")
        try:
            for raw in ws:
                ws.send(wire.pack(self._handle(wire.unpack(raw))))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._client_lock.release()
            log.info("client disconnected")

    def _run_ws(self):
        kwargs = dict(compression=None, max_size=None)
        # A client may be busy for minutes (a model update, an XLA compile): no keep-alive pings, they would close the link.
        if "ping_interval" in inspect.signature(websockets.sync.server.serve).parameters:
            kwargs["ping_interval"] = None
        with websockets.sync.server.serve(self._connection, self.host, self.port, **kwargs) as server:
            self._ws = server
            self.port = server.socket.getsockname()[1]
            log.info("env server listening on ws://%s:%s", self.host, self.port)
            self.ready.set()
            server.serve_forever()
