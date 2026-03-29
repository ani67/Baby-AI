"""
Shared state between training process and HTTP server.

The training process writes state to a JSON file after each batch.
The HTTP server reads it for /status, /metrics, and WebSocket updates.
This decouples training (MPS GPU, heavy computation) from serving
(must be responsive, never blocked).

File-based IPC is simple, robust, and zero-dependency:
  - Atomic writes via rename (no partial reads)
  - No shared memory complexity
  - Works across process restarts
  - Human-readable for debugging
"""

import json
import os
import tempfile
import time


STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "shared_state.json")


def write_state(state: dict) -> None:
    """Atomically write state to shared file. Called by training process."""
    state["_timestamp"] = time.time()
    # Write to temp file then rename (atomic on POSIX)
    dir_name = os.path.dirname(STATE_FILE)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def read_state() -> dict | None:
    """Read shared state. Called by HTTP server. Returns None if no state yet."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def state_age() -> float:
    """Seconds since last state write. Returns inf if no state."""
    try:
        return time.time() - os.path.getmtime(STATE_FILE)
    except FileNotFoundError:
        return float("inf")
