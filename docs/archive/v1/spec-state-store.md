# SPEC: State Store
*Component 1 of 9 — Foundation layer, no dependencies*

---

## What it is

A persistent SQLite database plus a Python class that wraps all reads and writes.
Every other component that needs to save or recall anything goes through this.
Nothing else touches the database file directly.

It has two jobs:
1. **Audit log** — every question asked, every answer received, every growth event, recorded forever
2. **Checkpoint** — the Baby Model can be killed and restarted; the Store lets it resume from where it left off

---

## Location in the project

```
project/
  backend/
    state/
      store.py          ← the class
      schema.sql        ← table definitions (applied on first run)
      dev.db            ← the actual SQLite file (gitignored)
```

---

## Interface

Everything goes through one object: `StateStore`.
Instantiate once at backend startup. Pass it to everything that needs it.

```python
store = StateStore(path="backend/state/dev.db")
```

No connection pooling needed. SQLite handles concurrent reads fine.
Writes are serialized by Python's GIL anyway — this is a single-process app.

---

## Tables

### `dialogues`

One row per learning loop step. The full record of what the model asked and what it learned.

```sql
CREATE TABLE IF NOT EXISTS dialogues (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,          -- unix timestamp, float
    step            INTEGER NOT NULL,          -- loop iteration number
    stage           INTEGER NOT NULL,          -- developmental stage 0-4
    question        TEXT    NOT NULL,          -- what the model asked
    answer          TEXT    NOT NULL,          -- what the teacher said
    curiosity_score REAL    NOT NULL,          -- uncertainty x novelty at time of asking
    clusters_active TEXT    NOT NULL,          -- JSON array of cluster ids that fired
    delta_summary   TEXT    NOT NULL           -- JSON: what changed in the model after this step
);
```

`delta_summary` shape:
```json
{
  "edges_formed": ["c4->c7"],
  "edges_pruned": [],
  "clusters_budded": [],
  "layers_inserted": [],
  "weight_change_magnitude": 0.034
}
```

---

### `graph_events`

One row per structural change to the Baby Model.
These are the growth operations — BUD, CONNECT, INSERT, EXTEND, PRUNE, DORMANT.
Separate from dialogues so you can query "show me all the times a new cluster formed"
without scanning the whole dialogue log.

```sql
CREATE TABLE IF NOT EXISTS graph_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    REAL    NOT NULL,
    step         INTEGER NOT NULL,
    event_type   TEXT    NOT NULL,   -- BUD | CONNECT | INSERT | EXTEND | PRUNE | DORMANT
    cluster_a    TEXT,               -- primary cluster involved (nullable for EXTEND)
    cluster_b    TEXT,               -- secondary cluster (nullable for single-cluster events)
    metadata     TEXT    NOT NULL    -- JSON, event-specific detail
);
```

`metadata` per event type:
```json
// BUD — one cluster split into two
{ "parent": "c4", "child_a": "c4a", "child_b": "c4b",
  "reason": "bimodal_activation", "node_count_before": 24 }

// CONNECT — new edge formed
{ "from": "c4", "to": "c7",
  "reason": "coactivation_threshold", "correlation": 0.84 }

// INSERT — new layer inserted between two existing ones
{ "between_a": "c2", "between_b": "c5",
  "reason": "structured_residual", "residual_variance_explained": 0.61 }

// EXTEND — new layer appended at top
{ "new_layer_id": "c11",
  "reason": "top_layer_collapse", "collapse_score": 0.77 }

// PRUNE — edge removed
{ "from": "c4", "to": "c7",
  "reason": "disuse", "steps_unused": 340, "final_strength": 0.02 }

// DORMANT — cluster suspended
{ "cluster": "c9",
  "reason": "low_activation", "steps_inactive": 500 }
```

---

### `latent_snapshots`

Periodic snapshots of the full graph state — all nodes, edges, cluster assignments,
and the 3D projected positions for the visualizer.
NOT taken every step (too expensive). Taken every N steps (default N=50).

```sql
CREATE TABLE IF NOT EXISTS latent_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL    NOT NULL,
    step          INTEGER NOT NULL,
    node_count    INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL,
    edge_count    INTEGER NOT NULL,
    graph_json    TEXT    NOT NULL
);
```

`graph_json` shape:
```json
{
  "nodes": [
    {
      "id": "n_42",
      "cluster": "c4",
      "activation_mean": 0.61,
      "activation_variance": 0.12,
      "age_steps": 204,
      "pos": [0.42, 0.18, 0.61]
    }
  ],
  "clusters": [
    {
      "id": "c4",
      "type": "integration",
      "density": 0.78,
      "node_count": 18,
      "layer_index": 1,
      "label": null
    }
  ],
  "edges": [
    {
      "from": "c4",
      "to": "c7",
      "strength": 0.88,
      "age_steps": 204,
      "direction": "bidirectional"
    }
  ]
}
```

`pos` is the UMAP-projected 3D position. Null until first projection run.
`label` on clusters is always null at this stage — human can annotate later.

---

### `model_checkpoints`

Pointers to saved Baby Model weight files on disk.
The weights themselves are NOT stored in SQLite (too large).
SQLite stores the path and metadata; the actual `.pt` file lives in `backend/state/checkpoints/`.

```sql
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL    NOT NULL,
    step          INTEGER NOT NULL,
    weights_path  TEXT    NOT NULL,
    node_count    INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL,
    edge_count    INTEGER NOT NULL,
    stage         INTEGER NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'complete',  -- 'pending' | 'complete'
    notes         TEXT
);
```

---

### `human_chat`

Messages between the human and the Baby Model.
Separate from dialogues (those are model-to-teacher).

```sql
CREATE TABLE IF NOT EXISTS human_chat (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    step            INTEGER NOT NULL,
    role            TEXT    NOT NULL,   -- "human" | "model"
    message         TEXT    NOT NULL,
    clusters_active TEXT                -- JSON array, null for human messages
);
```

---

## Python Interface

### Initialization

```python
class StateStore:
    def __init__(self, path: str = "backend/state/dev.db"):
        """
        Opens (or creates) the SQLite database at path.
        Applies schema.sql if tables don't exist yet.
        Enables WAL mode for concurrent reads.
        Creates checkpoints/ directory if absent.
        Cleans up any 'pending' checkpoints from a previous crash.
        """
```

---

### Writing

```python
def log_dialogue(
    self,
    step: int,
    stage: int,
    question: str,
    answer: str,
    curiosity_score: float,
    clusters_active: list[str],
    delta_summary: dict
) -> int:
    """Returns the new row id."""

def log_graph_event(
    self,
    step: int,
    event_type: str,           # "BUD"|"CONNECT"|"INSERT"|"EXTEND"|"PRUNE"|"DORMANT"
    cluster_a: str | None,
    cluster_b: str | None,
    metadata: dict
) -> int:

def log_latent_snapshot(
    self,
    step: int,
    graph_json: dict
) -> int:

def save_checkpoint(
    self,
    step: int,
    stage: int,
    model_state_dict: dict,
    graph_json: dict,
    notes: str | None = None
) -> str:
    """
    Atomic write:
      1. Insert row with status='pending'
      2. Write .pt file to checkpoints/step_{step}.pt
      3. Update row status to 'complete'
    Returns the weights_path.
    If process dies between steps 1 and 2, the pending row
    is cleaned up on next StateStore.__init__.
    """

def log_human_message(
    self,
    step: int,
    role: str,
    message: str,
    clusters_active: list[str] | None = None
) -> int:
```

---

### Reading

```python
def get_dialogues(
    self,
    limit: int = 50,
    offset: int = 0,
    stage: int | None = None
) -> list[dict]:

def get_graph_events(
    self,
    limit: int = 100,
    event_type: str | None = None,
    since_step: int | None = None
) -> list[dict]:

def get_latest_snapshot(self) -> dict | None:
    """Returns most recent graph_json dict, or None if none taken yet."""

def get_snapshot_at_step(self, step: int) -> dict | None:
    """Returns the snapshot closest to but not after the given step."""

def get_latest_checkpoint(self) -> dict | None:
    """Returns most recent complete checkpoint row, or None."""

def load_checkpoint(self, checkpoint_id: int) -> dict:
    """
    Returns:
      {
        "state_dict": <pytorch state dict>,
        "graph_json": <dict>,
        "step": <int>,
        "stage": <int>
      }
    Raises FileNotFoundError if .pt file is missing.
    """

def get_human_chat(
    self,
    limit: int = 50,
    offset: int = 0
) -> list[dict]:

def get_status(self) -> dict:
    """
    Returns:
      {
        "total_steps": 1042,
        "total_dialogues": 1042,
        "total_graph_events": 87,
        "total_checkpoints": 21,
        "latest_step": 1042,
        "latest_stage": 2,
        "latest_snapshot_step": 1000
      }
    """
```

---

### Maintenance

```python
def prune_old_snapshots(self, keep_every_n: int = 10) -> int:
    """
    Deletes latent_snapshots keeping only every Nth one
    plus always keeping the most recent.
    Returns number of rows deleted.
    Call this every 500 steps or so.
    """

def export_dialogue_csv(self, path: str) -> None:
    """Exports dialogues table to CSV for external analysis."""
```

---

## Behavior on First Run

```
1. Database file does not exist
2. __init__ creates it, applies schema.sql
3. Creates backend/state/checkpoints/ directory
4. WAL mode enabled
5. All read methods return empty lists or None
6. get_status() returns all zeros
```

---

## Behavior on Restart

```
1. Database file exists with existing data
2. __init__ opens it, re-applies schema (idempotent — IF NOT EXISTS)
3. Cleans up any rows with status='pending' and their orphaned .pt files
4. Backend startup calls get_latest_checkpoint()
5. If found: loads weights + graph, resumes from that step
6. If not found: starts fresh (stage 0, step 0)
7. UI gets latest snapshot for initial render
```

---

## Error handling

All write methods re-raise SQLite exceptions — callers decide whether to abort or skip.

`load_checkpoint` raises `FileNotFoundError` if `.pt` is missing.
Caller should walk backward through checkpoints until one loads successfully.

Snapshot reads returning `None` are not errors.
Viz Emitter treats None as "emit empty graph."

---

## Tests

```python
store = StateStore(path=":memory:")   # in-memory, no files created

# 1. Fresh store
assert store.get_status()["total_steps"] == 0
assert store.get_latest_snapshot() is None
assert store.get_latest_checkpoint() is None

# 2. Dialogue round-trip
store.log_dialogue(
    step=1, stage=0,
    question="what is this?",
    answer="it is a dog",
    curiosity_score=0.91,
    clusters_active=["c0", "c1"],
    delta_summary={"edges_formed": [], "weight_change_magnitude": 0.02}
)
rows = store.get_dialogues()
assert len(rows) == 1
assert rows[0]["question"] == "what is this?"
assert rows[0]["stage"] == 0

# 3. Graph event round-trip
store.log_graph_event(
    step=1, event_type="CONNECT",
    cluster_a="c0", cluster_b="c1",
    metadata={"correlation": 0.84, "reason": "coactivation_threshold"}
)
events = store.get_graph_events(event_type="CONNECT")
assert len(events) == 1
assert events[0]["metadata"]["correlation"] == 0.84

# 4. Snapshot round-trip
graph = {"nodes": [], "clusters": [], "edges": []}
store.log_latent_snapshot(step=50, graph_json=graph)
snap = store.get_latest_snapshot()
assert snap is not None
assert "nodes" in snap

# 5. Status reflects all writes
status = store.get_status()
assert status["total_dialogues"] == 1
assert status["total_graph_events"] == 1

# 6. Stage filter
store.log_dialogue(step=2, stage=1, question="q", answer="a",
    curiosity_score=0.5, clusters_active=[], delta_summary={})
assert len(store.get_dialogues(stage=0)) == 1
assert len(store.get_dialogues(stage=1)) == 1
assert len(store.get_dialogues()) == 2

# 7. Snapshot pruning
for i in range(1, 25):
    store.log_latent_snapshot(step=i*50, graph_json=graph)
deleted = store.prune_old_snapshots(keep_every_n=10)
assert deleted > 0
latest = store.get_latest_snapshot()
assert latest is not None   # most recent always kept
```

---

## Hard parts

**Snapshot size creep.**
After 10,000 steps at 50-step intervals: 200 snapshots.
With a grown model each snapshot can be 100KB+.
`prune_old_snapshots` must be called automatically by the learning loop,
not left to the human. Wire it into the loop at every 500 steps.
Consider also: store node positions separately from graph structure,
since positions change every snapshot but topology changes less often.

**Checkpoint atomicity.**
`save_checkpoint` writes a `.pt` file AND a database row.
Process death between the two leaves either an orphaned file or a broken pointer.
The `status='pending'` pattern in `model_checkpoints` handles this:
on startup, any `pending` rows and their files are deleted before normal operation resumes.

**Concurrent reads during writes.**
FastAPI handles requests concurrently via asyncio.
Enable WAL mode on init — this allows reads during writes without blocking:
```python
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
```

---

## M1-specific notes

Pure Python + standard library `sqlite3`. No PyTorch, no MLX, no Metal.
Nothing compute-intensive. Runs on CPU, trivially fast.

Keep `dev.db` and `checkpoints/` on the internal SSD.
The `.pt` files for a grown model can be 50-200MB each.
On an external USB drive, checkpoint saves and loads will be noticeably slow.

`schema.sql` is applied via `conn.executescript()` which wraps the whole
file in a transaction — either all tables are created or none are.
