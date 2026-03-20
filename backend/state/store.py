import csv
import json
import os
import sqlite3
import time


class StateStore:
    def __init__(self, path: str = "backend/state/dev.db"):
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row

        # WAL mode for concurrent reads
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Apply schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path) as f:
            self._conn.executescript(f.read())

        # Create checkpoints directory
        if path != ":memory:":
            ckpt_dir = os.path.join(os.path.dirname(path), "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

        # Clean up pending checkpoints
        pending = self._conn.execute(
            "SELECT id, weights_path FROM model_checkpoints WHERE status='pending'"
        ).fetchall()
        for row in pending:
            try:
                os.remove(row["weights_path"])
            except OSError:
                pass
            self._conn.execute("DELETE FROM model_checkpoints WHERE id=?", (row["id"],))
        self._conn.commit()

    # ── Writing ──

    def log_dialogue(
        self,
        step: int,
        stage: int,
        question: str,
        answer: str,
        curiosity_score: float,
        clusters_active: list,
        delta_summary: dict,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO dialogues
               (timestamp, step, stage, question, answer, curiosity_score, clusters_active, delta_summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                step,
                stage,
                question,
                answer,
                curiosity_score,
                json.dumps(clusters_active),
                json.dumps(delta_summary),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_graph_event(
        self,
        step: int,
        event_type: str,
        cluster_a: str | None,
        cluster_b: str | None,
        metadata: dict,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO graph_events
               (timestamp, step, event_type, cluster_a, cluster_b, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (time.time(), step, event_type, cluster_a, cluster_b, json.dumps(metadata)),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_latent_snapshot(self, step: int, graph_json: dict) -> int:
        nodes = graph_json.get("nodes", [])
        clusters = graph_json.get("clusters", [])
        edges = graph_json.get("edges", [])
        cur = self._conn.execute(
            """INSERT INTO latent_snapshots
               (timestamp, step, node_count, cluster_count, edge_count, graph_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                step,
                len(nodes),
                len(clusters),
                len(edges),
                json.dumps(graph_json),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def save_checkpoint(
        self,
        step: int,
        stage: int,
        model_state_dict: dict,
        graph_json: dict,
        notes: str | None = None,
    ) -> str:
        nodes = graph_json.get("nodes", [])
        clusters = graph_json.get("clusters", [])
        edges = graph_json.get("edges", [])

        if self._path == ":memory:":
            weights_path = f"checkpoints/step_{step}.pt"
        else:
            ckpt_dir = os.path.join(os.path.dirname(self._path), "checkpoints")
            weights_path = os.path.join(ckpt_dir, f"step_{step}.pt")

        # 1. Insert pending row
        cur = self._conn.execute(
            """INSERT INTO model_checkpoints
               (timestamp, step, weights_path, node_count, cluster_count, edge_count, stage, status, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (
                time.time(),
                step,
                weights_path,
                len(nodes),
                len(clusters),
                len(edges),
                stage,
                notes,
            ),
        )
        row_id = cur.lastrowid
        self._conn.commit()

        # 2. Write file (skip for in-memory databases)
        if self._path != ":memory:":
            import pickle

            data = {"state_dict": model_state_dict, "graph_json": graph_json}
            with open(weights_path, "wb") as f:
                pickle.dump(data, f)

        # 3. Mark complete
        self._conn.execute(
            "UPDATE model_checkpoints SET status='complete' WHERE id=?", (row_id,)
        )
        self._conn.commit()

        return weights_path

    def log_human_message(
        self,
        step: int,
        role: str,
        message: str,
        clusters_active: list | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO human_chat
               (timestamp, step, role, message, clusters_active)
               VALUES (?, ?, ?, ?, ?)""",
            (
                time.time(),
                step,
                role,
                message,
                json.dumps(clusters_active) if clusters_active is not None else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    # ── Reading ──

    def get_dialogues(
        self, limit: int = 50, offset: int = 0, stage: int | None = None
    ) -> list[dict]:
        if stage is not None:
            rows = self._conn.execute(
                "SELECT * FROM dialogues WHERE stage=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (stage, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM dialogues ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_dict(r, json_fields=("clusters_active", "delta_summary")) for r in rows]

    def get_graph_events(
        self,
        limit: int = 100,
        event_type: str | None = None,
        since_step: int | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM graph_events WHERE 1=1"
        params: list = []
        if event_type is not None:
            query += " AND event_type=?"
            params.append(event_type)
        if since_step is not None:
            query += " AND step>=?"
            params.append(since_step)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(r, json_fields=("metadata",)) for r in rows]

    def get_latest_snapshot(self) -> dict | None:
        row = self._conn.execute(
            "SELECT graph_json FROM latent_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["graph_json"])

    def get_snapshot_at_step(self, step: int) -> dict | None:
        row = self._conn.execute(
            "SELECT graph_json FROM latent_snapshots WHERE step<=? ORDER BY step DESC LIMIT 1",
            (step,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["graph_json"])

    def get_latest_checkpoint(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM model_checkpoints WHERE status='complete' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def load_checkpoint(self, checkpoint_id: int) -> dict:
        row = self._conn.execute(
            "SELECT * FROM model_checkpoints WHERE id=?", (checkpoint_id,)
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"No checkpoint with id {checkpoint_id}")

        import pickle

        weights_path = row["weights_path"]
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file missing: {weights_path}")

        with open(weights_path, "rb") as f:
            data = pickle.load(f)

        return {
            "state_dict": data["state_dict"],
            "graph_json": data["graph_json"],
            "step": row["step"],
            "stage": row["stage"],
        }

    def get_human_chat(self, limit: int = 50, offset: int = 0) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM human_chat ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_dict(r, json_fields=("clusters_active",)) for r in rows]

    def get_status(self) -> dict:
        total_dialogues = self._conn.execute("SELECT COUNT(*) FROM dialogues").fetchone()[0]
        total_graph_events = self._conn.execute("SELECT COUNT(*) FROM graph_events").fetchone()[0]
        total_checkpoints = self._conn.execute(
            "SELECT COUNT(*) FROM model_checkpoints WHERE status='complete'"
        ).fetchone()[0]

        latest_dialogue = self._conn.execute(
            "SELECT step, stage FROM dialogues ORDER BY id DESC LIMIT 1"
        ).fetchone()

        latest_snapshot = self._conn.execute(
            "SELECT step FROM latent_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()

        return {
            "total_steps": latest_dialogue["step"] if latest_dialogue else 0,
            "total_dialogues": total_dialogues,
            "total_graph_events": total_graph_events,
            "total_checkpoints": total_checkpoints,
            "latest_step": latest_dialogue["step"] if latest_dialogue else 0,
            "latest_stage": latest_dialogue["stage"] if latest_dialogue else 0,
            "latest_snapshot_step": latest_snapshot["step"] if latest_snapshot else 0,
        }

    def clear_for_reset(self):
        """Delete all dialogues, graph_events, model_checkpoints, and latent_snapshots."""
        self._conn.execute("DELETE FROM dialogues")
        self._conn.execute("DELETE FROM graph_events")
        self._conn.execute("DELETE FROM model_checkpoints")
        self._conn.execute("DELETE FROM latent_snapshots")
        self._conn.execute("DELETE FROM human_chat")
        self._conn.commit()

    # ── Maintenance ──

    def prune_old_snapshots(self, keep_every_n: int = 10) -> int:
        rows = self._conn.execute(
            "SELECT id, step FROM latent_snapshots ORDER BY step ASC"
        ).fetchall()
        if not rows:
            return 0

        # Always keep the most recent
        most_recent_id = rows[-1]["id"]

        to_delete = []
        for i, row in enumerate(rows):
            if row["id"] == most_recent_id:
                continue
            # Keep every Nth snapshot (0-indexed position in ordered list)
            if (i + 1) % keep_every_n != 0:
                to_delete.append(row["id"])

        if not to_delete:
            return 0

        self._conn.execute(
            f"DELETE FROM latent_snapshots WHERE id IN ({','.join('?' * len(to_delete))})",
            to_delete,
        )
        self._conn.commit()
        return len(to_delete)

    def get_recent_dialogues_for_clusters(self, limit: int = 500) -> list[tuple[str, str]]:
        """
        Return (clusters_active JSON, answer text) for the most recent dialogues.
        Used by /clusters/labels to compute emergent labels.
        """
        rows = self._conn.execute(
            "SELECT clusters_active, answer FROM dialogues ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(row["clusters_active"], row["answer"]) for row in rows]

    def batch_update_cofiring(self, pairs: list[tuple[str, str]], step: int) -> None:
        """Increment co-firing counts for a batch of cluster pairs."""
        for a, b in pairs:
            # Canonical ordering so (a,b) and (b,a) map to the same row
            lo, hi = (a, b) if a < b else (b, a)
            self._conn.execute(
                """INSERT INTO cluster_cofiring (cluster_a, cluster_b, count, last_updated)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(cluster_a, cluster_b)
                   DO UPDATE SET count = count + 1, last_updated = ?""",
                (lo, hi, step, step),
            )
        self._conn.commit()

    def get_cofiring_pairs(self) -> list[dict]:
        """Return all co-firing pairs with counts."""
        rows = self._conn.execute(
            "SELECT cluster_a, cluster_b, count, last_updated FROM cluster_cofiring ORDER BY count DESC"
        ).fetchall()
        return [{"a": r["cluster_a"], "b": r["cluster_b"], "count": r["count"], "last_updated": r["last_updated"]} for r in rows]

    def export_dialogue_csv(self, path: str) -> None:
        rows = self._conn.execute("SELECT * FROM dialogues ORDER BY id ASC").fetchall()
        if not rows:
            return
        fieldnames = rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

    # ── Internal ──

    def _row_to_dict(self, row: sqlite3.Row, json_fields: tuple = ()) -> dict:
        d = dict(row)
        for field in json_fields:
            if field in d and d[field] is not None:
                d[field] = json.loads(d[field])
        return d
