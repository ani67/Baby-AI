"""
Migrate existing concept graph from 256-dim to 512-dim embeddings.

Uses a learned linear projection: R^256 -> R^512
Preserves all semantic relationships while expanding dimensionality.

Why linear projection and not re-encoding from scratch:
  - re-encoding loses the W matrix calibration
  - re-encoding loses the affect trace relationships
  - linear projection preserves cosine similarities exactly
  - it just adds dimensions (zero-initialized then fine-tuned)

The projection is: new_embedding = W_proj @ old_embedding
where W_proj is (512, 256):
  top 256 rows: identity matrix (preserve existing)
  bottom 256 rows: small random expansion (new dimensions, ~learned later)
"""
import os
import sys
import time
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IMPORTANT: backend.persistence._blob_to_vec enforces expected_len=D_REP
# at load time. config.D_REP is now 512 but the on-disk mind is at 256;
# MindPersistence.load would reject every embedding. Patch persistence's
# local D_REP to 256 for the load pass, do the projection in memory,
# then flip back to 512 so save uses the new dim.
from backend import persistence as _pers_mod
_pers_mod.D_REP = 256

from backend.graph import MPSConceptIndex                       # noqa: E402
from backend.mind_paths import MindPaths                        # noqa: E402
from backend.persistence import MindPersistence                 # noqa: E402


def main() -> int:
    mind_name = os.environ.get('MIND_NAME', 'first')
    paths = MindPaths(mind_name=mind_name)
    print(f"[migrate] loading {paths.db} (persistence.D_REP forced to 256 for load)")
    loop = MindPersistence.load(paths.db)
    _pers_mod.D_REP = 512   # post-load: flip back so save uses the new dim
    g = loop.graph

    n_nodes = g.node_count
    if n_nodes == 0:
        print("[migrate] graph is empty; nothing to do")
        return 0

    first_cid = next(iter(g.nodes))
    src_dim = int(g.nodes[first_cid].embedding.shape[0])
    print(f"[migrate] before: {n_nodes} nodes at {src_dim}-dim")

    if src_dim == 512:
        print("[migrate] already at 512-dim — no-op")
        return 0
    if src_dim != 256:
        raise RuntimeError(f"unexpected source dim {src_dim}; expected 256")

    rng = np.random.default_rng(seed=42)
    W_proj = np.zeros((512, 256), dtype=np.float32)
    W_proj[:256, :256] = np.eye(256, dtype=np.float32)
    W_proj[256:, :] = rng.standard_normal((256, 256)).astype(np.float32) * 0.01

    print("[migrate] projecting all concept embeddings 256 -> 512 ...")
    migrated = 0
    for cid, node in g.nodes.items():
        old_emb = node.embedding
        new_emb = W_proj @ old_emb
        n = float(np.linalg.norm(new_emb))
        if n > 1e-9:
            new_emb = new_emb / n
        node.embedding = new_emb.astype(np.float32, copy=False)
        migrated += 1
        if migrated % 10000 == 0:
            print(f"  {migrated}/{n_nodes} migrated")

    print("[migrate] expanding W matrix (12, 256) -> (12, 512)")
    old_W = loop.affect.W
    new_W = np.zeros((12, 512), dtype=np.float32)
    new_W[:, :256] = old_W
    new_W[:, 256:] = rng.standard_normal((12, 256)).astype(np.float32) * 0.001
    loop.affect.W = new_W
    loop.affect._W_pinv_cache = None  # invalidate cached inverse

    # Simulation replay buffer entries hold D_REP-shaped actual_repr.
    # Apply the same projection so a post-migration load doesn't choke
    # on a 256-dim blob in the simulation_replay table.
    sim_buf = getattr(loop.simulation, '_buffer', None) or {}
    if sim_buf:
        print(f"[migrate] projecting {len(sim_buf)} simulation replay entries ...")
        n_sim = 0
        for entry_id, entry in sim_buf.items():
            if entry.actual_repr is None:
                continue
            if entry.actual_repr.shape[0] != 256:
                continue
            new_rep = W_proj @ entry.actual_repr
            nrm = float(np.linalg.norm(new_rep))
            if nrm > 1e-9:
                new_rep = new_rep / nrm
            entry.actual_repr = new_rep.astype(np.float32, copy=False)
            n_sim += 1
        print(f"  migrated {n_sim} replay entries")

    print("[migrate] rebuilding MPS index at 512-dim ...")
    g._index = MPSConceptIndex()
    for cid, node in g.nodes.items():
        g._index.add(node.embedding, cid)

    backup_path = paths.db + '.pre-512-migration'
    print(f"[migrate] backing up {paths.db} -> {backup_path}")
    shutil.copy(paths.db, backup_path)

    print(f"[migrate] saving migrated mind to {paths.db} ...")
    MindPersistence(paths.db).save(loop, time.time())
    print(f"[migrate] complete. {migrated} concepts at 512-dim. "
          f"backup at {backup_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
