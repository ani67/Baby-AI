"""S7.4 — manual abstraction pass + save."""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")

from backend.mind_paths import MindPaths              # noqa: E402
from backend.persistence import MindPersistence       # noqa: E402


def main() -> int:
    paths = MindPaths("first")
    print(f"loading {paths.db}")
    loop = MindPersistence.load(paths.db)
    now = time.time()
    pre_nodes = loop.graph.node_count
    pre_edges = loop.graph.edge_count
    pre_isa   = sum(
        1 for e in loop.graph._edges.values()
        if e.type.value == "is_a"
    )
    print(f"before: {pre_nodes:,} nodes / {pre_edges:,} edges / "
          f"{pre_isa:,} is_a edges")

    abstractions = loop._form_abstractions(now)
    post_nodes = loop.graph.node_count
    post_edges = loop.graph.edge_count
    post_isa   = sum(
        1 for e in loop.graph._edges.values()
        if e.type.value == "is_a"
    )

    print(f"abstractions formed: {abstractions:,}")
    print(f"after:  {post_nodes:,} nodes / {post_edges:,} edges / "
          f"{post_isa:,} is_a edges")
    print(f"delta:  +{post_nodes-pre_nodes} nodes, "
          f"+{post_edges-pre_edges} edges, "
          f"+{post_isa-pre_isa} is_a")

    print("saving …")
    MindPersistence(paths.db).save(loop, now=time.time())
    print("saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
