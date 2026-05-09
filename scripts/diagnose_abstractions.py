"""Why are abstractions not forming?

Reads data/first/mind.db read-only (safe to run while the curriculum
process holds the writer) and analyses the similar_to subgraph against
MainLoop.sleep's abstraction-formation rule:

  ABSTRACTION_MIN_MEMBERS       = 3   members per component
  ABSTRACTION_DENSITY_THRESHOLD = 0.5 mutual edge density

A connected component qualifies iff size ≥ 3 AND density ≥ 0.5.
Density on n members = actual_pairs / (n*(n-1)/2).

Reports:
  - similar_to subgraph size and edge count
  - number of connected components, top-10 sizes
  - per-component density for the top components
  - how many components currently meet the (≥3, ≥0.5) bar
  - what threshold would have to drop to to admit the largest non-trivial
    component
"""
from __future__ import annotations

import sqlite3
import sys

import networkx as nx


DB = "data/first/mind.db"
MIN_MEMBERS = 3
DENSITY_THRESHOLD = 0.5


def main() -> int:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    edges = conn.execute(
        "SELECT source_id, target_id, weight FROM concept_edges "
        "WHERE edge_type = 'similar_to'"
    ).fetchall()
    n_nodes_total = conn.execute(
        "SELECT COUNT(*) FROM concept_nodes"
    ).fetchone()[0]
    n_edges_total = conn.execute(
        "SELECT COUNT(*) FROM concept_edges"
    ).fetchone()[0]

    # Treat similar_to as undirected — pairs are laid bidirectionally
    # but every (a, b) and (b, a) row collapses to one undirected edge.
    G = nx.Graph()
    for src, dst, w in edges:
        if src == dst:
            continue
        G.add_edge(int(src), int(dst), weight=float(w))

    print("=" * 78)
    print(" ABSTRACTION DIAGNOSIS")
    print("=" * 78)
    print(f"  graph total       : {n_nodes_total:,} nodes  /  {n_edges_total:,} edges")
    print(f"  similar_to subgraph: {G.number_of_nodes():,} nodes  /  "
          f"{G.number_of_edges():,} undirected edges")
    print(f"  abstraction bar    : size ≥ {MIN_MEMBERS}, density ≥ {DENSITY_THRESHOLD}")
    print()

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    sizes = [len(c) for c in components]
    print(f"  connected components: {len(components):,}")
    print(f"  top-10 sizes        : {sizes[:10]}")
    print(f"  size distribution   :")
    bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 50), (51, 200), (201, 1000),
            (1001, 10**9)]
    for lo, hi in bins:
        c = sum(1 for s in sizes if lo <= s <= hi)
        if c:
            print(f"      size {lo:>4d}-{hi if hi < 10**8 else '∞':>4}: "
                  f"{c:>5d} components")
    print()

    # Density of the top 10 components
    print(f"  density of top-10 by size:")
    print(f"  {'#':>3s}  {'size':>5s}  {'edges':>6s}  {'max_pairs':>9s}  "
          f"{'density':>8s}  {'meets_thresh':>12s}")
    for i, comp in enumerate(components[:10]):
        sub = G.subgraph(comp)
        n = sub.number_of_nodes()
        e = sub.number_of_edges()
        max_pairs = n * (n - 1) // 2
        density = nx.density(sub)
        ok = (n >= MIN_MEMBERS) and (density >= DENSITY_THRESHOLD)
        flag = "YES" if ok else ""
        print(f"  {i:>3d}  {n:>5d}  {e:>6d}  {max_pairs:>9d}  "
              f"{density:>8.4f}  {flag:>12s}")
    print()

    # Across ALL components, count how many qualify
    qualifying = 0
    near_misses_size = 0
    near_misses_density = 0
    eligible_density_breakdown = []
    for comp in components:
        sub = G.subgraph(comp)
        n = sub.number_of_nodes()
        if n < MIN_MEMBERS:
            near_misses_size += 1
            continue
        density = nx.density(sub)
        eligible_density_breakdown.append((n, density))
        if density >= DENSITY_THRESHOLD:
            qualifying += 1
        else:
            near_misses_density += 1

    print(f"  ABSTRACTION-ELIGIBLE TODAY: {qualifying:,} components")
    print(f"    components with size < 3   (skipped):       {near_misses_size:,}")
    print(f"    components with size ≥ 3 but density < 0.5: {near_misses_density:,}")
    print()

    # Density distribution among size>=3 components
    if eligible_density_breakdown:
        densities = [d for _, d in eligible_density_breakdown]
        print(f"  density distribution among size-≥-{MIN_MEMBERS} components:")
        for lo, hi in [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3),
                       (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.0)]:
            c = sum(1 for d in densities if lo <= d < hi)
            if c:
                print(f"      density {lo:.2f}-{hi:.2f}: {c:>5d}")
        print()
        max_d = max(densities)
        print(f"  highest density seen on a size-≥-3 component: {max_d:.4f}")
        # Find the threshold at which 1, 5, 20 components would qualify
        sorted_d = sorted(densities, reverse=True)
        for n_qual in (1, 5, 20, 50, 100):
            if n_qual <= len(sorted_d):
                print(f"  threshold to admit {n_qual:>4d} components: "
                      f"≥ {sorted_d[n_qual - 1]:.4f}")

    print()
    print("=" * 78)
    print(" INTERPRETATION")
    print("=" * 78)

    # Under K=1 auto-link, each new concept adds exactly 1 (or 0) edge to
    # the existing graph — so connected components grow as TREES. A tree
    # on n nodes has n-1 edges; max_pairs is n(n-1)/2; density = 2/n.
    # That hits 0.5 only at n ≤ 4. For n=5, tree density = 2/5 = 0.4. For
    # n=20, density = 2/20 = 0.1.
    print("  At AUTO_LINK_K=1 each fresh concept adds 1 edge into the existing")
    print("  similar_to subgraph. Components grow as TREES — tree on n nodes")
    print("  has density 2/n. Hits the 0.5 floor only at n ≤ 4. At n=10, density")
    print("  is 0.2; at n=20, 0.1; at n=100, 0.02. Density CAN'T cross 0.5 for")
    print("  any non-trivial component grown this way without densifying edges")
    print("  from elsewhere (replays, explicit add_edge calls).")
    print()
    print("  At AUTO_LINK_K=2 (each new concept lays 2 pair edges) the")
    print("  expected density is roughly 4/n — same shape, half-life'd. At")
    print("  AUTO_LINK_K=3 (the v0.4 default) it's roughly 6/n. So even the")
    print("  v0.4 setting only hit 0.5 for components of size ≤ 12 — and")
    print("  the BIG components (where abstraction would matter) never make it.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
