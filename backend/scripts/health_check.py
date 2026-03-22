#!/usr/bin/env python3
"""
Health check script — run against a live Baby AI server.
Reports all key metrics in one shot.

Usage:
    python3 -m scripts.health_check
    python3 -m scripts.health_check --url http://localhost:8000
"""

import argparse
import json
import sys
import urllib.request


def fetch(url):
    try:
        r = urllib.request.urlopen(url, timeout=5)
        return json.loads(r.read())
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Baby AI health check")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.url

    print("=" * 60)
    print("  BABY AI HEALTH CHECK")
    print("=" * 60)

    # 1. Server status
    status = fetch(f"{base}/status")
    if not status:
        print("\n  ✗ Server not reachable at", base)
        sys.exit(1)

    step = status["step"]
    state = status["state"]
    gs = status["graph_summary"]
    clusters = gs["cluster_count"]
    nodes = gs["node_count"]
    edges = gs["edge_count"]
    layers = gs["layer_count"]
    dormant = gs["dormant_count"]

    print(f"\n  1. SERVER")
    print(f"     State: {state}")
    print(f"     Step:  {step}")
    print(f"     Stage: {status['stage']}")

    # 2. Graph size
    print(f"\n  2. GRAPH")
    print(f"     Clusters: {clusters} ({dormant} dormant)")
    print(f"     Nodes:    {nodes}  ({nodes/max(clusters,1):.1f} per cluster)")
    print(f"     Edges:    {edges}  ({edges/max(clusters,1):.1f} per cluster)")
    print(f"     Layers:   {layers}")

    growth_rate = "unknown"
    if step > 0:
        growth_rate = f"{clusters / (step/1000):.1f} clusters per 1K steps"
    print(f"     Growth:   {growth_rate}")

    # Grade graph
    if clusters < 10 and step > 5000:
        print(f"     ⚠ STALLED — only {clusters} clusters at {step} steps")
    elif clusters > 500:
        print(f"     ⚠ EXPLOSION — {clusters} clusters, may be too many")
    else:
        print(f"     ✓ Healthy")

    # 3. Signal balance (from a debug cluster)
    labels = fetch(f"{base}/clusters/labels")
    if labels and labels.get("labels"):
        sample_id = sorted(labels["labels"].keys())[0]
        debug = fetch(f"{base}/debug/cluster/{sample_id}")
        if debug:
            pos = debug["positive_steps"]
            neg = debug["negative_steps"]
            total = pos + neg
            print(f"\n  3. SIGNAL BALANCE (cluster {sample_id}, last 500 dialogues)")
            print(f"     Positive: {pos}  Negative: {neg}")
            if total > 0:
                ratio = pos / total * 100
                print(f"     Ratio:    {ratio:.0f}% / {100-ratio:.0f}%")
                if 35 <= ratio <= 65:
                    print(f"     ✓ Healthy (target: 35-65%)")
                else:
                    print(f"     ⚠ IMBALANCED (target: 35-65%)")
            else:
                print(f"     — Not enough data yet")
        else:
            print(f"\n  3. SIGNAL BALANCE — debug endpoint unavailable")
    else:
        print(f"\n  3. SIGNAL BALANCE — no clusters yet")

    # 4. Activation diversity
    print(f"\n  4. ACTIVATION DIVERSITY")
    if clusters > 0:
        active_per_step = gs.get("active_tiles", clusters)
        pct = active_per_step / max(clusters, 1) * 100
        print(f"     Active tiles: {active_per_step} / {clusters} ({pct:.0f}%)")
        if pct > 50:
            print(f"     ⚠ TOO MANY firing — resonance may be too permissive")
        elif pct < 5:
            print(f"     ⚠ TOO FEW firing — resonance may be too strict")
        else:
            print(f"     ✓ Healthy (target: 10-30%)")

    # 5. Growth operations
    print(f"\n  5. GROWTH")
    tree = fetch(f"{base}/clusters/tree")
    if tree and tree.get("nodes"):
        depths = {}
        phantoms = 0
        for n in tree["nodes"]:
            d = n["depth"]
            depths[d] = depths.get(d, 0) + 1
            if n.get("phantom"):
                phantoms += 1
        print(f"     BUD tree depth: {tree['max_depth']}")
        for d in sorted(depths):
            print(f"       depth {d}: {depths[d]} clusters")
        print(f"     Phantom parents: {phantoms}")

        if tree["max_depth"] == 0 and step > 5000:
            print(f"     ⚠ NO BUD SPLITS after {step} steps")
        else:
            print(f"     ✓ BUD is working")

        if layers < 3 and step > 10000:
            print(f"     ⚠ EXTEND not triggering — stuck at {layers} layers")
        elif layers >= 3:
            print(f"     ✓ EXTEND is working ({layers} layers)")
        else:
            print(f"     — EXTEND waiting (need 30+ clusters, currently {clusters})")
    else:
        print(f"     — Tree data unavailable")

    # 6. Edge health
    print(f"\n  6. EDGES")
    edge_ratio = edges / max(clusters, 1)
    print(f"     Edge ratio: {edge_ratio:.1f} edges per cluster")
    if edge_ratio > 6:
        print(f"     ⚠ HIGH — edges may be saturating (Oja should prevent this)")
    elif edge_ratio < 1.5:
        print(f"     ⚠ LOW — not enough connectivity")
    else:
        print(f"     ✓ Healthy (target: 1.5-6.0)")

    cofiring = fetch(f"{base}/clusters/cofiring")
    if cofiring and cofiring.get("pairs"):
        pairs = cofiring["pairs"]
        print(f"     Co-firing pairs: {len(pairs)}")
        if pairs:
            strengths = [p["strength"] for p in pairs[:100]]
            avg = sum(strengths) / len(strengths)
            print(f"     Top-100 avg strength: {avg:.3f}")
    else:
        print(f"     Co-firing: no data yet")

    # 7. Emergent labels
    print(f"\n  7. EMERGENT LABELS")
    if labels and labels.get("labels"):
        has_labels = sum(1 for v in labels["labels"].values() if v)
        total_c = len(labels["labels"])
        print(f"     Clusters with labels: {has_labels}/{total_c}")
        if has_labels > 0:
            sample = list(labels["labels"].items())[:5]
            for cid, words in sample:
                print(f"       {cid}: {words}")
            print(f"     ✓ Labels emerging")
        else:
            print(f"     — No labels yet (need more training)")
    else:
        print(f"     — Labels unavailable")

    # Summary
    print(f"\n" + "=" * 60)
    print(f"  SUMMARY: step={step} clusters={clusters} layers={layers}")
    print(f"  nodes/cluster={nodes/max(clusters,1):.1f} edges/cluster={edge_ratio:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
