def compute_diff(old_json: dict, new_json: dict) -> dict:
    """
    Computes the structural difference between two graph states.
    Returns only what changed — not the full graph.
    """
    if not old_json:
        return {
            "nodes_added": [],
            "nodes_removed": [],
            "edges_added": new_json.get("edges", []),
            "edges_removed": [],
            "edges_updated": [],
            "clusters_added": new_json.get("clusters", []),
            "clusters_removed": [],
            "clusters_updated": [],
        }

    old_clusters = {c["id"]: c for c in old_json.get("clusters", [])}
    new_clusters = {c["id"]: c for c in new_json.get("clusters", [])}
    old_edges = {(e["from"], e["to"]): e for e in old_json.get("edges", [])}
    new_edges = {(e["from"], e["to"]): e for e in new_json.get("edges", [])}
    old_nodes = {n["id"]: n for n in old_json.get("nodes", [])}
    new_nodes = {n["id"]: n for n in new_json.get("nodes", [])}

    return {
        "nodes_added": [
            new_nodes[k] for k in new_nodes if k not in old_nodes
        ],
        "nodes_removed": [
            old_nodes[k] for k in old_nodes if k not in new_nodes
        ],
        "edges_added": [
            new_edges[k] for k in new_edges if k not in old_edges
        ],
        "edges_removed": [
            old_edges[k] for k in old_edges if k not in new_edges
        ],
        "edges_updated": [
            new_edges[k] for k in new_edges
            if k in old_edges
            and abs(new_edges[k]["strength"] - old_edges[k]["strength"]) > 0.02
        ],
        "clusters_added": [
            new_clusters[k] for k in new_clusters if k not in old_clusters
        ],
        "clusters_removed": [
            old_clusters[k] for k in old_clusters if k not in new_clusters
        ],
        "clusters_updated": [
            new_clusters[k] for k in new_clusters
            if k in old_clusters
            and new_clusters[k] != old_clusters[k]
        ],
    }
