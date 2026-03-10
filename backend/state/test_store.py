from store import StateStore


def test_all():
    store = StateStore(path=":memory:")

    # 1. Fresh store
    assert store.get_status()["total_steps"] == 0
    assert store.get_latest_snapshot() is None
    assert store.get_latest_checkpoint() is None
    print("PASS: 1. Fresh store")

    # 2. Dialogue round-trip
    store.log_dialogue(
        step=1, stage=0,
        question="what is this?",
        answer="it is a dog",
        curiosity_score=0.91,
        clusters_active=["c0", "c1"],
        delta_summary={"edges_formed": [], "weight_change_magnitude": 0.02},
    )
    rows = store.get_dialogues()
    assert len(rows) == 1
    assert rows[0]["question"] == "what is this?"
    assert rows[0]["stage"] == 0
    print("PASS: 2. Dialogue round-trip")

    # 3. Graph event round-trip
    store.log_graph_event(
        step=1, event_type="CONNECT",
        cluster_a="c0", cluster_b="c1",
        metadata={"correlation": 0.84, "reason": "coactivation_threshold"},
    )
    events = store.get_graph_events(event_type="CONNECT")
    assert len(events) == 1
    assert events[0]["metadata"]["correlation"] == 0.84
    print("PASS: 3. Graph event round-trip")

    # 4. Snapshot round-trip
    graph = {"nodes": [], "clusters": [], "edges": []}
    store.log_latent_snapshot(step=50, graph_json=graph)
    snap = store.get_latest_snapshot()
    assert snap is not None
    assert "nodes" in snap
    print("PASS: 4. Snapshot round-trip")

    # 5. Status reflects all writes
    status = store.get_status()
    assert status["total_dialogues"] == 1
    assert status["total_graph_events"] == 1
    print("PASS: 5. Status reflects all writes")

    # 6. Stage filter
    store.log_dialogue(
        step=2, stage=1, question="q", answer="a",
        curiosity_score=0.5, clusters_active=[], delta_summary={},
    )
    assert len(store.get_dialogues(stage=0)) == 1
    assert len(store.get_dialogues(stage=1)) == 1
    assert len(store.get_dialogues()) == 2
    print("PASS: 6. Stage filter")

    # 7. Snapshot pruning
    for i in range(1, 25):
        store.log_latent_snapshot(step=i * 50, graph_json=graph)
    deleted = store.prune_old_snapshots(keep_every_n=10)
    assert deleted > 0
    latest = store.get_latest_snapshot()
    assert latest is not None
    print(f"PASS: 7. Snapshot pruning (deleted {deleted})")

    print("\nAll 7 tests passed.")


if __name__ == "__main__":
    test_all()
