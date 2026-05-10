"""Active inference — the mind generates its own questions in idle time.

What this is
------------
When the runtime is idle (no fresh stimulus, arousal low), MainLoop already
runs D.replay_loop to re-experience surprising memories. That's reactive:
it consolidates what already arrived. Active inference is the complement —
the mind picks two concepts that have been recently active, synthesizes a
"bridge" representation between them, and runs the predict/observe loop on
that bridge as if it had just been perceived. If the bridge surprises B,
the existing observe() path writes a new concept and auto-links it. The
mind has, in effect, asked itself a question and discovered something the
graph didn't already cover.

Why this layer exists
---------------------
This is the architecture's first concrete answer to "what does the mind do
when nothing is happening to it?" Replay alone is rumination. Active
inference is generative — the mind composes a new candidate from things
it already cares about (activation_count is the proxy for "currently in
awareness") and tests whether the composition is novel. Surprise is still
the only teacher; we just give the system a way to manufacture inputs
when the world is quiet.

What this does NOT do
---------------------
- It does not invent edges by hand. New edges arrive through the standard
  write_on_surprise → link_to_nearest_neighbors path inside observe().
- It does not bypass the surprise threshold. If the bridge isn't novel
  (the graph already has a concept within R_MATCH), nothing is written.
- It does not pick pairs that are already directly connected — that would
  be wasted work since their relationship is already represented.
- It does not pick at random uniformly. Sampling weights are activation
  counts, so the more a concept has been thought-about, the more likely
  it is to be a bridge endpoint. Recency-of-mind drives generative bias.

Sampling
--------
- Eligible concepts: any node in the graph with activation_count > 0.
- Per-cycle selection: K_PAIRS pairs, sampled without replacement on
  activation-count weights (uniform fallback when all weights are 0).
- Existing-pair check: the unordered (a, b) pair is skipped if any edge
  of any EdgeType exists in either direction — already represented.

Bridge
------
The bridge representation is the L2-normalized midpoint of the two
endpoints' embeddings. It's a deliberately simple composition: not a
learned operator, just "what's halfway between these two things in
representation space?" If observe() finds this point genuinely novel,
the concept that gets written sits between them; auto-link will then
SIMILAR_TO it to its true cosine neighbors, which may or may not be
the original pair.
"""
from __future__ import annotations

import numpy as np

from backend.affect import AffectStack
from backend.graph import ConceptGraph, EdgeType
from backend.predict import PredictionEngine


# How many bridge attempts per idle cycle. Cheap — each is one predict()
# (top-1 FAISS search) and one observe() (vector arithmetic + maybe a
# write). Default of 4 keeps the per-cycle budget bounded.
K_PAIRS = 4

# Minimum activation_count for a concept to be eligible as a bridge
# endpoint. activation_count == 0 means the node was written but never
# strengthened or revisited — there's no recent-mind signal to use it.
MIN_ACTIVATION_COUNT = 1


class ActiveInference:
    """Generate predict/observe traces from concept pairs the graph
    already knows about. Producer of self-driven novelty during idle.
    """

    def __init__(
        self,
        *,
        graph: ConceptGraph,
        predict_engine: PredictionEngine,
        affect: AffectStack,
        rng_seed: int = 0,
    ) -> None:
        self.graph = graph
        self.predict_engine = predict_engine
        self.affect = affect
        self._rng = np.random.default_rng(rng_seed)

    # ---- one cycle of inference ----

    def run_inference_cycle(self, now: float) -> tuple[int, int]:
        """Run K_PAIRS bridge attempts.

        Returns (events, new_writes):
          events       — number of (a, b) pairs we actually probed (i.e.
                         passed the eligibility + not-already-connected
                         filters and ran predict/observe on).
          new_writes   — subset of events where observe() ended up writing
                         a new concept (gap.was_new_write True).

        Two-pass structure
        ------------------
        1. Pass 1 (sample): draw up to K_PAIRS bridges that pass the
           eligibility + not-already-connected filters. Compose each
           bridge representation up front. The bridges are independent —
           composing one doesn't affect another's filter.
        2. Pass 2 (predict+observe): one batched predict_batch call over
           every bridge (one MPS matmul instead of K). Then observe each
           with its precomputed Prediction.

        Within-batch staleness
        ----------------------
        Predictions are computed against the graph state at the start of
        pass 2. As observe() writes new bridge concepts, those new nodes
        are NOT visible to predictions for later bridges in the same
        cycle. This matches the un-batched semantics: the original loop
        also called predict() before observe() per pair, and a write in
        pair-N didn't retroactively re-score pair-N+1 either. Each pair's
        prediction was always against the state at the moment of its own
        predict() call. We collapse all those predict() calls to a single
        batched one, which is equivalent up to the per-item ordering of
        the predict step within the cycle.
        """
        eligible = self._eligible_endpoints()
        if len(eligible) < 2:
            return 0, 0

        weights = self._weights_for(eligible)

        # ---- pass 1: sample pairs and compose bridges ----
        attempts = 0
        # Allow up to 3 × K_PAIRS draws so we don't loop forever when
        # the graph is dense (most pairs already connected) but also
        # don't sample for an unbounded time.
        max_attempts = K_PAIRS * 3
        seen_pairs: set[tuple[int, int]] = set()
        pair_data: list[tuple[int, int, np.ndarray]] = []

        while len(pair_data) < K_PAIRS and attempts < max_attempts:
            attempts += 1
            cid_a, cid_b = self._pick_pair(eligible, weights)
            if cid_a == cid_b:
                continue
            pair_key = (min(cid_a, cid_b), max(cid_a, cid_b))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            if self._already_connected(cid_a, cid_b):
                continue

            bridge = self._compose_bridge(cid_a, cid_b)
            if bridge is None:
                continue

            pair_data.append((cid_a, cid_b, bridge))

        if not pair_data:
            return 0, 0

        # ---- pass 2: batched predict + per-item observe ----
        bridges = np.stack([b for _, _, b in pair_data]).astype(
            np.float32, copy=False
        )
        predictions = self.predict_engine.predict_batch(bridges, layer="INPUT")

        events = 0
        new_writes = 0
        for (cid_a, cid_b, bridge), prediction in zip(pair_data, predictions):
            # The bridge IS the actual we observe — the prediction is
            # the graph's current best guess of what should be there
            # (top-1 nearest in rep space). If those agree, no surprise;
            # if they differ enough, observe() runs the standard
            # write/strengthen path including auto-link.
            name_hint = f"bridge:{cid_a}-{cid_b}"
            gap = self.predict_engine.observe(
                prediction=prediction,
                actual=bridge,
                now=now,
                name_hint=name_hint,
                replay_origin=False,
            )

            events += 1
            if gap.was_new_write and gap.concept_id is not None:
                new_writes += 1
                # Mirror MainLoop.cycle's post-write auto-link: every
                # fresh concept lays SIMILAR_TO edges to its top-K
                # cosine-nearest existing neighbors. observe() writes
                # the node but does NOT link — auto-linking is the
                # caller's responsibility (cycle does it; we do too).
                # Without this, bridge concepts would be structural
                # islands and the active-inference output would be
                # invisible to F.spread.
                self.graph.link_to_nearest_neighbors(
                    gap.concept_id,
                    now=now,
                )

        return events, new_writes

    # ---- helpers ----

    def _eligible_endpoints(self) -> list[int]:
        """Concept ids with activation_count above the floor. Order is
        the dict iteration order — caller treats it as an unordered
        sample population."""
        return [
            cid
            for cid, node in self.graph.nodes.items()
            if node.activation_count >= MIN_ACTIVATION_COUNT
        ]

    def _weights_for(self, ids: list[int]) -> np.ndarray:
        """Probability weights over `ids`. Activation-count proportional
        with a uniform fallback when the column sums to zero (every node
        has count == 0 — possible right after a fresh-write before any
        strengthen). np.random.choice requires non-zero sum and values
        summing to 1, so we guard both conditions."""
        raw = np.array(
            [float(self.graph.nodes[cid].activation_count) for cid in ids],
            dtype=np.float64,
        )
        s = float(raw.sum())
        if s <= 0.0 or not np.isfinite(s):
            n = len(ids)
            return np.full(n, 1.0 / n, dtype=np.float64)
        return raw / s

    def _pick_pair(
        self, ids: list[int], weights: np.ndarray,
    ) -> tuple[int, int]:
        """Sample two concept ids without replacement under the given
        weights. Two single draws (with the second from a renormalized
        leftover) is exactly the np.random.choice(replace=False) path
        but tolerates a length-1 list via the caller's len-check.
        """
        idx_a = int(self._rng.choice(len(ids), p=weights))
        # Zero out the chosen index and renormalize. If everything else
        # was zero (only one nonzero weight), fall back to uniform
        # over the remaining ids.
        leftover = weights.copy()
        leftover[idx_a] = 0.0
        s = float(leftover.sum())
        if s <= 0.0:
            n = len(ids)
            if n <= 1:
                return ids[idx_a], ids[idx_a]
            uniform = np.full(n, 1.0 / (n - 1), dtype=np.float64)
            uniform[idx_a] = 0.0
            idx_b = int(self._rng.choice(n, p=uniform))
        else:
            idx_b = int(self._rng.choice(len(ids), p=leftover / s))
        return ids[idx_a], ids[idx_b]

    def _already_connected(self, cid_a: int, cid_b: int) -> bool:
        """True if any edge of any type exists between a and b in
        either direction. The graph stores edges keyed by
        (src, tgt, EdgeType) — we collapse to unordered pairs and
        sweep all 10 EdgeTypes (no need to enumerate; the dict scan
        is O(E) and we're only doing this K_PAIRS times per cycle).
        """
        # Cheaper than scanning all of self.graph._edges: walk this
        # node's out + in edge lists via get_edges, which uses the
        # adjacency caches.
        out_a = self.graph.get_edges(cid_a, type=None, direction="both")
        for e in out_a:
            if e.source_id == cid_b or e.target_id == cid_b:
                return True
        return False

    def _compose_bridge(
        self, cid_a: int, cid_b: int,
    ) -> np.ndarray | None:
        """L2-normalized midpoint of the two embeddings. Returns None
        if either node is missing (defensive — should not happen since
        ids came from self.graph.nodes) or the midpoint has zero norm
        (the two embeddings were exactly antipodal — extremely unlikely
        on real data, possible on synthetic fixtures)."""
        node_a = self.graph.nodes.get(cid_a)
        node_b = self.graph.nodes.get(cid_b)
        if node_a is None or node_b is None:
            return None
        mid = (node_a.embedding + node_b.embedding) * 0.5
        n = float(np.linalg.norm(mid))
        if n < 1e-9:
            return None
        return (mid / n).astype(np.float32)
