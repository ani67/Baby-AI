"""
MindRuntime: the continuously running mind.

Replaces the sequential pipeline assumption. Nothing waits. Everything
runs simultaneously. The mind is always:
  - sensing (if input available)
  - thinking (wave field always running)
  - feeling (affect always updating)
  - self-modeling (self model always updating)
  - inferring (active inference during low arousal)
  - ready to express (when field settles)

Not a pipeline. A living process.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


log = logging.getLogger('mind_runtime')


@dataclass
class InputEvent:
    text: str
    person_id: str = 'default'
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class OutputEvent:
    text: str
    gap: float
    active_concept_count: int
    arousal: float
    timestamp: float = 0.0
    generator: str = 'unknown'

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class MindRuntime:
    """Continuously running mind. dt=0.05s background loop."""

    DT = 0.05
    EXPRESSION_CHECK_INTERVAL = 10
    ACTIVE_INFERENCE_AROUSAL = 0.4
    SLEEP_AROUSAL = 0.2
    SLEEP_TIME_THRESHOLD = 30.0

    def __init__(self, *, loop, wave_field, predictive_coding, self_model,
                 fusion, contradiction_detector, expression_graph,
                 affect, graph, persistence, paths):
        self.loop = loop
        self.wave_field = wave_field
        self.pc = predictive_coding
        self.self_model = self_model
        self.fusion = fusion
        self.contradiction = contradiction_detector
        self.expression_graph = expression_graph
        self.affect = affect
        self.graph = graph
        self.persistence = persistence
        self.paths = paths

        self._input_queue: queue.Queue = queue.Queue()
        self._output_queue: queue.Queue = queue.Queue()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._step_count = 0
        self._last_input_t = time.time()
        self._last_save_t = time.time()
        self._last_expression_t = time.time()
        self._expression_pending = False

        # PC learning off the hot path. _process_input does forward-only
        # (~1 ms) and queues the fused rep here; the background
        # _pc_learn_loop thread pulls in batches of 16 and runs
        # forward+backward+Adam without blocking wave-field stepping.
        self._pc_learn_queue: list = []
        self._pc_learn_thread: Optional[threading.Thread] = None

        # v0.9 ingest+cycle off the hot path. The full v0.9 pipeline
        # (predict.observe + find_or_match + 4-tick attention spread on
        # the 60K-node graph) costs 200-500ms per input — was the last
        # bottleneck capping main-loop step rate to ~6/s. _process_input
        # now does only encode + wave-inject + affect-gate (~5ms total)
        # and queues the (rep, text, now) tuple here; _cycle_loop pulls
        # one at a time and runs the full v0.9 cycle without blocking
        # wave-field stepping.
        self._cycle_queue: list = []
        self._cycle_thread: Optional[threading.Thread] = None

        # Expression off the hot path. graph_expression.generate walks
        # 27K word-concept nodes × max_words × 5 wave-steps-per-word —
        # ~5 seconds per call on this graph. Running on the main loop
        # froze it the moment _expression_pending fired. The express
        # thread polls _expression_pending, runs generate, pushes the
        # surface onto _output_queue.
        self._express_thread: Optional[threading.Thread] = None

        self.total_inputs = 0
        self.total_outputs = 0
        self.total_steps = 0

    @classmethod
    def load(cls, mind_name: str = 'first') -> 'MindRuntime':
        from backend.mind_paths import MindPaths
        from backend.persistence import MindPersistence
        from backend.wave_field import WaveField
        from backend.predictive_coding import PredictiveCodingHierarchy
        from backend.self_model import SelfModel
        from backend.fusion import MultimodalFusion
        from backend.contradiction import ContradictionDetector
        from backend.expression_graph import GraphTraversalExpression

        paths = MindPaths(mind_name)
        log.info(f"[runtime] loading {paths.db}")
        loop = MindPersistence.load(paths.db)
        graph = loop.graph
        graph._rebuild_index()
        affect = loop.affect

        wave_field = WaveField(graph)
        pc = PredictiveCodingHierarchy()
        self_model = SelfModel()
        fusion = MultimodalFusion()
        contradiction = ContradictionDetector(graph, wave_field)
        graph._contradiction_detector = contradiction  # wire the hook
        expression_graph = GraphTraversalExpression(graph, wave_field)
        persistence = MindPersistence(paths.db)

        log.info(f"[runtime] initialized: {graph.node_count} nodes")

        return cls(
            loop=loop, wave_field=wave_field, predictive_coding=pc,
            self_model=self_model, fusion=fusion,
            contradiction_detector=contradiction,
            expression_graph=expression_graph,
            affect=affect, graph=graph, persistence=persistence,
            paths=paths,
        )

    # ---- public ----

    def send(self, text: str, person_id: str = 'default'):
        self._input_queue.put(InputEvent(text=text, person_id=person_id))

    def receive(self, timeout: float = 5.0) -> Optional[OutputEvent]:
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._main_loop, daemon=True, name='MindRuntime',
        )
        self._thread.start()
        self._pc_learn_thread = threading.Thread(
            target=self._pc_learn_loop, daemon=True, name='MindRuntime-PC',
        )
        self._pc_learn_thread.start()
        self._cycle_thread = threading.Thread(
            target=self._cycle_loop, daemon=True, name='MindRuntime-Cycle',
        )
        self._cycle_thread.start()
        self._express_thread = threading.Thread(
            target=self._express_loop, daemon=True, name='MindRuntime-Express',
        )
        self._express_thread.start()
        log.info("[runtime] started (main + pc-learn + cycle + express threads)")

    def stop(self):
        log.info("[runtime] stopping ...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        if self._pc_learn_thread:
            self._pc_learn_thread.join(timeout=5)
        if self._cycle_thread:
            self._cycle_thread.join(timeout=10)
        if self._express_thread:
            self._express_thread.join(timeout=10)
        self._save()
        log.info("[runtime] stopped")

    def _pc_learn_loop(self):
        """Background thread: pull fused reps off the learn queue in
        batches of 16 and run pc.process(learn=True). Decouples the
        28M-param forward+backward+Adam from the wave-field main loop."""
        while self._running:
            queue = self._pc_learn_queue
            if queue:
                batch = queue[:16]
                del queue[:len(batch)]
                for rep in batch:
                    try:
                        self.pc.process(rep, learn=True)
                    except Exception as exc:
                        log.debug(f"[runtime] pc bg learn failed: {exc}")
            else:
                time.sleep(0.1)

    def _express_loop(self):
        """Background thread: when _expression_pending is set, run
        graph_expression.generate (read-only walk over the live wave
        field) and push the surface to _output_queue via the
        _maybe_express path. Polls every 0.2 s."""
        while self._running:
            if not self._expression_pending:
                time.sleep(0.2)
                continue
            now = time.time()
            try:
                self._maybe_express(now)
            except Exception as exc:
                log.debug(f"[runtime] express loop err: {exc}")
                self._expression_pending = False
            time.sleep(0.2)

    def _cycle_loop(self):
        """Background thread: pull (rep, text, now) off _cycle_queue
        and run the full v0.9 ingest+cycle (predict.observe →
        find_or_match → 4-tick attention spread on the 60K-node
        graph → possible write_on_surprise). Costs 200-500 ms per
        input — too expensive for the wave-field main loop. Concept
        writing, surprise detection, and contradiction checking all
        happen here without blocking the runtime's stepping cadence."""
        while self._running:
            queue = self._cycle_queue
            if not queue:
                time.sleep(0.01)
                continue
            rep, text, t = queue.pop(0)
            try:
                prediction = self.loop.predict_engine.predict(
                    rep, layer='INPUT',
                )
                gap = self.loop.predict_engine.observe(
                    prediction, rep, t, name_hint=text[:64],
                )
                # If a new concept was written by the v0.9 surprise path,
                # add it to the wave field so it can participate in
                # propagation immediately, and let contradiction
                # detection scan its neighborhood.
                if (getattr(gap, 'is_surprise', False)
                        and getattr(gap, 'was_new_write', False)
                        and gap.concept_id is not None):
                    cid = int(gap.concept_id)
                    if cid in self.graph.nodes:
                        node = self.graph.nodes[cid]
                        # NOTE: wave_field.add_node extends per-node
                        # tensors but does NOT rebuild adjacency.
                        # rebuild_if_dirty over 55K nodes / 287K edges
                        # costs 1-2 s and runs in this background
                        # thread while main thread reads the same
                        # tensors — froze the runtime entirely on
                        # surprise. New concepts are written to the
                        # graph immediately and will fully integrate
                        # into the wave field on next runtime
                        # restart (when adjacency rebuilds from disk).
                        try:
                            self.wave_field.add_node(cid, node.embedding)
                        except Exception:
                            pass
                        try:
                            self.contradiction.check_new_concept(cid, t)
                        except Exception:
                            pass

                if getattr(gap, 'is_surprise', False):
                    try:
                        self.affect.inject(
                            'INPUT',
                            getattr(gap, 'gap_signal', None) or np.zeros(12, dtype=np.float32),
                            float(getattr(gap, 'magnitude', 0.0)),
                            t,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                log.debug(f"[runtime] cycle bg failed: {exc}")

    def status(self) -> dict:
        now = time.time()
        return {
            'running': self._running,
            'step_count': self._step_count,
            'total_inputs': self.total_inputs,
            'total_outputs': self.total_outputs,
            'wave_energy': self.wave_field.energy,
            'peak_activation': self.wave_field.peak_activation,
            'arousal': self.affect.current_arousal(now),
            'node_count': self.graph.node_count,
            'active_concepts': len(self.wave_field.get_top_concepts(10)),
            'contradiction_buffer': len(
                self.contradiction.get_active_contradictions()),
            'self_prediction_loss': self.self_model.self_prediction_loss,
        }

    # ---- main loop ----

    def _main_loop(self):
        while self._running:
            loop_start = time.time()
            now = loop_start

            # === SENSE ===
            new_input = None
            try:
                new_input = self._input_queue.get_nowait()
            except queue.Empty:
                pass
            if new_input is not None:
                try:
                    self._process_input(new_input, now)
                    self._expression_pending = True
                except Exception as exc:
                    log.warning(f"[runtime] input failed: {exc}")

            # === THINK ===
            try:
                self.wave_field.step()
            except Exception as exc:
                log.warning(f"[runtime] wave step failed: {exc}")

            # PC top-down disabled until PC is trained. An untrained PC
            # emits a near-uniform softmax over 60K concepts (each
            # entry ~1.67e-5); wave_step computes
            #   top_down_strength * (top_down - activation)
            # which, with activation ~1.0 and top_down ~1e-5, evaluates
            # to roughly -0.3 * activation — i.e. it pulls every
            # active node back toward zero. The wave can't build a
            # peak above 0.01 while this is active. Re-enable once
            # PC has been trained for enough steps that its softmax
            # actually concentrates around relevant concepts. Until
            # then it's pure damping.
            #
            # if self._step_count % 10 == 0:
            #     try:
            #         td = self.pc.get_top_down_for_wave_field(self.wave_field)
            #         if td is not None:
            #             self.wave_field.set_top_down(td)
            #     except Exception:
            #         pass

            # === FEEL ===
            if self._step_count % 5 == 0:
                try:
                    affect_composite = self.affect.composite(now)
                    # update affect gate; wave_field expects D_REP-space affect
                    # (we have N_AFF) — best-effort, gate may skip if dims mismatch
                    self.wave_field.update_affect_gate(affect_composite)
                except Exception:
                    pass

            # === SELF-MODEL ===
            if self._step_count % 20 == 0:
                try:
                    affect_composite = self.affect.composite(now)
                    centroid = self.wave_field.get_field_centroid()
                    self.self_model.update(affect_composite, centroid, now)
                except Exception:
                    pass

            # === ACTIVE INFERENCE ===
            try:
                arousal = self.affect.current_arousal(now)
            except Exception:
                arousal = 0.5
            time_since_input = now - self._last_input_t
            if (arousal < self.ACTIVE_INFERENCE_AROUSAL
                    and time_since_input > 5.0
                    and self._step_count % 100 == 0):
                self._run_active_inference(now)

            if self._step_count % 200 == 0:
                unresolved = self.contradiction.get_active_contradictions()
                if unresolved:
                    c = unresolved[0]
                    try:
                        self.wave_field.inject(
                            [c.concept_a, c.concept_b], strength=0.3,
                        )
                    except Exception:
                        pass

            # === EXPRESSION ===
            # Expression generation (graph_traversal walks 27K word nodes
            # x max_words x 5 wave-steps-per-word — ~5 seconds per call)
            # runs in the _express_thread, not on the main loop.
            # _expression_pending is the cross-thread signal.
            pass

            # === SLEEP ===
            if (arousal < self.SLEEP_AROUSAL
                    and time_since_input > self.SLEEP_TIME_THRESHOLD
                    and self._step_count % 1000 == 0):
                self._sleep(now)

            # === SAVE ===
            if now - self._last_save_t > 300:
                self._save()
                self._last_save_t = now

            self._step_count += 1
            self.total_steps += 1

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, self.DT - elapsed)
            time.sleep(sleep_time)

    # ---- handlers ----

    def _process_input(self, event: InputEvent, now: float):
        self._last_input_t = now
        self.total_inputs += 1

        is_echo = self.self_model.is_self_echo(event.text, now)

        # encode
        try:
            from backend.encoders import encode_text_glove
            from backend import config as cfg
            rep = encode_text_glove(event.text, dim=cfg.D_REP)
        except Exception as exc:
            log.warning(f"[runtime] encode failed: {exc}")
            return
        if rep is None:
            return

        fused = self.fusion.fuse(rep.astype(np.float32))

        # update PC — forward only on hot path. Queue for background
        # learning so the 28M-param backward + Adam step doesn't block
        # the wave-field main loop.
        try:
            self.pc.process(fused, learn=False)
        except Exception as exc:
            log.debug(f"[runtime] pc forward failed: {exc}")
        try:
            self._pc_learn_queue.append(fused.copy())
        except Exception:
            pass

        # inject into wave field
        inject_strength = 1.0
        if is_echo:
            resonance = self.self_model.self_echo_resonance(
                event.text, np.zeros(rep.shape[0]).astype(np.float32),
                self.affect.composite(now), now,
            )
            inject_strength = 0.3 * resonance

        try:
            # Inject the RAW encoded text rep, not the fusion output.
            # MultimodalFusion is randomly initialised (no training yet);
            # passing rep through it produces a near-random projection
            # with ~0 cosine to every concept in the graph, which the
            # min_sim threshold then rejects entirely. Until fusion is
            # trained, the raw GloVe-PCA-projected rep is the
            # representation that lives in the same space as the
            # concept embeddings.
            self.wave_field.inject_representation(rep.astype(np.float32),
                                                  strength=inject_strength)
        except Exception:
            pass

        # v0.9 ingest+cycle (predict.observe, find_or_match, attention
        # spread, possible write_on_surprise) moved off the hot path —
        # see _cycle_loop. It cost 200-500ms per input and was the last
        # bottleneck capping the wave-field main loop to ~6 steps/s.
        # Background thread consumes _cycle_queue and runs it
        # asynchronously; concept writes still happen, contradictions
        # still get checked, surprise still updates affect — just
        # asynchronously from expression.
        try:
            self._cycle_queue.append((rep.astype(np.float32), event.text, now))
        except Exception:
            pass

        # theory of mind (cheap — small per-person dict update)
        try:
            self.self_model.update_other(
                event.person_id, fused, 0.5, now - self._last_input_t,
            )
        except Exception:
            pass

    def _maybe_express(self, now: float):
        top_concepts = self.wave_field.get_top_concepts(50)
        if not top_concepts:
            return

        # Read-only walk over the live wave field. mutate_field=False
        # because the main thread is stepping the wave concurrently;
        # injecting+stepping from this thread would race on shared
        # torch tensors.
        try:
            surface = self.expression_graph.generate(
                max_words=25, mutate_field=False,
            )
        except Exception as exc:
            log.debug(f"[runtime] graph expression failed: {exc}")
            surface = ""

        # Fallback expression disabled in the runtime hot path —
        # loop.expression.generate runs the native_head transformer
        # (28M params, CPU device) and takes seconds, freezing the
        # main loop. Graph traversal is the runtime's voice; if the
        # wave hasn't settled enough to pick word-concepts yet, just
        # return without emitting and the runtime will try again on
        # the next expression check.
        if not surface.strip():
            return

        # gap vs field centroid
        try:
            from backend.encoders import encode_text_glove
            from backend import config as cfg
            rep = encode_text_glove(surface, dim=cfg.D_REP).astype(np.float32)
            centroid = self.wave_field.get_field_centroid()
            n_r = np.linalg.norm(rep) + 1e-9
            n_c = np.linalg.norm(centroid) + 1e-9
            cos = float(np.dot(rep / n_r, centroid / n_c))
            gap = float(1.0 - cos)
        except Exception:
            rep = None
            gap = 0.0

        if gap > 0.91:
            return  # suppress as too far

        try:
            arousal = self.affect.current_arousal(now)
        except Exception:
            arousal = 0.5

        output = OutputEvent(
            text=surface, gap=gap,
            active_concept_count=len(top_concepts),
            arousal=arousal, generator='graph_traversal',
        )
        self._output_queue.put(output)
        self.total_outputs += 1
        self._last_expression_t = now
        self._expression_pending = False

        # self-echo registration
        try:
            if rep is not None:
                self.self_model.register_output(surface, rep, now)
        except Exception:
            pass

        # self-overhearing: feed own output back as input
        self._input_queue.put(InputEvent(
            text=surface, person_id='self', timestamp=now + 0.1,
        ))

        log.info(f"[runtime] expressed: '{surface}' "
                 f"gap={gap:.3f} active={len(top_concepts)}")

    def _fallback_expression(self, now: float) -> str:
        """Fall back to existing expression pipeline."""
        try:
            # try the existing Expression.generate signature; this
            # codebase may have a different one
            from backend.identity import ChosenCandidate
            # use a minimal active_concepts dict from wave field
            actives = {
                cid: act for cid, act
                in self.wave_field.get_top_concepts(10)
            }
            res = self.loop.expression.generate(
                active_concepts=actives,
                affect=self.affect.composite(now),
                now=now,
            )
            if isinstance(res, ChosenCandidate):
                return res.surface_text or ""
            if res and hasattr(res, 'surface_text'):
                return res.surface_text
        except Exception as exc:
            log.debug(f"[runtime] fallback expr failed: {exc}")
        return ""

    def _run_active_inference(self, now: float):
        try:
            self.loop.active_inference.run_inference_cycle(now)
        except Exception as exc:
            log.debug(f"[runtime] active inference failed: {exc}")

    def _sleep(self, now: float):
        log.info("[runtime] sleep cycle starting ...")
        try:
            self.loop._form_abstractions(now)
        except Exception as exc:
            log.debug(f"[runtime] sleep failed: {exc}")
        log.info(f"[runtime] sleep done. nodes={self.graph.node_count}")

    def _save(self):
        try:
            self.persistence.save(self.loop, time.time())
            log.debug("[runtime] checkpoint saved")
        except Exception as exc:
            log.warning(f"[runtime] save failed: {exc}")
