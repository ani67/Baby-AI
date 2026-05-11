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
        # This keeps the runtime main loop at ~20 steps/s instead of
        # the 6 steps/s it dropped to when learning ran inline.
        self._pc_learn_queue: list = []
        self._pc_learn_thread: Optional[threading.Thread] = None

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
        log.info("[runtime] started (main + pc-learn threads)")

    def stop(self):
        log.info("[runtime] stopping ...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        if self._pc_learn_thread:
            self._pc_learn_thread.join(timeout=5)
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
                # mutate the shared list in place — single-writer
                # invariant (only _process_input appends; only this
                # thread pops). On py3, list slice/del is GIL-protected.
                del queue[:len(batch)]
                for rep in batch:
                    try:
                        self.pc.process(rep, learn=True)
                    except Exception as exc:
                        log.debug(f"[runtime] pc bg learn failed: {exc}")
            else:
                time.sleep(0.1)

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
            if (self._expression_pending
                    and self._step_count % self.EXPRESSION_CHECK_INTERVAL == 0):
                self._maybe_express(now)

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

        # existing surprise pipeline (keeps concept writes flowing)
        try:
            ingest = self.loop.input_pipeline.ingest_text(
                event.text, now=now, representation=rep,
            )
            self.loop.cycle(ingest, now=now + 1e-3,
                            force_respond=False, skip_simulation=True)
        except Exception as exc:
            log.debug(f"[runtime] ingest+cycle failed: {exc}")

        # theory of mind
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

        # try graph-traversal expression
        try:
            surface = self.expression_graph.generate(max_words=25)
        except Exception as exc:
            log.debug(f"[runtime] graph expression failed: {exc}")
            surface = ""

        if not surface.strip():
            try:
                surface = self._fallback_expression(now)
            except Exception:
                surface = ""

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
