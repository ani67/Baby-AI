"""
LearningLoop — the autonomous learning cycle orchestrator.
"""

import asyncio
import enum
import logging
import re
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# Category nouns for tracking per-category performance (COCO labels)
_CATEGORY_NOUNS = re.compile(
    r'\b(dog|cat|bird|car|bus|train|person|man|woman|horse|elephant|giraffe|'
    r'zebra|bear|cow|sheep|truck|boat|bicycle|motorcycle|airplane|skateboard|'
    r'surfboard|snowboard|tennis|baseball|pizza|cake|sandwich|broccoli|banana|'
    r'apple|orange|chair|couch|bed|table|toilet|laptop|phone|clock|vase|book|'
    r'umbrella|knife|fork|bottle|cup|bowl)\b', re.IGNORECASE
)

from .curiosity import CuriosityScorer
from .health_monitor import HealthMonitor
from .question_gen import QuestionGenerator
from .sentence_splitter import split_sentences


logger = logging.getLogger(__name__)


STOPWORDS = {
    "about", "their", "there", "these", "those",
    "would", "could", "should", "which", "where", "after",
    "being", "other", "every", "still", "while", "since",
    "until", "before", "between", "through", "during",
}


class LoopState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"
    ERROR = "error"


@dataclass
class LoopStatus:
    state: str
    step: int
    stage: int
    delay_ms: int
    error_message: str | None
    graph_summary: dict
    teacher_healthy: bool


@dataclass
class StepResult:
    step: int = 0
    question: str = ""
    answer: str = ""
    curiosity_score: float = 0.0
    is_positive: bool = True
    delta_summary: dict = field(default_factory=dict)
    growth_events: list = field(default_factory=list)
    duration_ms: int = 0
    skipped: bool = False
    reason: str = ""


class LearningLoop:
    def __init__(
        self,
        model,
        teacher,
        encoder: tuple,
        decoder,
        store,
        viz_emitter,
        curriculum,
    ):
        self.model = model
        self.teacher = teacher
        self.encoders = encoder
        self.decoder = decoder
        self.store = store
        self.viz_emitter = viz_emitter
        self.curriculum = curriculum

        self.curiosity = CuriosityScorer()
        self.question_gen = QuestionGenerator()
        self.health_monitor = HealthMonitor()

        self._state = LoopState.IDLE
        self._stage = 0
        self._delay_ms = 0
        self._error_message: str | None = None
        self._recent_questions: deque = deque(maxlen=50)
        self._known_words: set = set()
        self._similarity_history: deque = deque(maxlen=25)
        self._positive_history: deque = deque(maxlen=50)
        self._stage0_stable_steps: int = 0
        self._stage0_last_active: int = 0
        self._stage0_completion_step: int | None = None
        self._stage1_completion_step: int | None = None
        self._loop_task: asyncio.Task | None = None
        self._cofiring_buffer: list[tuple[str, str]] = []
        self._cofiring_steps_since_flush: int = 0
        self._prev_active_cids: list[str] = []  # cross-step temporal co-firing
        self._batch_count: int = 0
        self._batch_total_ms: float = 0.0

        # Episodic memory
        from .memory import EpisodicMemory
        self.memory = EpisodicMemory(store, capacity=2000)

    # ── Control ──

    async def start(self) -> None:
        if self._state not in (LoopState.IDLE, LoopState.PAUSED):
            return
        self._state = LoopState.RUNNING

        # Warm up the teacher model
        try:
            await self.teacher.ask("Hello", stage=0)
        except Exception:
            pass

        while self._state == LoopState.RUNNING:
            try:
                await self.step_once()
            except Exception as e:
                if "unavailable" in type(e).__name__.lower():
                    self._state = LoopState.ERROR
                    self._error_message = str(e)
                    break
                logger.error(f"Step error: {e}", exc_info=True)

            if self._delay_ms > 0:
                await asyncio.sleep(self._delay_ms / 1000)
            else:
                await asyncio.sleep(0)

    async def pause(self) -> None:
        self._state = LoopState.PAUSED

    async def resume(self) -> None:
        if self._state == LoopState.PAUSED:
            self._state = LoopState.RUNNING
            self._loop_task = asyncio.create_task(self._run_loop())

    async def _run_loop(self) -> None:
        while self._state == LoopState.RUNNING:
            try:
                await self.step_once()
            except Exception as e:
                if "unavailable" in type(e).__name__.lower():
                    self._state = LoopState.ERROR
                    self._error_message = str(e)
                    break
                logger.error(f"Step error: {e}", exc_info=True)

            if self._delay_ms > 0:
                await asyncio.sleep(self._delay_ms / 1000)
            else:
                await asyncio.sleep(0)

    async def step_once(self) -> StepResult:
        """Executes one full learning cycle. Delegates to step_batch for precomputed."""
        # Route based on the curriculum instance's actual source, not the module constant.
        # The instance may have fallen back to "live" if the cache was empty at startup.
        is_precomputed = getattr(self.curriculum, '_source', 'live') == 'precomputed'
        has_batch = hasattr(self.curriculum, 'next_batch')
        use_batch = is_precomputed and has_batch

        if not hasattr(self, '_logged_routing'):
            self._logged_routing = True
            print(
                f"[batch] routing: precomputed={is_precomputed} "
                f"has_next_batch={has_batch} batch_path={use_batch}",
                flush=True,
            )

        if use_batch:
            return await self._step_batch(128)
        return await self._step_single()

    async def _step_single(self) -> StepResult:
        """Single-sample learning step (original path)."""

        # ── 1. OBSERVE ──
        graph_summary = self.model.graph.summary()
        active_clusters = [
            c for c in self.model.graph.clusters if not c.dormant
        ]

        # ── 2. SCORE ──
        curriculum_item = self.curriculum.next_item(
            stage=self._stage,
            model_state=graph_summary,
        )
        curiosity_score = self.curiosity.score(
            item=curriculum_item,
            model=self.model,
        )

        # ── 3. SELECT (implicit in step 2) ──

        # ── 4. QUESTION ──
        question = self.question_gen.generate(
            item=curriculum_item,
            stage=self._stage,
            recent_questions=list(self._recent_questions),
        )
        self._recent_questions.append(question)

        # ── 5. ASK + 6. ENCODE ──
        # Precomputed path: skip Ollama entirely, use cached embeddings
        if getattr(curriculum_item, "precomputed", False) and curriculum_item.expected_vector is not None:
            answer_vectors = [curriculum_item.expected_vector]
            teacher_answer = curriculum_item.description or ""
        else:
            # Live path: call Ollama teacher
            image_bytes = None
            if curriculum_item.item_type == "image" and curriculum_item.image_path:
                try:
                    with open(curriculum_item.image_path, "rb") as f:
                        image_bytes = f.read()
                except FileNotFoundError:
                    pass

            # Strip [IMAGE: ...] from question when sending actual image
            ask_question = question
            if image_bytes is not None:
                import re
                ask_question = re.sub(r'\s*\[IMAGE:\s*[^\]]*\]', '', question).strip()
                if not ask_question:
                    ask_question = f"What is this?"

            try:
                teacher_response = await self.teacher.ask(
                    question=ask_question,
                    stage=self._stage,
                    context=curriculum_item.context,
                    image_bytes=image_bytes,
                )
            except Exception as e:
                ename = type(e).__name__
                if "unavailable" in ename.lower():
                    self._state = LoopState.ERROR
                    self._error_message = str(e)
                    raise
                if "timeout" in ename.lower():
                    return StepResult(skipped=True, reason="teacher_timeout")
                raise

            # Teacher returned None — repetition detected, skip this step
            if teacher_response is None:
                return StepResult(skipped=True, reason="teacher_repetition")

            answer_vectors = self._encode_answer(
                teacher_response.answer,
                curriculum_item,
            )
            teacher_answer = teacher_response.answer

        # ── 7. PREDICT ──
        input_vec = answer_vectors[0] if answer_vectors else torch.randn(self.model.input_dim)
        prediction, activations = self.model.forward(
            input_vec, return_activations=True,
        )

        # Adaptive threshold from step 0 — no stage gating
        is_positive = self._compute_is_positive(
            prediction=prediction,
            answer_vectors=answer_vectors,
        )

        # ── 8. UPDATE ──
        changes = {}
        if is_positive:
            update_vectors = answer_vectors
        else:
            # Negative: feed a random vector so model learns contrast
            update_vectors = [F.normalize(torch.randn(self.model.input_dim), dim=0)]
        for vec in update_vectors:
            step_changes = self.model.update(
                x=vec,
                is_positive=is_positive,
            )
            for k, v in step_changes.items():
                changes[k] = changes.get(k, 0.0) + v

        # ── 9. MEASURE ──
        delta_summary = {
            "weight_change_magnitude": sum(changes.values()),
            "edges_formed": [],
            "edges_pruned": [],
            "clusters_budded": [],
            "layers_inserted": [],
            "is_positive": is_positive,
            "curiosity_score": curiosity_score,
        }

        # ── 10. GROW ──
        growth_events = self.model.growth_check(self.store)
        for event in growth_events:
            etype = event.get("event_type", "")
            if etype == "CONNECT":
                delta_summary["edges_formed"].append(
                    f"{event.get('cluster_a', '')}->{event.get('cluster_b', '')}"
                )
            elif etype == "PRUNE":
                delta_summary["edges_pruned"].append(
                    f"{event.get('cluster_a', '')}->{event.get('cluster_b', '')}"
                )
            elif etype == "BUD":
                delta_summary["clusters_budded"].append(
                    event.get("cluster_a", "")
                )
            elif etype == "INSERT":
                delta_summary["layers_inserted"].append(
                    event.get("metadata", {}).get("new_cluster", "")
                )

        # ── 10b. HEALTH MONITOR ──
        active_count = len(activations)  # clusters that actually fired this step
        total_clusters = len(self.model.graph.clusters)
        self.health_monitor.record_step(active_count, total_clusters)
        self.health_monitor.check(
            step=self.model.step,
            stage=self._stage,
            model=self.model,
            positive_history=self._positive_history,
            similarity_history=self._similarity_history,
        )

        # ── 10c. CO-FIRING (z-score filtered) ──
        # Only co-fire clusters that fired UNUSUALLY strongly for this input.
        # v2 fires 100+ clusters; threshold creates one blob. Z-score captures specialization.
        if activations:
            scores = list(activations.values())
            mean_s = sum(scores) / len(scores)
            std_s = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5
            z_threshold = mean_s + 1.0 * std_s  # 1 sigma above mean
            significant = [cid for cid, v in activations.items() if v > z_threshold]
            for i in range(len(significant)):
                for j in range(i + 1, len(significant)):
                    self._cofiring_buffer.append((significant[i], significant[j]))
            # Temporal co-firing: z-score filtered from previous step
            if self._prev_active_cids:
                for prev_cid in self._prev_active_cids:
                    for curr_cid in significant[:5]:
                        if prev_cid != curr_cid:
                            self._cofiring_buffer.append((prev_cid, curr_cid))
            self._prev_active_cids = significant[:5]
        self._cofiring_steps_since_flush += 1
        if (self._cofiring_steps_since_flush >= 50 or len(self._cofiring_buffer) >= 50000) and self._cofiring_buffer:
            print(f"[cofiring] flushed {len(self._cofiring_buffer)} pairs at step {self.model.step}", flush=True)
            self.store.batch_update_cofiring(self._cofiring_buffer, self.model.step)
            self._cofiring_buffer = []
            self._cofiring_steps_since_flush = 0

        # ── 11. LOG (z-score filtered clusters) ──
        if activations:
            a_vals = list(activations.values())
            a_mean = sum(a_vals) / len(a_vals)
            a_std = (sum((v - a_mean) ** 2 for v in a_vals) / max(len(a_vals), 1)) ** 0.5
            sig_active = [c for c, v in activations.items() if v > a_mean + a_std]
        else:
            sig_active = []
        self.store.log_dialogue(
            step=self.model.step,
            stage=self._stage,
            question=question,
            answer=teacher_answer,
            curiosity_score=curiosity_score,
            clusters_active=sig_active,
            delta_summary=delta_summary,
        )

        # Periodic snapshot
        if self.model.step % self.model.snapshot_interval == 0:
            graph_json = self.model.graph.to_json()
            self.store.log_latent_snapshot(
                step=self.model.step,
                graph_json=graph_json,
            )

        # Periodic checkpoint
        if self.model.step > 0 and self.model.step % 100 == 0:
            state_dict = {}
            if hasattr(self.model, 'brain'):
                state_dict["brain_state"] = self.model.brain.state_dict()
            else:
                for cluster in self.model.graph.clusters:
                    for node in cluster.nodes:
                        state_dict[f"{node.id}.weights"] = node.weights
                        state_dict[f"{node.id}.bias"] = node.bias
            state_dict["_activation_buffer"] = self.model._activation_buffer
            graph_json = self.model.graph.to_json()
            nc = len(graph_json["clusters"])
            ne = len(graph_json["edges"])
            print(f"[checkpoint] saved step={self.model.step} clusters={nc} edges={ne}", flush=True)
            self.store.save_checkpoint(
                step=self.model.step,
                stage=self._stage,
                model_state_dict=state_dict,
                graph_json=graph_json,
            )
            self.store.prune_old_snapshots()

        # ── 12. EMIT ──
        # Train decoder on teacher's CLIP vector (not model prediction!)
        teacher_clip = answer_vectors[0] if answer_vectors else prediction
        self.decoder.train_step(teacher_clip, teacher_answer)
        # Decode model's prediction into words for the frontend
        model_response = self.decoder.decode(prediction, max_words=15, model_step=self.model.step)

        if self.viz_emitter is not None:
            asyncio.ensure_future(self.viz_emitter.emit_step(
                step=self.model.step,
                stage=self._stage,
                graph=self.model.graph,
                activations=activations,
                last_question=question,
                last_answer=teacher_answer,
                model_answer=model_response,
                is_positive=is_positive,
                growth_events=growth_events,
                image_url=(
                    f"/images/{curriculum_item.image_path}" if curriculum_item.image_path
                    else getattr(curriculum_item, "image_url", None)
                ),
            ))

        # Stages collapsed — no auto-advance. LR decays naturally.

        return StepResult(
            step=self.model.step,
            question=question,
            answer=teacher_answer,
            curiosity_score=curiosity_score,
            is_positive=is_positive,
            delta_summary=delta_summary,
            growth_events=growth_events,
            duration_ms=teacher_response.duration_ms if not getattr(curriculum_item, "precomputed", False) else 0,
            skipped=False,
        )

    async def _step_batch(self, batch_size: int = 32) -> StepResult:
        """Batched learning step for precomputed curriculum.
        Runs CPU-heavy computation in a thread pool so FastAPI stays responsive."""
        import time as _time

        graph_summary = self.model.graph.summary()

        # Compute adversarial category weights (inverse of per-category similarity)
        cat_weights = None
        try:
            cats = self.store.get_category_performance()
            if len(cats) >= 10:
                max_sim = max(c["avg_sim"] for c in cats) or 1.0
                cat_weights = {c["category"]: max(0.1, 1.0 - c["avg_sim"] / max_sim) for c in cats}
        except Exception:
            pass

        items = self.curriculum.next_batch(batch_size, stage=self._stage, model_state=graph_summary, category_weights=cat_weights)
        if not items:
            return StepResult(skipped=True, reason="empty_batch")

        # ── Saturation check: if >20% of clusters are near-saturated,
        # reduce batch to 8 to prevent weight explosion ──
        active_clusters = [c for c in self.model.graph.clusters if not c.dormant]
        saturated = sum(1 for c in active_clusters if c.mean_activation > 0.85)
        saturation_ratio = saturated / max(len(active_clusters), 1)
        effective_size = len(items)
        if saturation_ratio > 0.2:
            effective_size = min(8, len(items))
            items = items[:effective_size]
            if self._batch_count % 10 == 0:
                print(
                    f"[batch] saturation cap: {saturated}/{len(active_clusters)} clusters > 0.85, "
                    f"reduced batch to {effective_size}",
                    flush=True,
                )

        # ── Episodic replay: mix into batch ──
        replay_samples = self.memory.sample_replay(n=8, category_weights=cat_weights)

        # ── Run heavy computation in thread pool ──
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._batch_compute, items, replay_samples)

        # Unpack results from the sync computation
        changes, prediction, activations, anchor_pred, elapsed_ms, all_activations = result


        # ── Main-thread work: growth, health, co-firing, logging, viz ──

        # Category tracking — sample items and run per-item forward for real prediction accuracy.
        # Only sample 4 items per batch to avoid 128 forward passes on the main thread.
        import random as _rand
        sample_items = _rand.sample(items, min(4, len(items)))
        for item in sample_items:
            if item.expected_vector is not None:
                # Prefer clean label from curriculum; fall back to regex on caption
                category = item.label
                if not category and item.description:
                    match = _CATEGORY_NOUNS.search(item.description)
                    category = match.group(1).lower() if match else None
                if category:
                    item_pred, _ = self.model.forward(item.expected_vector, return_activations=False)
                    sim = torch.dot(item_pred, F.normalize(item.expected_vector, dim=0)).item()
                    self.store.update_category_performance(category, sim, sim > 0.2, self.model.step)

        # Growth check (must run on main thread — uses SQLite store)
        growth_events = self.model.growth_check(self.store)

        # Episodic memory: store significant experiences
        for item in sample_items:
            if item.expected_vector is not None:
                item_pred, _ = self.model.forward(item.expected_vector, return_activations=False)
                error_mag = 1.0 - torch.dot(item_pred, F.normalize(item.expected_vector, dim=0)).item()
                self.memory.maybe_store(item, error_mag, self.model.step, growth_events)

        # Health monitor
        active_count = len(activations)  # clusters that actually fired this step
        total_clusters = len(self.model.graph.clusters)
        self.health_monitor.record_step(active_count, total_clusters)
        self.health_monitor.check(
            step=self.model.step, stage=self._stage, model=self.model,
            positive_history=self._positive_history,
            similarity_history=self._similarity_history,
        )

        # Co-firing: z-score filtered — only neurons that fired UNUSUALLY for this input.
        # v2 fires 100+ neurons; top-K creates one blob. Z-score captures specialization.
        if activations:
            scores = list(activations.values())
            mean_s = sum(scores) / len(scores) if scores else 0
            std_s = (sum((s - mean_s) ** 2 for s in scores) / max(len(scores), 1)) ** 0.5
            z_threshold = mean_s + 1.0 * std_s  # 1 sigma above mean
            significant = [cid for cid, v in activations.items() if v > z_threshold]
            # Record co-firing only among significant neurons (typically 5-15)
            for i in range(len(significant)):
                for j in range(i + 1, len(significant)):
                    self._cofiring_buffer.append((significant[i], significant[j]))
        # Temporal co-firing: z-score filtered per sample
        if all_activations:
            for t in range(1, len(all_activations)):
                for acts in [all_activations[t - 1], all_activations[t]]:
                    vals = list(acts.values())
                    if not vals:
                        continue
                prev_vals = list(all_activations[t - 1].values())
                curr_vals = list(all_activations[t].values())
                if not prev_vals or not curr_vals:
                    continue
                p_mean = sum(prev_vals) / len(prev_vals)
                p_std = (sum((v - p_mean) ** 2 for v in prev_vals) / len(prev_vals)) ** 0.5
                c_mean = sum(curr_vals) / len(curr_vals)
                c_std = (sum((v - c_mean) ** 2 for v in curr_vals) / len(curr_vals)) ** 0.5
                prev_sig = [c for c, v in all_activations[t - 1].items() if v > p_mean + p_std][:5]
                curr_sig = [c for c, v in all_activations[t].items() if v > c_mean + c_std][:5]
                for p_cid in prev_sig:
                    for c_cid in curr_sig:
                        if p_cid != c_cid:
                            self._cofiring_buffer.append((p_cid, c_cid))
            last_vals = list(all_activations[-1].values())
            if last_vals:
                l_mean = sum(last_vals) / len(last_vals)
                l_std = (sum((v - l_mean) ** 2 for v in last_vals) / len(last_vals)) ** 0.5
                self._prev_active_cids = [c for c, v in all_activations[-1].items() if v > l_mean + l_std][:5]
        self._cofiring_steps_since_flush += 1
        if (self._cofiring_steps_since_flush >= 50 or len(self._cofiring_buffer) >= 50000) and self._cofiring_buffer:
            print(f"[cofiring] flushed {len(self._cofiring_buffer)} pairs at step {self.model.step}", flush=True)
            self.store.batch_update_cofiring(self._cofiring_buffer, self.model.step)
            self._cofiring_buffer = []
            self._cofiring_steps_since_flush = 0

        # Log
        teacher_answer = items[-1].description or ""
        wcm = sum(changes.values())
        if self._batch_count % 10 == 0:
            print(f"[learn] step={self.model.step} weight_change={wcm:.6f} clusters_updated={len(changes)}", flush=True)
        delta_summary = {
            "weight_change_magnitude": wcm,
            "edges_formed": [], "edges_pruned": [],
            "clusters_budded": [], "layers_inserted": [],
            "is_positive": True, "curiosity_score": 0.0,
            "batch_size": len(items),
        }
        # Log z-score significant clusters (not top-K — captures specialization)
        if activations:
            a_vals = list(activations.values())
            a_mean = sum(a_vals) / len(a_vals) if a_vals else 0
            a_std = (sum((v - a_mean) ** 2 for v in a_vals) / max(len(a_vals), 1)) ** 0.5
            sig_active = [c for c, v in activations.items() if v > a_mean + a_std]
        else:
            sig_active = []
        self.store.log_dialogue(
            step=self.model.step, stage=self._stage,
            question=f"[batch {len(items)}]", answer=teacher_answer,
            curiosity_score=0.0,
            clusters_active=sig_active,
            delta_summary=delta_summary,
        )

        # Periodic snapshot/checkpoint
        if self.model.step % self.model.snapshot_interval == 0:
            graph_json = self.model.graph.to_json()
            self.store.log_latent_snapshot(step=self.model.step, graph_json=graph_json)
        if self.model.step > 0 and self.model.step % 100 == 0:
            state_dict = {}
            # V2: save brain state directly if available
            if hasattr(self.model, 'brain'):
                state_dict["brain_state"] = self.model.brain.state_dict()
            else:
                for cluster in self.model.graph.clusters:
                    for node in cluster.nodes:
                        state_dict[f"{node.id}.weights"] = node.weights
                        state_dict[f"{node.id}.bias"] = node.bias
            state_dict["_activation_buffer"] = self.model._activation_buffer
            graph_json = self.model.graph.to_json()
            self.store.save_checkpoint(
                step=self.model.step, stage=self._stage,
                model_state_dict=state_dict, graph_json=graph_json,
            )
            self.store.prune_old_snapshots()

        # Episodic memory eviction + logging
        if self.model.step > 0 and self.model.step % 500 == 0:
            evicted = self.memory.evict()
            mem_count = self.memory.count()
            if self._batch_count % 50 == 0 or evicted:
                print(f"[memory] total={mem_count} evicted={evicted}", flush=True)

        # Train decoder on teacher's CLIP vector (not model prediction!)
        teacher_clip = items[-1].expected_vector if items[-1].expected_vector is not None else prediction
        self.decoder.train_step(teacher_clip, teacher_answer)
        # Emit viz (non-blocking)
        model_response = self.decoder.decode(prediction, max_words=15, model_step=self.model.step)
        if self.viz_emitter is not None:
            asyncio.ensure_future(self.viz_emitter.emit_step(
                step=self.model.step, stage=self._stage,
                graph=self.model.graph, activations=activations,
                last_question=f"[batch {len(items)}]", last_answer=teacher_answer,
                model_answer=model_response, is_positive=True,
                growth_events=growth_events,
                image_url=getattr(items[-1], "image_url", None),
            ))

        # Batch logging
        self._batch_count += 1
        self._batch_total_ms += elapsed_ms
        if self._batch_count % 100 == 0:
            avg = self._batch_total_ms / self._batch_count
            print(
                f"[batch] size={len(items)} avg_ms_per_sample={avg / len(items):.2f} "
                f"batches={self._batch_count} step={self.model.step}",
                flush=True,
            )

        # Stages collapsed — no auto-advance. LR decays naturally.

        return StepResult(
            step=self.model.step,
            question=f"[batch {len(items)}]",
            answer=teacher_answer,
            curiosity_score=0.0,
            is_positive=True,
            delta_summary=delta_summary,
            growth_events=growth_events,
            duration_ms=int(elapsed_ms),
            skipped=False,
        )

    def _batch_compute(self, items, replay_samples=None) -> tuple:
        """Synchronous CPU-heavy batch computation. Runs in thread pool."""
        import time as _time
        t0 = _time.perf_counter()

        # Build (vector, is_positive) pairs.
        # Use ONE forward pass to get model's current output direction,
        # then determine positive/negative for each sample via cosine
        # similarity — avoids 32 forward passes per batch.
        anchor_vec = items[0].expected_vector if items[0].expected_vector is not None else torch.randn(self.model.input_dim)
        anchor_pred, _ = self.model.forward(anchor_vec, return_activations=False)

        samples: list[tuple] = []
        skipped = 0
        for idx, item in enumerate(items):
            vec = item.expected_vector if item.expected_vector is not None else torch.randn(self.model.input_dim)
            teacher_vec = vec.clone()
            # Cheap signal: cosine similarity between model's anchor prediction and this sample
            sim = torch.dot(anchor_pred, F.normalize(vec, dim=0)).item()
            self._similarity_history.append(sim)

            # Diff-based skip: if model already predicts this item well, don't re-learn it.
            # Never skip the first item (anchor) or if we'd empty the batch.
            if idx > 0 and sim > 0.85 and len(items) - skipped > 4:
                skipped += 1
                continue

            if len(self._similarity_history) >= 10:
                sorted_sims = sorted(self._similarity_history)
                threshold = sorted_sims[len(sorted_sims) // 2]
                is_positive = sim > threshold
            else:
                is_positive = True
            self._positive_history.append(1.0 if is_positive else 0.0)
            if not is_positive:
                vec = F.normalize(torch.randn(self.model.input_dim), dim=0)
            patches = getattr(item, "patches", None)  # C.3: (49, 512) or None
            samples.append((vec, is_positive, teacher_vec, patches))

        # Log skip rate periodically
        if skipped > 0 and self._batch_count % 50 == 0:
            print(f"[diff-skip] {skipped}/{len(items)} items skipped (sim>0.85), training on {len(samples)}", flush=True)

        # Append episodic replay samples
        if replay_samples:
            samples.extend(replay_samples)

        # Batched forward+update
        changes, all_activations = self.model.update_batch(samples)

        # Final forward for activations (viz)
        last_vec = items[-1].expected_vector if items[-1].expected_vector is not None else torch.randn(self.model.input_dim)
        prediction, activations = self.model.forward(last_vec, return_activations=True)

        # NOTE: growth_check and health_monitor moved to _step_batch (main thread)
        # because they call store.log_graph_event() which requires same-thread SQLite access.

        elapsed_ms = (_time.perf_counter() - t0) * 1000
        return (changes, prediction, activations, anchor_pred, elapsed_ms, all_activations)

    # ── Speed / stage ──

    def set_speed(self, delay_ms: int) -> None:
        self._delay_ms = delay_ms

    def set_stage(self, stage: int) -> None:
        self._stage = stage

    # ── Status ──

    def get_status(self) -> LoopStatus:
        return LoopStatus(
            state=self._state.value,
            step=self.model.step,
            stage=self._stage,
            delay_ms=self._delay_ms,
            error_message=self._error_message,
            graph_summary=self.model.graph.summary(),
            teacher_healthy=True,
        )

    # ── Human interaction ──

    async def human_message(self, text: str) -> str:
        _, text_encoder, _ = self.encoders
        input_vector = text_encoder.encode(text)
        output_vector, activations = self.model.forward(
            input_vector, return_activations=True,
        )
        # Autoregressive: brain generates token by token, each feeding back
        response = self.decoder.generate(
            output_vector, brain=self.model.brain,
            max_tokens=12, model_step=self.model.step,
        )

        active_clusters = list(activations.keys())

        # Store state for correction-based learning
        self._last_chat_input = input_vector
        self._last_chat_output = output_vector
        self._last_chat_text = text

        self.store.log_human_message(
            step=self.model.step,
            role="human",
            message=text,
        )
        self.store.log_human_message(
            step=self.model.step,
            role="model",
            message=response,
            clusters_active=active_clusters,
        )

        return response

    async def human_correct(self, correction: str) -> str:
        """Baby spoke wrong → human corrects → baby learns from the correction."""
        _, text_encoder, _ = self.encoders
        last_input = getattr(self, '_last_chat_input', None)
        if last_input is None:
            return "(nothing to correct — send a message first)"

        # Encode correction as the teacher vector
        correction_vector = text_encoder.encode(correction)

        # Re-forward to set brain state, then update toward correction
        self.model.forward(last_input, return_activations=False)
        self.model.brain.update(last_input, correction_vector)

        # Train decoder on the correction text
        self.decoder.train_step(correction_vector, correction)

        # Log the correction
        self.store.log_human_message(
            step=self.model.step,
            role="human",
            message=f"[correction] {correction}",
        )

        # Verify what baby says now
        new_output, _ = self.model.forward(last_input, return_activations=False)
        new_response = self.decoder.decode(new_output, max_words=30, model_step=self.model.step)

        self.store.log_human_message(
            step=self.model.step,
            role="model",
            message=f"[after correction] {new_response}",
        )

        # Clear last chat state
        self._last_chat_input = None

        return new_response

    # ── Reset ──

    async def reset(self) -> None:
        await self.pause()
        from model.baby_model_v2 import BabyModelV2
        self.model = BabyModelV2()
        self.curiosity = CuriosityScorer()
        self.health_monitor = HealthMonitor()
        self._recent_questions.clear()
        self._known_words.clear()
        self._similarity_history.clear()
        self._positive_history.clear()
        self._stage = 0
        self._state = LoopState.IDLE
        self._error_message = None

    # ── Internal ──

    def _encode_answer(self, answer: str, item) -> list[torch.Tensor]:
        _, text_encoder, _ = self.encoders
        words = answer.split()

        if len(words) < 30:
            return [text_encoder.encode(answer)]

        sentences = split_sentences(answer)
        vectors = [text_encoder.encode(s) for s in sentences if s.strip()]

        for word in words:
            clean = word.strip(".,!?;:").lower()
            if (
                len(clean) > 4
                and clean not in self._known_words
                and clean.isalpha()
                and clean not in STOPWORDS
            ):
                self._known_words.add(clean)
                self.curriculum.add_teacher_vocabulary(clean)

        return vectors if vectors else [text_encoder.encode(answer)]

    def _compute_is_positive(
        self,
        prediction: torch.Tensor,
        answer_vectors: list[torch.Tensor],
    ) -> bool:
        if not answer_vectors:
            return True
        mean_answer = torch.stack(answer_vectors).mean(dim=0)
        mean_answer = F.normalize(mean_answer, dim=0)
        similarity = torch.dot(prediction, mean_answer).item()
        self._similarity_history.append(similarity)
        # Adaptive threshold: adjust percentile based on positive rate
        if len(self._similarity_history) >= 10:
            sorted_scores = sorted(self._similarity_history)
            # Check positive rate and lower percentile if too many negatives
            percentile = 0.5
            if len(self._positive_history) >= 20:
                positive_rate = sum(self._positive_history) / len(self._positive_history)
                if positive_rate < 0.4:
                    percentile = 0.4
                    print(f"[signal] ratio warning: positive_rate={positive_rate*100:.0f}%, adjusting percentile to 40", flush=True)
            idx = int(len(sorted_scores) * percentile)
            threshold = sorted_scores[idx]
            result = similarity > threshold
            self._positive_history.append(1.0 if result else 0.0)
            print(f"[signal] step={self.model.step} sim={similarity:.4f} threshold={threshold:.4f} positive={result}", flush=True)
            return result
        print(f"[signal] step={self.model.step} sim={similarity:.4f} threshold=warmup positive=True", flush=True)
        self._positive_history.append(1.0)
        return True
