"""
LearningLoop — the autonomous learning cycle orchestrator.
"""

import asyncio
import enum
import logging
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

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
        """Executes one full learning cycle."""

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

        # ── 5. ASK ──
        # Load image bytes if available for vision-capable models
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

        # ── 6. ENCODE ──
        answer_vectors = self._encode_answer(
            teacher_response.answer,
            curriculum_item,
        )

        # ── 7. PREDICT ──
        # Always run forward to build activations for growth tracking
        input_vec = answer_vectors[0] if answer_vectors else torch.randn(self.model.input_dim)
        prediction, activations = self.model.forward(
            input_vec, return_activations=True,
        )
        if self._stage >= 1:
            is_positive = self._compute_is_positive(
                prediction=prediction,
                answer_vectors=answer_vectors,
            )
        else:
            # Stage 0: every 3rd step is a negative example (random vector)
            is_positive = (self.model.step % 3 != 0)
            if answer_vectors:
                mean_answer = torch.stack(answer_vectors).mean(dim=0)
                mean_answer = F.normalize(mean_answer, dim=0)
                stage0_sim = torch.dot(prediction, mean_answer).item()
            else:
                stage0_sim = 0.0
            print(f"[signal] step={self.model.step} sim={stage0_sim:.4f} stage0 positive={is_positive}", flush=True)

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
        active_count = sum(1 for c in self.model.graph.clusters if not c.dormant)
        total_clusters = len(self.model.graph.clusters)
        self.health_monitor.record_step(active_count, total_clusters)
        self.health_monitor.check(
            step=self.model.step,
            stage=self._stage,
            model=self.model,
            positive_history=self._positive_history,
            similarity_history=self._similarity_history,
        )

        # ── 11. LOG ──
        self.store.log_dialogue(
            step=self.model.step,
            stage=self._stage,
            question=question,
            answer=teacher_response.answer,
            curiosity_score=curiosity_score,
            clusters_active=list(activations.keys()),
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
            for cluster in self.model.graph.clusters:
                for node in cluster.nodes:
                    state_dict[f"{node.id}.weights"] = node.weights
                    state_dict[f"{node.id}.bias"] = node.bias
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
        # Decode model's prediction into words for the frontend
        model_response = self.decoder.decode(prediction, max_words=15)

        if self.viz_emitter is not None:
            await self.viz_emitter.emit_step(
                step=self.model.step,
                stage=self._stage,
                graph=self.model.graph,
                activations=activations,
                last_question=question,
                last_answer=teacher_response.answer,
                model_answer=model_response,
                is_positive=is_positive,
                growth_events=growth_events,
                image_url=f"/images/{curriculum_item.image_path}" if curriculum_item.image_path else None,
            )

        # ── 13. AUTO-ADVANCE stage 0 → 1 ──
        if self._stage == 0:
            active_count = sum(1 for c in self.model.graph.clusters if not c.dormant)

            if (self.model.step >= 800
                    and active_count >= 60
                    and self._stage0_completion_step is None):
                self._stage0_completion_step = self.model.step + 100
                print(f"[stage] advance condition met: step>=800 clusters={active_count}, advancing in 100 steps", flush=True)

            if self._stage0_completion_step is not None and self.model.step >= self._stage0_completion_step:
                self.set_stage(1)
                self.model.stage = 1
                print(f"[stage] auto-advanced to stage 1 at step={self.model.step}", flush=True)

        # ── 14. AUTO-ADVANCE stage 1 → 2 ──
        if self._stage == 1:
            active_count = sum(1 for c in self.model.graph.clusters if not c.dormant)
            positive_rate = (
                sum(self._positive_history) / len(self._positive_history)
                if len(self._positive_history) >= 20 else 0.0
            )

            if (self.model.step >= 3000
                    and active_count > 120
                    and positive_rate > 0.55
                    and self._stage1_completion_step is None):
                self._stage1_completion_step = self.model.step + 100
                print(f"[stage] advance 1→2 condition met: step>=3000 clusters={active_count} positive_rate={positive_rate*100:.0f}%, advancing in 100 steps", flush=True)

            if self._stage1_completion_step is not None and self.model.step >= self._stage1_completion_step:
                self.set_stage(2)
                self.model.stage = 2
                print(f"[stage] auto-advanced to stage 2 at step={self.model.step}", flush=True)

        return StepResult(
            step=self.model.step,
            question=question,
            answer=teacher_response.answer,
            curiosity_score=curiosity_score,
            is_positive=is_positive,
            delta_summary=delta_summary,
            growth_events=growth_events,
            duration_ms=teacher_response.duration_ms,
            skipped=False,
        )

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
        response = self.decoder.decode(output_vector, max_words=30)

        active_clusters = list(activations.keys())

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

    # ── Reset ──

    async def reset(self) -> None:
        await self.pause()
        from model.baby_model import BabyModel
        self.model = BabyModel()
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
