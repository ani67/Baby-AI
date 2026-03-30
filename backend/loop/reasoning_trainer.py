"""
ReasoningTrainer -- curriculum of tasks that require multi-step thinking.

Feeds comparison, sequence, analogy, memory-retrieval, and odd-one-out tasks
to the brain via working memory + reason().  Each task produces a learning
signal (positive reinforcement or teacher correction) that flows into the
existing brain.update() pathway.

    +-----------+      encode       +-------+     reason()     +--------+
    |  task.json| ---- text ------> | brain | ---- steps ----> | output |
    +-----------+                   +-------+                  +--------+
                                       |                           |
                                  working memory            compare to expected
                                  (cleared per task)        --> learning signal
"""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F


class ReasoningTrainer:
    """Feeds reasoning tasks to the brain using working memory + reason()."""

    # Map task type string to runner method name
    _RUNNERS = {
        "comparison": "run_comparison",
        "sequence": "run_sequence",
        "analogy": "run_analogy",
        "memory_retrieval": "run_memory_retrieval",
        "odd_one_out": "run_odd_one_out",
    }

    def __init__(self, tasks_path: str, text_encoder, brain):
        """
        Parameters
        ----------
        tasks_path : str
            Path to reasoning_tasks.json.
        text_encoder : NativeTextEncoder
            The native text encoder (.encode(text) -> (512,) L2-normed).
        brain : BrainState
            The brain instance (.forward(), .reason(), .update(), working memory).
        """
        with open(tasks_path) as f:
            self._tasks: list[dict] = json.load(f)

        self._encoder = text_encoder
        self._brain = brain

        # Partition tasks by type for balanced sampling
        self._by_type: dict[str, list[dict]] = defaultdict(list)
        for t in self._tasks:
            self._by_type[t["type"]].append(t)

        # Round-robin through types
        self._type_cycle = list(self._by_type.keys())
        random.shuffle(self._type_cycle)
        self._type_idx = 0

        # Rolling stats (last 100 per type)
        self._history: dict[str, deque] = {
            tp: deque(maxlen=100) for tp in self._type_cycle
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> torch.Tensor:
        """Encode text to L2-normalized 512-d vector (detached, on brain device)."""
        return self._encoder.encode(text).detach().to(self._brain.device)

    def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two vectors."""
        b = b.to(a.device)
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def _get_memory(self):
        """Return the brain's working memory, creating if needed."""
        if not hasattr(self._brain, "_working_memory") or self._brain._working_memory is None:
            from model.working_memory import WorkingMemory
            self._brain._working_memory = WorkingMemory(
                slots=8, dim=self._brain.dim, device=str(self._brain.device)
            )
        return self._brain._working_memory

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_task(self) -> dict:
        """Return a random task, cycling through types for balance."""
        tp = self._type_cycle[self._type_idx]
        self._type_idx = (self._type_idx + 1) % len(self._type_cycle)
        return random.choice(self._by_type[tp])

    # ------------------------------------------------------------------
    # Task runners
    # ------------------------------------------------------------------

    def run_comparison(self, task: dict) -> tuple[bool, float]:
        """
        Encode input_a, forward through brain, write to memory.
        Encode input_b, forward through brain, read from memory.
        Compare similarity to decide same/different.
        """
        memory = self._get_memory()
        memory.clear()

        vec_a = self._encode(task["input_a"])
        pred_a, _ = self._brain.forward(vec_a)
        memory.write(pred_a.detach())

        vec_b = self._encode(task["input_b"])
        pred_b, _ = self._brain.forward(vec_b)
        mem_read = memory.read(pred_b.detach())

        sim = self._cosine(pred_b, mem_read)

        if task["expected"] == "same":
            correct = sim > 0.5
        else:
            correct = sim < 0.5

        return correct, sim

    def run_sequence(self, task: dict) -> tuple[bool, float]:
        """
        Encode each sequence item through brain sequentially.
        Use reason() on last item to predict next.
        Compare prediction to expected_next encoding.
        """
        memory = self._get_memory()
        memory.clear()

        # Feed sequence items through brain, building up activation buffer
        for item in task["sequence"]:
            vec = self._encode(item)
            pred, _ = self._brain.forward(vec)
            memory.write(pred.detach())

        # Reason from last item to predict next
        last_vec = self._encode(task["sequence"][-1])
        prediction, _ = self._brain.reason(last_vec, steps=3, memory=memory)

        # Compare to expected
        expected_vec = self._encode(task["expected_next"])
        sim = self._cosine(prediction, expected_vec)

        correct = sim > 0.3
        return correct, sim

    def run_analogy(self, task: dict) -> tuple[bool, float]:
        """
        Encode A, B, C.  relation = B - A.  predicted = C + relation.
        Compare to expected encoding.
        """
        vec_a = self._encode(task["a"])
        vec_b = self._encode(task["b"])
        vec_c = self._encode(task["c"])

        relation = vec_b - vec_a
        predicted = F.normalize(vec_c + relation, dim=0)

        expected_vec = self._encode(task["expected"])
        sim = self._cosine(predicted, expected_vec)

        correct = sim > 0.3
        return correct, sim

    def run_memory_retrieval(self, task: dict) -> tuple[bool, float]:
        """
        Encode each context sentence through brain (writes to memory).
        Encode query, use reason() with memory to retrieve answer.
        Compare output to expected answer encoding.
        """
        memory = self._get_memory()
        memory.clear()

        # Store context facts in memory
        for sentence in task["context"]:
            vec = self._encode(sentence)
            pred, _ = self._brain.forward(vec)
            memory.write(pred.detach())

        # Query with reasoning
        query_vec = self._encode(task["query"])
        output, _ = self._brain.reason(query_vec, steps=5, memory=memory)

        expected_vec = self._encode(task["expected"])
        sim = self._cosine(output, expected_vec)

        correct = sim > 0.3
        return correct, sim

    def run_odd_one_out(self, task: dict) -> tuple[bool, float]:
        """
        Encode all items.  Compute pairwise similarities.
        Item with lowest average similarity to others = odd one out.
        """
        items = task["items"]
        vecs = [self._encode(item) for item in items]
        stacked = torch.stack(vecs)  # (N, 512)

        # Pairwise cosine similarity matrix
        sim_matrix = stacked @ stacked.T  # (N, N)

        # Average similarity to others (exclude self)
        n = len(items)
        mask = 1.0 - torch.eye(n)
        avg_sims = (sim_matrix * mask).sum(dim=1) / (n - 1)

        predicted_idx = avg_sims.argmin().item()
        predicted_item = items[predicted_idx]

        correct = predicted_item == task["expected"]

        # Confidence: how much lower is the odd-one's similarity vs the mean of others
        odd_sim = avg_sims[predicted_idx].item()
        other_mean = avg_sims.sum().item() / n
        confidence = max(0.0, other_mean - odd_sim)

        return correct, confidence

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self) -> dict:
        """
        Sample a random task, run it, compute learning signal.

        If correct: positive signal to brain (reinforce current state).
        If wrong:   use expected answer as teacher vector.

        Returns dict with {type, correct, similarity, task}.
        """
        task = self.sample_task()
        tp = task["type"]

        runner = getattr(self, self._RUNNERS[tp])
        correct, similarity = runner(task)

        # Learning signal
        if correct:
            # Reinforce: use the brain's own last prediction as teacher
            if self._brain._last_prediction is not None:
                teacher = F.normalize(self._brain._last_prediction.detach(), dim=0)
                # Mild reinforcement -- small positive push
                if self._brain._last_fired is not None:
                    self._brain.update(teacher, teacher)
        else:
            # Correction: encode the expected answer as teacher
            expected_text = self._extract_expected_text(task)
            if expected_text:
                teacher_vec = self._encode(expected_text)
                # Feed expected through brain to get a proper input vector
                input_text = self._extract_input_text(task)
                if input_text:
                    input_vec = self._encode(input_text)
                    self._brain.forward(input_vec)
                    self._brain.update(input_vec, teacher_vec)

        # Track stats
        self._history[tp].append(1.0 if correct else 0.0)

        return {
            "type": tp,
            "correct": correct,
            "similarity": round(similarity, 4),
            "task": task,
        }

    # ------------------------------------------------------------------
    # Extract text helpers
    # ------------------------------------------------------------------

    def _extract_expected_text(self, task: dict) -> str | None:
        """Pull the expected answer as a text string for encoding."""
        tp = task["type"]
        if tp == "comparison":
            return task["expected"]  # "same" or "different"
        elif tp == "sequence":
            return task["expected_next"]
        elif tp == "analogy":
            return task["expected"]
        elif tp == "memory_retrieval":
            return task["expected"]
        elif tp == "odd_one_out":
            return task["expected"]
        return None

    def _extract_input_text(self, task: dict) -> str | None:
        """Pull a representative input text for the task."""
        tp = task["type"]
        if tp == "comparison":
            return f"{task['input_a']} {task['input_b']}"
        elif tp == "sequence":
            return " ".join(task["sequence"])
        elif tp == "analogy":
            return f"{task['a']} {task['c']}"
        elif tp == "memory_retrieval":
            return task["query"]
        elif tp == "odd_one_out":
            return " ".join(task["items"])
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return accuracy per task type over last 100 tasks."""
        result = {}
        for tp, history in self._history.items():
            if len(history) > 0:
                result[tp] = {
                    "accuracy": round(sum(history) / len(history), 3),
                    "count": len(history),
                }
            else:
                result[tp] = {"accuracy": 0.0, "count": 0}
        return result
