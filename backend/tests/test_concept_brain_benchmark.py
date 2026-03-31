"""
Benchmark: Concept Brain v3 — factual accuracy, inference, chat stability.

Tests the concept graph's ability to answer questions via edge traversal,
multi-hop inference, and template-based generation.

Run with pytest:
    cd backend && python -m pytest tests/test_concept_brain_benchmark.py -v -s

Run standalone:
    cd backend && python tests/test_concept_brain_benchmark.py
"""

import os
import sys
import time
import traceback

# Ensure backend/ is on the path.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

import torch

# Graceful imports — skip if concept modules not available.
try:
    from concept.graph import ConceptGraph
    from concept.node import ConceptNode, TypedEdge
    from concept.parser import ConceptParser
    from concept.generator import ConceptGenerator
    from concept.operations import (
        spread_activation,
        parse_query_intent,
        ActivationResult,
        QueryIntent,
    )
    CONCEPT_AVAILABLE = True
except ImportError as e:
    CONCEPT_AVAILABLE = False
    _IMPORT_ERROR = str(e)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(BACKEND_DIR, "data")
GRAPH_DB_PATH = os.path.join(DATA_DIR, "concept_graph.db")
GRAPH_TENSOR_PATH = os.path.join(DATA_DIR, "concept_vectors.pt")


# ---------------------------------------------------------------------------
# Benchmark questions
# ---------------------------------------------------------------------------

FACTUAL_QUESTIONS: list[tuple[str, str, str]] = [
    # Basic properties
    ("What color is the sky?", "blue", "color"),
    ("What color is an apple?", "red", "color"),
    ("What color is grass?", "green", "color"),
    ("What color is snow?", "white", "color"),
    ("What color is the sun?", "yellow", "color"),
    # Category membership
    ("Is a dog an animal?", "yes", "is_a"),
    ("Is a cat an animal?", "yes", "is_a"),
    ("What is a banana?", "fruit", "is_a"),
    ("What is a rose?", "flower", "is_a"),
    ("What is water?", "liquid", "is_a"),
    # Common knowledge
    ("Where do fish live?", "water", "location"),
    ("What do cows produce?", "milk", "produces"),
    ("Is fire hot or cold?", "hot", "property"),
    ("Is ice hot or cold?", "cold", "property"),
    ("Is the sun bright?", "bright", "property"),
    # Facts from training data
    ("What is the capital of France?", "paris", "factual"),
    ("How many legs does a cat have?", "four", "property"),
    ("What sound does a dog make?", "bark", "action"),
    ("What do birds do?", "fly", "action"),
    ("What do fish do?", "swim", "action"),
]

INFERENCE_QUESTIONS: list[tuple[str, list[str], str]] = [
    ("What color is a sweet fruit?", ["red", "yellow", "orange"], "taste->fruit->color (2 hops)"),
    ("Can animals that bark swim?", ["yes", "dog"], "bark->dog->swim (2 hops)"),
    ("What do friendly pets eat?", ["kibble", "meat", "fish", "food"], "friendly->pet->eat (2 hops)"),
    ("Is something with fur an animal?", ["yes", "animal"], "fur->dog/cat->animal (2 hops)"),
    ("What color are things in the sky?", ["blue", "white", "yellow"], "sky->cloud/sun->color (2 hops)"),
    ("Do things that swim live in water?", ["yes", "water"], "swim->fish->water (2 hops)"),
    ("Are fast animals dangerous?", ["yes", "dangerous"], "fast->animal->danger (2 hops)"),
    ("What shape is the moon?", ["round", "circle"], "moon->shape (1-2 hops)"),
    ("Do plants need water?", ["yes", "water"], "plant->need->water (2 hops)"),
    ("Are big animals strong?", ["yes", "strong"], "big->animal->strong (2 hops)"),
]

CHAT_QUESTIONS: list[tuple[str, list[str]]] = [
    ("what is a dog", ["animal", "pet", "bark", "mammal", "friend"]),
    ("the sky is", ["blue", "above", "high", "clear", "big"]),
    ("two plus three equals", ["five", "5"]),
]


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

def query_graph(
    graph: ConceptGraph,
    question: str,
    parser: ConceptParser,
    generator: ConceptGenerator,
) -> str:
    """Run a question through the concept brain pipeline.

    parse -> find seed concepts -> spread activation -> generate text.
    """
    # Parse question to extract concepts.
    parse_result = parser.parse_text(question)
    concept_names = parse_result.concepts

    # Also extract keywords from question directly.
    keywords = [w.lower().strip(".,!?;:\"'()-") for w in question.lower().split()]
    keywords = [w for w in keywords if len(w) > 2 and w not in {
        "what", "how", "where", "who", "when", "why", "does", "the",
        "is", "are", "was", "were", "can", "and", "or", "not", "has",
        "have", "had", "do", "did", "will", "would", "that", "this",
    }]

    # Find seed nodes: try by name first, then by vector similarity.
    seeds: list[tuple[str, float]] = []
    all_names = set(concept_names) | set(keywords)

    for name in all_names:
        matches = graph.get_by_name(name)
        if matches:
            seeds.append((matches[0].id, 1.0))

    # If no name matches, try vector similarity via the parser's concepts.
    # (Requires encoder, but we can still proceed with name-based lookup.)

    if not seeds:
        return ""

    # Spread activation.
    # The operations module expects graph.nodes and graph.get_edges_from —
    # adapt by using the graph's internal API directly.
    activated: dict[str, float] = {}
    paths: dict[str, list[str]] = {}

    frontier = list(seeds)
    for node_id, strength in frontier:
        node = graph.get_node(node_id)
        if node is not None:
            activated[node_id] = strength

    # Manual 2-hop spread (operations.spread_activation expects .nodes attribute).
    for hop in range(3):
        next_frontier: list[tuple[str, float]] = []
        for node_id, act in frontier:
            edges = graph.get_edges(node_id, direction="outgoing")
            for edge in edges:
                new_act = act * edge.strength * 0.5
                if new_act < 0.05:
                    continue
                tid = edge.target_id
                if tid not in activated or new_act > activated[tid]:
                    activated[tid] = new_act
                    next_frontier.append((tid, new_act))
        frontier = next_frontier

    if not activated:
        return ""

    # Parse query intent for edge filtering.
    intent = parse_query_intent(question)

    # Collect activated nodes with names.
    result_parts: list[tuple[str, float]] = []
    for nid, score in sorted(activated.items(), key=lambda x: x[1], reverse=True)[:20]:
        node = graph.get_node(nid)
        if node is not None and node.name:
            result_parts.append((node.name, score))

    # Also try direct edge traversal for factual queries.
    for node_id, _ in seeds:
        if intent.target_relation:
            edges = graph.get_edges(node_id, relation=intent.target_relation, direction="outgoing")
            for edge in edges:
                target = graph.get_node(edge.target_id)
                if target and target.name:
                    result_parts.append((target.name, edge.strength * 2.0))

        # Always check outgoing edges for relevant info.
        edges = graph.get_edges(node_id, direction="outgoing")
        for edge in edges:
            target = graph.get_node(edge.target_id)
            if target and target.name:
                result_parts.append((target.name, edge.strength))

    # Deduplicate and sort by score.
    seen: set[str] = set()
    unique_parts: list[tuple[str, float]] = []
    for name, score in sorted(result_parts, key=lambda x: x[1], reverse=True):
        if name not in seen:
            seen.add(name)
            unique_parts.append((name, score))

    # Build response from top concept names.
    response_words = [name for name, _ in unique_parts[:10]]
    return " ".join(response_words)


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def get_graph_memory_mb(graph: ConceptGraph) -> float:
    """Estimate memory footprint of the graph in MB."""
    total_bytes = 0

    # Vector matrix.
    if graph._vector_matrix is not None:
        total_bytes += graph._vector_matrix.nelement() * graph._vector_matrix.element_size()

    # Per-node vectors.
    for node in graph._nodes.values():
        total_bytes += node.vector.nelement() * node.vector.element_size()
        for vec in node.modality_vectors.values():
            total_bytes += vec.nelement() * vec.element_size()

    # Edge metadata (estimate ~100 bytes per edge for Python objects).
    total_bytes += graph.edge_count * 100

    # Node metadata (estimate ~200 bytes per node for Python objects).
    total_bytes += graph.node_count * 200

    return total_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkResults:
    def __init__(self):
        self.factual_correct = 0
        self.factual_total = 0
        self.inference_correct = 0
        self.inference_total = 0
        self.chat_results: dict[str, dict] = {}
        self.total_query_time_ms = 0.0
        self.query_count = 0
        self.memory_mb = 0.0

    @property
    def factual_pct(self) -> float:
        return (self.factual_correct / self.factual_total * 100) if self.factual_total else 0.0

    @property
    def inference_pct(self) -> float:
        return (self.inference_correct / self.inference_total * 100) if self.inference_total else 0.0

    @property
    def avg_query_ms(self) -> float:
        return (self.total_query_time_ms / self.query_count) if self.query_count else 0.0


def run_benchmark() -> BenchmarkResults:
    """Run the full benchmark suite. Returns results object."""
    results = BenchmarkResults()

    # Load graph.
    if not os.path.exists(GRAPH_DB_PATH) or not os.path.exists(GRAPH_TENSOR_PATH):
        print("[benchmark] graph files not found — run migrate_to_v3.py first")
        print(f"  expected: {GRAPH_DB_PATH}")
        print(f"  expected: {GRAPH_TENSOR_PATH}")
        return results

    print("[benchmark] loading concept graph...")
    graph = ConceptGraph.load(GRAPH_DB_PATH, GRAPH_TENSOR_PATH)
    summary = graph.summary()
    print(f"  nodes: {summary['node_count']:,}")
    print(f"  edges: {summary['edge_count']:,}")
    print(f"  relations: {summary['relation_types']}")

    results.memory_mb = get_graph_memory_mb(graph)
    print(f"  memory: {results.memory_mb:.1f} MB")

    parser = ConceptParser()
    generator = ConceptGenerator()

    # -- Level 1: Factual --------------------------------------------------
    print("\n" + "=" * 60)
    print("LEVEL 1: FACTUAL QUESTIONS")
    print("=" * 60)

    results.factual_total = len(FACTUAL_QUESTIONS)
    for question, expected, relation_type in FACTUAL_QUESTIONS:
        t0 = time.perf_counter()
        answer = query_graph(graph, question, parser, generator)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.total_query_time_ms += elapsed_ms
        results.query_count += 1

        correct = expected.lower() in answer.lower()
        if correct:
            results.factual_correct += 1

        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] {question}")
        print(f"         expected: {expected}  |  got: {answer[:80]}  ({elapsed_ms:.1f}ms)")

    print(f"\n  Factual: {results.factual_correct}/{results.factual_total} "
          f"({results.factual_pct:.0f}%)")

    # -- Level 2: Inference ------------------------------------------------
    print("\n" + "=" * 60)
    print("LEVEL 2: INFERENCE QUESTIONS")
    print("=" * 60)

    results.inference_total = len(INFERENCE_QUESTIONS)
    for question, expected_any, trace in INFERENCE_QUESTIONS:
        t0 = time.perf_counter()
        answer = query_graph(graph, question, parser, generator)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.total_query_time_ms += elapsed_ms
        results.query_count += 1

        correct = any(e.lower() in answer.lower() for e in expected_any)
        if correct:
            results.inference_correct += 1

        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] {question}  ({trace})")
        print(f"         expected any of: {expected_any}  |  got: {answer[:80]}  ({elapsed_ms:.1f}ms)")

    print(f"\n  Inference: {results.inference_correct}/{results.inference_total} "
          f"({results.inference_pct:.0f}%)")

    # -- Level 3: Chat Benchmark -------------------------------------------
    print("\n" + "=" * 60)
    print("LEVEL 3: CHAT BENCHMARK (stability + relevance)")
    print("=" * 60)

    for question, relevant_words in CHAT_QUESTIONS:
        answers: list[str] = []
        times_ms: list[float] = []

        for trial in range(5):
            t0 = time.perf_counter()
            answer = query_graph(graph, question, parser, generator)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            answers.append(answer)
            times_ms.append(elapsed_ms)
            results.total_query_time_ms += elapsed_ms
            results.query_count += 1

        # Stability: how many unique answers out of 5 (1 = perfectly stable).
        unique_answers = len(set(answers))
        stability = 1.0 / unique_answers if unique_answers > 0 else 0.0

        # Relevance: does the answer contain any relevant word?
        sample = answers[0]
        relevant = any(w.lower() in sample.lower() for w in relevant_words)

        results.chat_results[question] = {
            "answers": answers,
            "unique_count": unique_answers,
            "stability": stability,
            "relevant": relevant,
            "sample": sample,
            "avg_ms": sum(times_ms) / len(times_ms),
        }

        status = "PASS" if relevant else "FAIL"
        print(f"\n  [{status}] \"{question}\"")
        print(f"         answer:    \"{sample[:80]}\"")
        print(f"         relevant:  {relevant}  (checking for: {relevant_words})")
        print(f"         stability: {unique_answers}/5 unique answers")
        print(f"         avg time:  {sum(times_ms) / len(times_ms):.1f}ms")

    # -- Summary Table -----------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<30s} {'Result':>15s}  {'Target':>15s}  {'Pass':>6s}")
    print(f"  {'-' * 30} {'-' * 15}  {'-' * 15}  {'-' * 6}")

    factual_str = f"{results.factual_correct}/{results.factual_total} ({results.factual_pct:.0f}%)"
    factual_pass = results.factual_correct >= 6
    print(f"  {'Factual accuracy':<30s} {factual_str:>15s}  {'>=30% (6/20)':>15s}  {'YES' if factual_pass else 'NO':>6s}")

    time_str = f"{results.avg_query_ms:.1f}ms"
    time_pass = results.avg_query_ms < 100
    print(f"  {'Avg query time':<30s} {time_str:>15s}  {'<100ms':>15s}  {'YES' if time_pass else 'NO':>6s}")

    mem_str = f"{results.memory_mb:.1f}MB"
    mem_pass = results.memory_mb < 100
    print(f"  {'Memory usage':<30s} {mem_str:>15s}  {'<100MB':>15s}  {'YES' if mem_pass else 'NO':>6s}")

    chat_relevant = sum(1 for r in results.chat_results.values() if r["relevant"])
    chat_total = len(results.chat_results)
    chat_str = f"{chat_relevant}/{chat_total}"
    chat_pass = chat_relevant >= 2
    print(f"  {'Chat relevance':<30s} {chat_str:>15s}  {'>=2/3':>15s}  {'YES' if chat_pass else 'NO':>6s}")

    print(f"\n  Overall: {'PASS' if (factual_pass and time_pass and mem_pass) else 'NEEDS IMPROVEMENT'}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# pytest interface
# ---------------------------------------------------------------------------

import pytest


@pytest.fixture(scope="module")
def benchmark_results():
    """Run benchmark once and share results across all test functions."""
    if not CONCEPT_AVAILABLE:
        pytest.skip(f"concept modules not available: {_IMPORT_ERROR}")
    if not os.path.exists(GRAPH_DB_PATH) or not os.path.exists(GRAPH_TENSOR_PATH):
        pytest.skip("concept graph not migrated yet — run migrate_to_v3.py first")
    return run_benchmark()


def test_factual_accuracy(benchmark_results):
    """Factual accuracy should be at least 30% (6/20)."""
    assert benchmark_results.factual_correct >= 6, (
        f"Factual accuracy too low: {benchmark_results.factual_correct}/{benchmark_results.factual_total} "
        f"({benchmark_results.factual_pct:.0f}%), need >=30%"
    )


def test_response_time(benchmark_results):
    """Average query response time should be under 100ms."""
    assert benchmark_results.avg_query_ms < 100, (
        f"Response time too slow: {benchmark_results.avg_query_ms:.1f}ms, need <100ms"
    )


def test_memory_usage(benchmark_results):
    """Graph memory usage should be under 100MB."""
    assert benchmark_results.memory_mb < 100, (
        f"Memory usage too high: {benchmark_results.memory_mb:.1f}MB, need <100MB"
    )


def test_chat_stability(benchmark_results):
    """Chat answers should be deterministic (concept graph is stateless per query)."""
    for question, result in benchmark_results.chat_results.items():
        # Concept graph should produce identical answers every time.
        assert result["unique_count"] == 1, (
            f'"{question}" produced {result["unique_count"]} unique answers in 5 runs, '
            f"expected deterministic output"
        )


def test_chat_relevance(benchmark_results):
    """At least 2 of 3 chat questions should produce relevant answers."""
    relevant_count = sum(1 for r in benchmark_results.chat_results.values() if r["relevant"])
    assert relevant_count >= 2, (
        f"Only {relevant_count}/3 chat questions produced relevant answers, need >=2"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not CONCEPT_AVAILABLE:
        print(f"ERROR: concept modules not available: {_IMPORT_ERROR}")
        sys.exit(1)

    results = run_benchmark()

    # Exit code: 0 if factual >= 30% and time < 100ms and memory < 100MB.
    passed = (
        results.factual_correct >= 6
        and results.avg_query_ms < 100
        and results.memory_mb < 100
    )
    sys.exit(0 if passed else 1)
