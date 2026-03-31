"""
Migrate training data into Concept Brain v3 graph.

Loads all training datasets, parses them through ConceptParser,
encodes concept vectors via NativeTextEncoder, writes to ConceptGraph,
and optionally attaches COCO image modality vectors.

Usage:
    cd backend && python scripts/migrate_to_v3.py
"""

import glob
import json
import os
import sys
import time

import torch

# Ensure backend/ is on the path.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from concept.parser import ConceptParser, ParseResult, ParsedTriple
from concept.graph import ConceptGraph
from encoder.vocab import Vocabulary
from encoder.native_text import NativeTextEncoder


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(BACKEND_DIR, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

CURRICULUM_FILES = [
    os.path.join(DATA_DIR, "text_curriculum.json"),
    os.path.join(DATA_DIR, "text_commonsense.json"),
    os.path.join(DATA_DIR, "text_diverse.json"),
    os.path.join(DATA_DIR, "text_conversations.json"),
    os.path.join(DATA_DIR, "reasoning_tasks.json"),
]

TEXT_ENCODER_CHECKPOINT = os.path.join(DATA_DIR, "checkpoints", "native_text.pt")
COCO_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "checkpoints", "coco_clip_embeddings.pt")

GRAPH_DB_PATH = os.path.join(DATA_DIR, "concept_graph.db")
GRAPH_TENSOR_PATH = os.path.join(DATA_DIR, "concept_vectors.pt")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_file(path: str) -> list[dict]:
    """Load a JSON file, returning a list of items."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  [skip] {os.path.basename(path)}: {e}")
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Some datasets are dict-of-lists; flatten to list of dicts.
        keys = list(data.keys())
        if keys and isinstance(data[keys[0]], list):
            first_list = data[keys[0]]
            if first_list and isinstance(first_list[0], dict):
                return first_list
        # Single-item dict: wrap it.
        return [data]
    return []


def load_all_datasets() -> list[dict]:
    """Load every training dataset and curriculum file."""
    all_items: list[dict] = []

    # Dataset directory (*.json only, skip subdirectories).
    dataset_paths = sorted(glob.glob(os.path.join(DATASETS_DIR, "*.json")))
    for path in dataset_paths:
        items = load_json_file(path)
        if items:
            print(f"  {os.path.basename(path):30s} -> {len(items):>7,} items")
            all_items.extend(items)

    # Curriculum / supplementary files.
    for path in CURRICULUM_FILES:
        if os.path.exists(path):
            items = load_json_file(path)
            if items:
                print(f"  {os.path.basename(path):30s} -> {len(items):>7,} items")
                all_items.extend(items)

    return all_items


# ---------------------------------------------------------------------------
# Encoder setup
# ---------------------------------------------------------------------------

def load_text_encoder() -> NativeTextEncoder:
    """Initialize vocab + NativeTextEncoder and restore checkpoint."""
    vocab = Vocabulary(max_size=8192)
    word_embeddings = torch.randn(len(vocab.word_to_id), 512) * 0.02
    word_embeddings.requires_grad_(True)

    encoder = NativeTextEncoder(vocab, word_embeddings)

    if os.path.exists(TEXT_ENCODER_CHECKPOINT):
        checkpoint = torch.load(TEXT_ENCODER_CHECKPOINT, weights_only=False)

        # Restore word embeddings if present.
        if "word_embeddings" in checkpoint:
            saved_embs = checkpoint["word_embeddings"]
            n = min(saved_embs.shape[0], word_embeddings.shape[0])
            word_embeddings.data[:n] = saved_embs[:n]
            print(f"[encoder] restored {n} word embeddings")

        # Restore MLP + attention weights.
        encoder.load_state_dict(checkpoint)
        print(f"[encoder] loaded checkpoint from {TEXT_ENCODER_CHECKPOINT}")
    else:
        print(f"[encoder] no checkpoint at {TEXT_ENCODER_CHECKPOINT}, using random init")

    return encoder


# ---------------------------------------------------------------------------
# Vector cache
# ---------------------------------------------------------------------------

class VectorCache:
    """Cache encoded vectors so each concept name is encoded only once."""

    def __init__(self, encoder: NativeTextEncoder):
        self._encoder = encoder
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, concept_name: str) -> torch.Tensor:
        if concept_name not in self._cache:
            with torch.no_grad():
                self._cache[concept_name] = self._encoder.encode(concept_name)
        return self._cache[concept_name]

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# COCO image modality
# ---------------------------------------------------------------------------

def attach_coco_images(graph: ConceptGraph, vector_cache: VectorCache) -> int:
    """Load COCO embeddings and attach as image modality to nearest concepts."""
    if not os.path.exists(COCO_EMBEDDINGS_PATH):
        print("[coco] no embeddings file found, skipping image modality")
        return 0

    print("[coco] loading embeddings...")
    coco_data = torch.load(COCO_EMBEDDINGS_PATH, weights_only=False)

    # coco_data may be dict[str, Tensor] or dict with 'embeddings' key.
    if isinstance(coco_data, dict) and "embeddings" in coco_data:
        embeddings = coco_data["embeddings"]
    elif isinstance(coco_data, dict):
        embeddings = coco_data
    else:
        print("[coco] unexpected format, skipping")
        return 0

    attached = 0
    total = len(embeddings) if isinstance(embeddings, dict) else 0

    if isinstance(embeddings, dict):
        for i, (fname, vec) in enumerate(embeddings.items()):
            if not isinstance(vec, torch.Tensor):
                continue

            # Ensure vector is the right shape.
            if vec.dim() > 1:
                vec = vec.squeeze()
            if vec.shape[0] != 512:
                continue

            vec = vec.detach().float()
            matches = graph.find_similar(vec, k=1)
            if matches and matches[0][1] > 0.7:
                node = matches[0][0]
                node.modality_vectors["image"] = vec
                node.modalities.add("image")
                if "image" not in node.modality_weights:
                    node.modality_weights["image"] = 1.0
                attached += 1

            if (i + 1) % 5000 == 0:
                print(f"  [coco] processed {i + 1:,}/{total:,} embeddings, attached {attached:,}")

    print(f"[coco] attached {attached:,} image vectors to concepts")
    return attached


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------

def migrate() -> None:
    t_start = time.time()

    # Step 1: Load all training data.
    print("\n=== Step 1: Loading datasets ===")
    items = load_all_datasets()
    print(f"\nTotal items loaded: {len(items):,}")

    if not items:
        print("No training data found. Exiting.")
        return

    # Step 2: Parse all items.
    print("\n=== Step 2: Parsing items ===")
    parser = ConceptParser()
    skipped = 0
    results: list[ParseResult] = []

    def progress(done: int, total: int) -> None:
        if done % 10_000 < 100 or done == total:
            elapsed = time.time() - t_start
            rate = done / max(elapsed, 0.01)
            print(f"  parsed {done:>8,}/{total:,}  ({rate:,.0f} items/sec)")

    for i, item in enumerate(items):
        try:
            result = parser.parse_item(item)
            if result.triples:
                results.append(result)
            else:
                skipped += 1
        except Exception:
            skipped += 1

        if (i + 1) % 10_000 == 0 or (i + 1) == len(items):
            progress(i + 1, len(items))

    print(f"\nParsed: {len(results):,} items with triples, skipped: {skipped:,}")

    # Collect unique concepts.
    unique_concepts: set[str] = set()
    total_triples = 0
    for r in results:
        unique_concepts.update(r.concepts)
        total_triples += len(r.triples)
    print(f"Unique concepts found: {len(unique_concepts):,}")
    print(f"Total triples: {total_triples:,}")

    # Step 3: Vectorize concepts.
    print("\n=== Step 3: Vectorizing concepts ===")
    encoder = load_text_encoder()
    vector_cache = VectorCache(encoder)

    # Pre-encode all unique concept names.
    concept_list = sorted(unique_concepts)
    for i, name in enumerate(concept_list):
        vector_cache.get(name)
        if (i + 1) % 5_000 == 0 or (i + 1) == len(concept_list):
            print(f"  encoded {i + 1:,}/{len(concept_list):,} concepts")

    print(f"Vector cache size: {vector_cache.size:,}")

    # Step 4: Write to graph.
    print("\n=== Step 4: Writing to concept graph ===")
    graph = ConceptGraph()

    for i, result in enumerate(results):
        # Build triple tuples: (subject, relation, object).
        triples = [(t.subject, t.relation, t.object) for t in result.triples]

        # Build vector dict for concepts in this result.
        vectors: dict[str, torch.Tensor] = {}
        for concept_name in result.concepts:
            vectors[concept_name] = vector_cache.get(concept_name)

        graph.write(triples, vectors, result.modality)

        if (i + 1) % 10_000 == 0 or (i + 1) == len(results):
            print(f"  written {i + 1:,}/{len(results):,} items  "
                  f"(nodes: {graph.node_count:,}, edges: {graph.edge_count:,})")

    # Step 5: Attach COCO image vectors.
    print("\n=== Step 5: COCO image modality ===")
    image_count = attach_coco_images(graph, vector_cache)

    # Step 6: Dedup pass.
    print("\n=== Step 6: Deduplication ===")
    if graph.node_count < 50_000:
        merged = graph.dedup_pass(threshold=0.95)
        print(f"Merged {merged:,} near-duplicate concepts")
    else:
        print("Skipping full dedup (graph too large for brute-force), will run incrementally")

    # Step 7: Save.
    print("\n=== Step 7: Saving graph ===")
    os.makedirs(os.path.dirname(GRAPH_DB_PATH), exist_ok=True)
    graph.save(GRAPH_DB_PATH, GRAPH_TENSOR_PATH)
    print(f"Saved: {GRAPH_DB_PATH}")
    print(f"Saved: {GRAPH_TENSOR_PATH}")

    # Summary.
    elapsed = time.time() - t_start
    summary = graph.summary()
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"  Concepts:       {summary['node_count']:>8,}")
    print(f"  Edges:          {summary['edge_count']:>8,}")
    print(f"  Relation types: {len(summary['relation_types']):>8,}  {summary['relation_types']}")
    print(f"  Modalities:     {summary['modalities']}")
    print(f"  Images attached: {image_count:>7,}")
    print(f"  Time elapsed:   {elapsed:>8.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    migrate()
