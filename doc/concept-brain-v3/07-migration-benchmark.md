# Migration + Benchmark Design

## 1. Migration Strategy

### Option C: Ignore old neurons, re-ingest training data

BrainV2's 10K neurons are 512-dim "smoothies" — each blends many concepts. Extracting clean concepts from smoothies is lossy and unreliable. Instead: feed all 482K training items through the new parser → concept graph. Fresh start, clean knowledge.

**Why not import neurons:**
- A neuron whose vector ≈ 0.3×dog + 0.4×cat + 0.3×animal isn't any single concept
- Clustering neurons into concepts requires arbitrary thresholds
- The parser produces BETTER concepts from raw data than reverse-engineering neurons

**What we keep from the old system:**
- Pre-encoded COCO embeddings (113MB, `coco_clip_embeddings.pt`) — reuse as image modality vectors
- Native encoder checkpoints (text + vision) — used by parser for vectorization
- Training data files (482K items) — the actual knowledge source

### Ingestion Pipeline

```
Step 1: Load all datasets (2 minutes)
  Read 9 JSON files from data/datasets/
  Read existing curriculum files (text_curriculum, text_commonsense, etc.)
  Total: ~482K items

Step 2: Parse all items (3-4 minutes)
  parser.parse_dataset(all_items) → list of ParseResult
  Each ParseResult: concepts + typed relations + modality
  Expected: ~5K-10K unique concepts, ~30K-60K relations

Step 3: Vectorize concepts (2-3 minutes)
  For each unique concept name:
    text_vector = native_text_encoder.encode(name)
  For COCO images:
    image_vector = load from coco_clip_embeddings.pt

Step 4: Write to graph (1-2 minutes)
  For each ParseResult:
    graph.write(triples, vectors, modality)
  Handles dedup automatically (find_or_create)

Step 5: Initial clustering (30 seconds)
  graph.cluster() → emergent regions
  graph.build_hierarchy() → fractal levels

Step 6: Save checkpoint
  graph.save(db_path, tensor_path)

Total: ~8-10 minutes for full ingestion.
```

### Migration Script

```python
# backend/scripts/migrate_to_v3.py
def migrate():
    # Load encoders
    text_enc = NativeTextEncoder(...)
    text_enc.load(checkpoint)

    # Load all training data
    items = load_all_datasets()  # 482K items

    # Parse
    parser = ConceptParser(vocab)
    results = parser.parse_dataset(items, progress_callback=print_progress)

    # Vectorize + write
    graph = ConceptGraph(dim=512)
    for result in results:
        vectors = {}
        for concept_name in result.concepts:
            vectors[concept_name] = text_enc.encode(concept_name)
        graph.write(result.triples, vectors, result.modality)

    # Add COCO image vectors
    coco_embs = torch.load("data/checkpoints/coco_clip_embeddings.pt")
    for fname, vec in coco_embs.items():
        # Find matching concept by nearest vector
        matches = graph.find_similar(vec, k=1)
        if matches and matches[0][1] > 0.7:
            node = matches[0][0]
            node.modality_vectors["image"] = vec
            node.modalities.add("image")

    # Cluster + save
    graph.cluster()
    graph.build_hierarchy()
    graph.save("data/concept_graph.db", "data/concept_vectors.pt")

    print(f"Migration complete: {graph.node_count} concepts, {graph.edge_count} edges")
```

## 2. Benchmark Design

### Level 1: Factual (50 questions)

Questions the concept graph should answer from stored edges:

```python
FACTUAL_QUESTIONS = [
    # Basic properties
    ("What color is the sky?", "blue", "color"),
    ("What color is an apple?", "red", "color"),
    ("What sound does a dog make?", "bark", "action:sound"),
    ("Is fire hot or cold?", "hot", "property"),
    ("How many legs does a cat have?", "four", "property:count"),

    # Category membership
    ("Is a dog an animal?", "yes", "is_a"),
    ("Is a car an animal?", "no", "is_a"),
    ("What is a banana?", "fruit", "is_a"),

    # Common knowledge
    ("Where do fish live?", "water", "location"),
    ("What do cows produce?", "milk", "produces"),

    # From our training data
    ("What is the capital of France?", "paris", "factual"),
    ("Who wrote Romeo and Juliet?", "shakespeare", "factual"),
    # ... 40 more covering colors, animals, foods, science, geography
]
```

**Scoring:** exact match OR synonym match (via graph similar_to edges). Score = correct / total.

### Level 2: Inference (30 questions)

Questions requiring 2+ hop traversal:

```python
INFERENCE_QUESTIONS = [
    ("What color is a sweet fruit?", ["red", "yellow"], "taste→fruit→color (2 hops)"),
    ("Can animals that bark swim?", ["yes"], "bark→dog→swim (2 hops)"),
    ("What do friendly pets eat?", ["kibble", "meat", "fish"], "friendly→pet→dog/cat→eats (2-3 hops)"),
    ("Is something that has fur an animal?", ["yes"], "fur→dog/cat→animal→is_a (2 hops)"),
    # ... 26 more
]
```

**Scoring:** any correct answer in result set = pass. Score = correct / total.

### Level 3: Creative (20 prompts)

Composition and novel concept creation:

```python
CREATIVE_PROMPTS = [
    ("Describe a purple dog", "should mention: dog properties + purple color"),
    ("What if fish could fly?", "should combine fish + flying concepts"),
    ("Describe a cold fire", "should combine fire properties + cold"),
    # ... 17 more
]
```

**Scoring:** human evaluation or keyword matching (contains relevant concept words).

### The 3 Benchmark Chat Questions (continuity)

Same as before, tracked over time:
```
Q1: "what is a dog"          — tests object knowledge
Q2: "the sky is"             — tests common sense completion
Q3: "two plus three equals"  — tests basic math
```

### A/B Comparison Methodology

```
For each question, run on BOTH systems:

BrainV2 path:
  encode(question) → brain.forward() → decode_from_brain() → answer

ConceptBrain path:
  parse(question) → activate(concepts) → infer(edges) → generate(text) → answer

Metrics per question:
  1. Correctness: does the answer contain the right concept? (0/1)
  2. Relevance: how many output words are related to the topic? (0-1)
  3. Stability: ask 5 times, how many identical answers? (0-1)
  4. Explainability: can it trace WHY it answered this? (0/1)
  5. Response time: milliseconds
  6. Memory used: bytes

Aggregate:
  System with better score on 4+ of 6 metrics = winner
```

### Success Criteria

```
MUST BEAT BrainV2 on:
  ✓ Factual accuracy (BrainV2 baseline: ~0% — can't answer factual questions)
  ✓ Response time (BrainV2: ~8-25 seconds. Target: <100ms)
  ✓ Memory (BrainV2: ~2GB. Target: <100MB)
  ✓ Explainability (BrainV2: zero. Target: full edge trace)

MAY LOSE on:
  ✗ Fluency (templates vs neural generation)
  ✗ Creativity (structured composition vs emergent patterns)

CONCRETE TARGETS:
  Level 1 (factual): ≥60% accuracy (30/50 correct)
  Level 2 (inference): ≥30% accuracy (9/30 correct)
  Level 3 (creative): human-judged "relevant" ≥50%
  Chat Q1 "what is a dog": answer contains "animal" or "pet"
  Chat Q2 "the sky is": answer contains "blue" or "above"
  Chat Q3 "two plus three equals": answer contains "five" or "5"
  Response time: <100ms per query
  Memory: <100MB total
```

## 3. Benchmark Script Structure

```python
# backend/tests/test_concept_brain_benchmark.py

def benchmark_concept_brain():
    # Load concept graph
    graph = ConceptGraph.load(db_path, tensor_path)
    generator = ConceptGenerator()

    # Level 1: Factual
    factual_correct = 0
    for question, expected, relation_type in FACTUAL_QUESTIONS:
        result = query_graph(graph, question)
        if expected.lower() in result.lower():
            factual_correct += 1

    # Level 2: Inference
    inference_correct = 0
    for question, expected_any, trace in INFERENCE_QUESTIONS:
        result = query_graph(graph, question)
        if any(e.lower() in result.lower() for e in expected_any):
            inference_correct += 1

    # Level 3: Creative
    # ... human eval or keyword scoring

    # Chat benchmark (same 3 questions, 5 times each for stability)
    chat_results = {}
    for q in ["what is a dog", "the sky is", "two plus three equals"]:
        answers = [query_graph(graph, q) for _ in range(5)]
        chat_results[q] = {
            "answers": answers,
            "stability": len(set(answers)) / 5,  # lower = more stable
            "sample": answers[0],
        }

    # Print results
    print(f"Factual: {factual_correct}/50 ({factual_correct/50*100:.0f}%)")
    print(f"Inference: {inference_correct}/30 ({inference_correct/30*100:.0f}%)")
    print(f"Memory: {get_graph_memory_mb(graph):.1f}MB")
    for q, r in chat_results.items():
        print(f'Q: "{q}" → "{r["sample"]}" (stability: {r["stability"]:.0f}%)')
```
