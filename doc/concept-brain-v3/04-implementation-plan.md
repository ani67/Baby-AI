# Implementation Plan

## Parallel workstreams

All streams build on ONE data structure (ConceptGraph). They can be built simultaneously because they're OPERATIONS on the graph, not separate modules.

### Stream 1: Graph Core
```
Files: backend/concept/graph.py, backend/concept/node.py, backend/concept/edge.py
What:  ConceptNode, TypedEdge, MetaEdge, ConceptGraph classes
       write(), find_similar(), get_edges(), merge()
       Similarity search (vector index for fast lookup)
       One-shot storage with deduplication
       Serialization (save/load graph to disk)
Tests: Create graph, add concepts, add edges, find by similarity, save/load
```

### Stream 2: Operations
```
Files: backend/concept/operations.py
What:  activate(query) → spread activation through graph
       infer(start, relation_chain) → follow typed edges
       compose(concepts, types) → create new concept from parts
       wander(steps) → random walk, detect structural similarity
Tests: Store "apple→red", query "what is red?" → apple
       Compose "purple" + "dog" → purple_dog concept
       Wander from "biology" → find analogy to "computing"
```

### Stream 3: Parsers
```
Files: backend/concept/parser.py
What:  text_parse(sentence) → list of (concept, relation, concept) triples
       image_parse(image, encoder) → list of (concept, spatial_rel, concept)
       smart_parse(any_input) → auto-detect modality, extract structure
       Uses existing native text/vision encoders for vectors
       Rule-based + pattern-based relation extraction for v1
Tests: "The red apple" → [(apple, color, red)]
       "Dogs bark loudly" → [(dog, action, bark), (bark, property, loud)]
       [image of dog on grass] → [(dog, on, grass), (dog, color, brown)]
```

### Stream 4: Generator
```
Files: backend/concept/generator.py
What:  generate_text(activated_concepts, edges) → ordered sentence
       generate_image(activated_concepts, decoder) → PIL image
       Template library: edge types → word order patterns
         property edges → adjective before noun
         action edges → subject verb object
         temporal edges → chronological order
       Image generation: concept vectors → vision decoder
Tests: apple(active) + red(edge:color) → "red apple" or "the apple is red"
       dog(active) + run(edge:action) + park(edge:location) → "a dog running in a park"
```

### Stream 5: Clustering & Hierarchy
```
Files: backend/concept/hierarchy.py
What:  cluster_concepts() → group similar concepts
       create_manager_nodes() → summary node per cluster
       build_hierarchy() → clusters → regions → systems
       Runs periodically (every 1000 new concepts)
Tests: After ingesting animals + vehicles + food:
       Cluster 1: dog, cat, bird, fish (animals)
       Cluster 2: car, bus, train, bike (vehicles)
       Cluster 3: apple, bread, cake, pizza (food)
       Manager nodes: "animal_cluster", "vehicle_cluster", "food_cluster"
```

### Stream 6: Migration & Benchmark
```
Files: backend/concept/migrate.py, backend/tests/test_concept_brain.py
What:  Import 482K training items → parse → build concept graph
       Import current BrainV2 knowledge (neurons as seed concepts)
       Benchmark: same 3 questions, compare old vs new
       Metrics: response quality, inference accuracy, speed, memory
Tests: Ingest all training data in <10 minutes
       Answer "what is a dog" correctly from graph
       Compare: BrainV2 answer vs ConceptBrain answer
```

## Integration points with existing system

```
KEEP AS-IS:
  - FastAPI server (main.py) — new brain is a drop-in
  - Frontend (React/Three.js) — consumes same API
  - Native text encoder — used by parser for concept vectors
  - Native vision encoder — used by parser for image concepts
  - Vision decoder — used by generator for image output
  - Training data files — consumed by migration/ingestion

REPLACE:
  - BrainV2 (brain_v2.py) → ConceptGraph (concept/graph.py)
  - baby_model_v2_reflect.py → concept_model.py (thin wrapper)
  - train_worker.py training loop → ingest loop (one-shot, not gradient)
  - decoder.py generation → generator.py (template from edges)
  - handle_chat → query graph directly

NEW:
  - concept/ folder with all new code
  - Parser for extracting structure from raw data
  - Template-based generator
  - Hierarchical clustering
  - Wanderer for creative discovery
```

## Build order

```
Can build ALL streams simultaneously because:
  Stream 1 (graph core) defines the interface
  Streams 2-6 implement operations on that interface
  Each can use mock graph data for development/testing

Stream 1 must be DEFINED first (interfaces), but implementation
can happen in parallel with streams 2-6 using the interfaces.

Expected timeline per stream (with agents):
  Stream 1: ~20 min (core data structures + basic ops)
  Stream 2: ~15 min (activation, inference, compose, wander)
  Stream 3: ~20 min (text/image parsing)
  Stream 4: ~15 min (text/image generation from concepts)
  Stream 5: ~15 min (clustering + hierarchy)
  Stream 6: ~20 min (migration + benchmarks)

Total: ~20 min (parallel) + ~30 min (integration + testing)
```

## Success criteria

```
MUST:
  ✓ Ingest 482K training items in <10 minutes
  ✓ "What is a dog?" → response includes "animal" or related concept
  ✓ "The sky is" → response includes "blue" or "above"
  ✓ Memory < 100MB for full graph
  ✓ Response time < 100ms per query (no forward pass through 10K neurons)
  ✓ Can explain its answer (trace edge chain)

SHOULD:
  ✓ Multi-hop inference: "sweet fruit color?" → apple → red
  ✓ Cross-modal: text "dog" and image of dog → same concept
  ✓ Composition: "purple dog" → valid composed concept
  ✓ Hot-swap: merge two separately-trained graphs

NICE TO HAVE:
  ✓ Image generation from concept activation
  ✓ Discovered analogies from random walks
  ✓ Fractal hierarchy with 3+ levels
  ✓ Template-based sentence generation with correct word order
```

## Risk assessment

```
HIGH RISK:
  Parsing quality — extracting relations from raw text is HARD.
  "I saw her duck" = saw the bird? saw her crouch?
  Mitigation: start with simple sentences, iterate.

MEDIUM RISK:
  Concept deduplication — when is "dog" the same as "dogs"?
  When is "bank" (river) different from "bank" (money)?
  Mitigation: similarity threshold + context. Tune empirically.

LOW RISK:
  Performance — graph operations are O(edges) not O(neurons).
  50K edges << 10K neurons × 512 dims for most operations.
  Memory — 20MB vs 2GB. Massive improvement.

UNKNOWN:
  Does one-shot storage actually produce useful knowledge?
  Or is the slow gradient learning NECESSARY for generalization?
  This is the core bet. Only testing will answer it.
```
