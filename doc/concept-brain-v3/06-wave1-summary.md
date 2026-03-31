# Wave 1 Think Results — Summary

## Completed: 5 of 6 agents

### 1. Graph Core (DONE)
- ConceptNode with EMA vector updates (alpha = 1/(1+count))
- Match threshold 0.85 cosine (configurable)
- Edges stored SEPARATELY from nodes (not on-node lists)
- Always directed, bidirectional = two edges
- Multiple edges per pair OK (different relation types)
- Brute-force matmul for similarity (<0.1ms at 5K concepts)
- SQLite for metadata + torch .pt for vectors
- Homonyms: multiple nodes with same name, differentiated by vector
- Synonyms: separate nodes with similar_to edges
- ~36MB for 5K concepts with all metadata

### 2. Parser (DONE)
- Rule-based pattern matching, ~1000 items/sec
- 15 bootstrapped relation types (color, size, property, action, is_a, spatial, etc.)
- Format-aware extraction per dataset (gsm8k, boolq, triviaqa, etc.)
- Compound word detection (static list + title case)
- ~3-4 minutes to parse all 482K items
- Parser outputs triples, does NOT call encoders (separation of concerns)

### 3. Operations (DONE)
- Spreading activation: bounded BFS with damping=0.5, threshold=0.05, top-K=20
- Typed inference: constraint intersection using reverse index
- Composition: edge-type-guided dimension blending (Option C)
- Random walks: weighted, structural analogy detection via pattern matching
- Selection: layered (activation × query_sim × intent_boost × recency)
- Context: activation decay (0.3/turn) + explicit pronoun resolution
- Full query pipeline: <5ms total

### 4. Generator (DONE)
- Edge types → grammar templates (15-20 templates)
- Tier ordering: definition → property → action → relation → temporal
- Confidence calibration: evidence=20+ → fact, evidence=1 → "possibly"
- Image generation: composed vector (base + weighted modifiers by edge type)
- Multi-sentence: one sentence per edge-type group
- GeneratorOutput includes text + confidence + source_edges for explainability

### 5. Hierarchy + Hot-swap (DONE)
- Hybrid clustering: k-means (vector) + Louvain (edges)
- Manager nodes as ConceptNodes (same type, same operations)
- Fractal: concepts → clusters → regions → systems (3-4 levels)
- Merge: ANN similarity search, confidence-weighted vector averaging, edge union
- Remove: retarget strong dangling edges, drop weak ones
- Import: auto-connect to nearest concepts at lower strength
- Layered graph: base + specializations + personal (privacy model)

### 6. Migration + Benchmark (PENDING)

## Key design decisions across all agents

1. **Edges off-node**: edges live in graph-level store, not per-node lists
2. **Reverse index mandatory**: (relation, target) → set(sources) for typed inference
3. **Brute-force similarity**: no FAISS needed at 5-10K concepts
4. **Rule-based parser v1**: fast, simple, ~90% precision on curriculum data
5. **Composition via edge types**: not vector arithmetic
6. **Manager nodes are ConceptNodes**: uniform treatment
7. **SQLite + torch .pt**: hybrid persistence
8. **Activation on nodes**: not separate buffer
9. **Context via activation decay**: natural 2-3 turn memory
10. **Templates for generation**: edge types define grammar
