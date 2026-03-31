# Data Structures

## ConceptNode — the atomic unit

Everything is a ConceptNode. One structure. Multiple roles.

```python
class ConceptNode:
    id: str                          # unique identifier
    vector: Tensor[512]              # position in concept space
    name: str | None                 # human-readable label (if known)
    modality: set[str]               # {text, image, sound, abstract, ...}
    confidence: float                # how well-established (increases with evidence)
    activation: float                # how "lit up" right now (0.0 to 1.0)
    created_at: int                  # step when first stored
    last_accessed: int               # step when last activated

    # Typed edges to other concepts
    edges: list[TypedEdge]

    # Cluster membership (emergent, not assigned)
    cluster_id: str | None           # which cluster this belongs to

    # Multi-modal vectors (enriched over time)
    modality_vectors: dict[str, Tensor]  # {"text": [...], "image": [...], ...}
    # Main vector = weighted average of modality vectors
```

## TypedEdge — relationships are first-class

```python
class TypedEdge:
    source: str                      # source concept id
    target: str                      # target concept id
    relation: str                    # "color", "is_a", "action", "above", "then", ...
    strength: float                  # 0.0 to 1.0
    evidence: int                    # how many times observed
    created_at: int
    last_used: int

    # Relations are NOT pre-defined — they emerge from data.
    # Common ones will include:
    #   is_a, has_property, color, size, action, location,
    #   part_of, contains, causes, before, after, above, below,
    #   similar_to, opposite_of, ...
    # But new types can be created at any time.
```

## MetaEdge — edges about edges (patterns about patterns)

```python
class MetaEdge:
    source_edge: (str, str, str)     # (concept_a, concept_b, relation)
    target_edge: (str, str, str)     # (concept_c, concept_d, relation)
    meta_relation: str               # "analogous_to", "generalizes", "contradicts"

    # Example:
    #   (dog, bark, action) analogous_to (cat, meow, action)
    #   Both are: animal → sound → action
    #   This IS the discovery of the pattern "animals make sounds"
```

## ConceptGraph — the unified substrate

```python
class ConceptGraph:
    nodes: dict[str, ConceptNode]

    # Operations (not modules):
    def write(input, modality) -> ConceptNode     # one-shot store
    def activate(query_vector) -> list[ConceptNode]  # spread activation
    def infer(start, relation_chain) -> ConceptNode  # follow edges
    def compose(concepts, relations) -> ConceptNode  # create new
    def wander(steps) -> list[ConceptNode]          # random walk
    def generate(activated_nodes) -> str | Image     # produce output
    def cluster() -> dict[str, list[ConceptNode]]   # emergent groups
    def merge(other_graph, shared_concepts) -> None  # transplant
```

## Why ONE structure, not modules

```
The graph IS the store IS the hierarchy IS the inference engine IS the generator.

"Storing" = adding a node + edges
"Relating" = adding a typed edge
"Clustering" = computing which nodes are near each other
"Selecting" = setting activation levels based on query
"Composing" = creating new node from existing vectors
"Wandering" = following random edges
"Generating" = reading activated nodes' labels/edges
"Parsing" = converting raw input into node-creation operations

Eight operations. One data structure. All superimposed.
```

## Memory budget

```
5,000 concepts × (512 floats × 4 bytes + metadata) ≈ 15MB
50,000 typed edges × 64 bytes ≈ 3MB
Meta-edges, clusters, etc. ≈ 2MB
Total: ~20MB

vs current BrainV2: ~2GB for 10K neurons + 240K untyped edges

100x more memory efficient AND more capable.
```
