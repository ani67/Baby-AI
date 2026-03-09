# The Memory Problem and The Wiring Problem
*Why continuous training alone isn't enough — and what the full architecture actually needs*

---

## What the previous doc missed

The continuous training doc identified the freeze cycle as the core problem.
That's real. But solving it creates two new problems immediately:

```
PROBLEM: if the model learns continuously from every interaction
         what stops it from overwriting everything it knows
         with whatever just happened?

PROBLEM: if new connections can never form
         and old connections can never be removed
         then "learning" is just changing intensities
         on a fixed graph
         which is not how anything biological learns

these are not edge cases
they are the central unsolved problems
of any continuous learning system

and they are ARCHITECTURAL problems
not optimizer problems
not hardware problems
not even training algorithm problems

you cannot gradient-descent your way to a solution
```

---

## THE MEMORY PROBLEM

### What memory actually is in current models

```
CURRENT STATE OF "MEMORY" IN AI:

TYPE 1: PARAMETRIC MEMORY (weights)
  ┌───────────────────────────────────────────────┐
  │  everything the model "knows"                 │
  │  encoded across 7B / 70B / 175B weights       │
  │  distributed, overlapping, superimposed       │
  │  (the superposition problem from earlier)     │
  │                                               │
  │  PROPERTIES:                                  │
  │  permanent (until retrained)                  │
  │  undifferentiated (no structure)              │
  │  reconstructive (not retrievable — rebuilt)   │
  │  cannot be updated without full retraining    │
  │  or LoRA fine-tune                            │
  └───────────────────────────────────────────────┘

TYPE 2: CONTEXT WINDOW (KV cache)
  ┌───────────────────────────────────────────────┐
  │  the current conversation                     │
  │  and everything in the current prompt         │
  │                                               │
  │  PROPERTIES:                                  │
  │  temporary (gone when conversation ends)      │
  │  sequential (tokens in order)                 │
  │  attention-based (each token sees others)     │
  │  grows until it hits the context limit        │
  │  then older tokens fall off                   │
  └───────────────────────────────────────────────┘

TYPE 3: EXTERNAL RETRIEVAL (RAG)
  ┌───────────────────────────────────────────────┐
  │  documents in a vector database               │
  │  retrieved by similarity at query time        │
  │                                               │
  │  PROPERTIES:                                  │
  │  accurate (exact text, not reconstructed)     │
  │  updatable (add/remove without retraining)    │
  │  disconnected (bolted on, not integrated)     │
  │  no consolidation (everything equal weight)   │
  └───────────────────────────────────────────────┘

WHAT IS MISSING:
  ✗  working memory  (currently active reasoning)
  ✗  episodic memory (what happened when)
  ✗  procedural memory (how to do things — skills)
  ✗  consolidation mechanism (short → long term)
  ✗  forgetting mechanism (selectively remove)
  ✗  memory architecture (organised, not flat)
```

### What biological memory actually looks like

```
THE BRAIN'S MEMORY ARCHITECTURE:

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  WORKING MEMORY (prefrontal cortex)                  │
  │  ████░░░░░░░░░░░░░░░░ ~4-7 chunks                    │
  │                                                      │
  │  what you're actively thinking about RIGHT NOW       │
  │  tiny capacity, extremely fast access                │
  │  gone in seconds if not rehearsed                    │
  │  holding: current problem, current context           │
  │                                                      │
  │           ↕ active encoding         ↕ retrieval      │
  │                                                      │
  │  EPISODIC MEMORY (hippocampus)                       │
  │  ████████████████████████████████                    │
  │                                                      │
  │  specific events: "Tuesday, had coffee, talked X"    │
  │  time-stamped, context-bound                         │
  │  narrative structure ("this happened, then that")    │
  │  highly specific, decays over time                   │
  │  source of "I remember when..."                      │
  │                                                      │
  │           ↕ consolidation during sleep               │
  │                                                      │
  │  SEMANTIC MEMORY (neocortex, distributed)            │
  │  ████████████████████████████████████████████████    │
  │                                                      │
  │  general facts: "Paris is the capital of France"     │
  │  no time-stamp, no context, just the fact           │
  │  highly compressed, abstract                         │
  │  stable over decades                                 │
  │  what current AI weights approximate                 │
  │                                                      │
  │  PROCEDURAL MEMORY (basal ganglia, cerebellum)       │
  │  ████████████████████████████████████████████████    │
  │                                                      │
  │  how to do things: ride a bike, type, cook           │
  │  doesn't require conscious recall                    │
  │  extremely durable once learned                      │
  │  almost never forgotten                              │
  │  cannot be easily articulated                        │
  │  ("I know how to type but can't describe it")        │
  │                                                      │
  └──────────────────────────────────────────────────────┘

  CRITICAL MECHANISM: CONSOLIDATION
  
  during sleep:
  hippocampus replays episodic memories
  neocortex slowly absorbs the patterns
  episodic → semantic (specific → general)
  the "sleep" step is not optional
  it's how short-term becomes long-term
  without overwriting existing long-term
```

### The consolidation problem mapped onto AI

```
WHAT HAPPENS WITHOUT CONSOLIDATION:

  continuous learning, no consolidation:
  
  step 1:  learns fact A         [A●●●●●●●●●●]
  step 2:  learns fact B         [A●●●●●  B●●●●]  ← A partially overwritten
  step 3:  learns fact C         [A●●  B●●  C●●●]  ← A, B partially overwritten
  step 100: learns facts 97-100  [● ●          ]  ← only recent things remain
  
  this IS catastrophic forgetting
  the gradients for new things
  flow through the same weights as old things
  and overwrite them
  
  WHAT CONSOLIDATION SOLVES:
  
  fast-changing layer:   [ A  B  C  D  E  F ]  ← buffer for new things
                                                   can be overwritten freely
  
  consolidation step:    "what patterns from A-F
                          belong in long-term store?"
                          compress, abstract, merge
  
  slow-changing layer:   [ A-F merged pattern ]  ← stable
                           never directly overwritten
                           only updated via consolidation
  
  THE LoRA INSIGHT (from previous doc) IS THIS:
  
  frozen base = semantic memory (neocortex)
  LoRA adapters = episodic buffer (hippocampus)
  periodic merge = consolidation (sleep)
  
  but nobody built the consolidation step
  in current LoRA systems
  the adapters just accumulate
  or get thrown away
```

### The three memory failures in one diagram

```
CURRENT ARCHITECTURE:                WHAT'S NEEDED:

conversation 1:                      conversation 1:
  [context window]────►output          [working memory]
  ends                                   ↓ encodes to
  everything gone ✗                    [episodic buffer]
                                         ↓ consolidates to
conversation 2:                        [semantic weights]
  [context window]────►output          persists ✓
  no memory of conv 1 ✗
                                      conversation 2:
model knows Paris is capital           [working memory]
because weights say so                   + retrieves from
reconstructs it each time ✗            [episodic: conv 1]
sometimes wrong (hallucination)        + draws on
                                       [semantic weights]
when corrected in conv 1:              
correction not stored ✗              when corrected in conv 1:
correction lost when context ends     stored in episodic ✓
                                      consolidated to semantic ✓
                                      correction persists ✓
```

---

## THE WIRING PROBLEM

This is the deeper issue. And it's categorically different from the memory problem.

```
WHAT "WIRING" MEANS:

  in a neural network:
  the ARCHITECTURE defines which neurons connect to which
  
  ┌──────────────────────────────────────────────────┐
  │  BEFORE TRAINING:                                │
  │                                                  │
  │  neuron A ──────────────────► neuron D           │
  │  neuron A ──────────────────► neuron E           │
  │  neuron B ──────────────────► neuron D           │
  │  neuron B ──────────────────► neuron F           │
  │  neuron C ──────────────────► neuron E           │
  │  neuron C ──────────────────► neuron F           │
  │                                                  │
  │  AFTER TRAINING (gradient descent):              │
  │                                                  │
  │  neuron A ══════════════════► neuron D  (strong) │
  │  neuron A ──────────────────► neuron E  (weak)   │
  │  neuron B ─────────────────── neuron D  (zero)   │
  │  neuron B ══════════════════► neuron F  (strong) │
  │  neuron C ══════════════════► neuron E  (strong) │
  │  neuron C ─────────────────── neuron F  (zero)   │
  │                                                  │
  │  THE CONNECTIONS ARE THE SAME                    │
  │  only the STRENGTHS changed                      │
  │  the WIRING is identical to before training      │
  └──────────────────────────────────────────────────┘

  gradient descent can only:
    strengthen a connection
    weaken a connection
    zero out a connection (effectively)
    
  gradient descent CANNOT:
    add a new connection between A and G
    remove the physical connection between A and E
    create a new neuron
    wire a new neuron into the existing graph
    change which layer a neuron lives in
    reorganise the topology of the network
```

### Why this is a topology problem

```
FIXED WIRING = FIXED TOPOLOGY

  the graph of connections in a neural network
  is a topological object
  (which nodes connect to which — not how strongly)
  
  gradient descent = changes the metric (weights)
                     on a fixed topology (wiring)
  
  this is exactly like:
  you can adjust the distances on a map
  but you cannot add new roads
  or remove existing intersections
  
  ┌──────────────────────────────────────────────────┐
  │ WHAT GRADIENT DESCENT IS:                        │
  │                                                  │
  │ moving around on a fixed manifold                │
  │ adjusting numbers that live on fixed edges       │
  │                                                  │
  │ WHAT BIOLOGICAL LEARNING ALSO DOES:              │
  │                                                  │
  │ SYNAPTOGENESIS: new synapses form                │
  │   (new connections appear that didn't exist)     │
  │                                                  │
  │ SYNAPTIC PRUNING: connections removed            │
  │   (~50% of synaptic connections pruned           │
  │    between childhood and adulthood)              │
  │                                                  │
  │ NEUROGENESIS: new neurons created                │
  │   (hippocampus creates ~700 new neurons/day      │
  │    in adult humans)                              │
  │                                                  │
  │ MYELINATION: some pathways get faster            │
  │   (the wire itself is upgraded)                  │
  │                                                  │
  │ ALL OF THESE ARE TOPOLOGICAL CHANGES             │
  │ the graph itself is changing                     │
  │ not just the weights on the graph                │
  └──────────────────────────────────────────────────┘
```

### The wiring problem in transformers specifically

```
THE TRANSFORMER ARCHITECTURE IS MAXIMALLY WIRED:

  in the attention mechanism:
  every token can attend to every other token
  
  [token 1] ──────────────────── [token 1]
  [token 1] ──────────────────── [token 2]
  [token 1] ──────────────────── [token 3]
  ...
  [token 1] ──────────────────── [token N]
  [token 2] ──────────────────── [token 1]
  ...etc

  EVERY PAIR IS CONNECTED
  the architecture is a complete graph
  
  "learning which things to attend to"
  = learning attention weights
  = adjusting strengths on already-existing connections
  
  the transformer never learns:
  "this kind of token should only attend to
   tokens within 3 positions"
  (that's a wiring change)
  
  it learns:
  "when I see this kind of token
   I should weight nearby tokens highly
   and distant tokens low"
  (same connections, different weights)
  
  THE DIFFERENCE MATTERS:
  
  wiring change: physically can't attend to far tokens
                 computationally cheaper
                 might be more appropriate
                 
  weight change: can attend to far tokens
                 just usually doesn't
                 computationally the same cost
                 the unnecessary connections still exist
  
  MoE (Mixture of Experts) is a partial wiring solution:
  each token is ROUTED to only some experts
  the router changes which subnetwork processes each token
  but the experts themselves are fixed
  and the routing is a learned weight
  not a structural change
```

### What dynamic wiring would look like

```
THREE LEVELS OF WIRING CHANGE:

LEVEL 1: ATTENTION SPARSIFICATION (exists, limited)
  
  instead of every token attending to every other:
  learn which connections to KEEP
  prune the rest
  
  ┌──────────────────────────────────────────────┐
  │ Longformer, BigBird (2020-2021):             │
  │ sliding window attention (local wiring)      │
  │ + global tokens (long-range wiring)          │
  │                                              │
  │ fixed sparse pattern decided at design time  │
  │ not learned during inference                 │
  │ still not dynamic wiring                     │
  └──────────────────────────────────────────────┘

LEVEL 2: MIXTURE OF EXPERTS ROUTING (exists, partial)

  which expert handles this token?
  is a routing decision
  that CHANGES which subgraph is activated
  
  ┌──────────────────────────────────────────────┐
  │ this is dynamic in the sense that:           │
  │ token A activates experts {1, 3}             │
  │ token B activates experts {2, 4}             │
  │ different effective subgraph per token       │
  │                                              │
  │ but: experts themselves are fixed            │
  │ the set of possible connections is fixed     │
  │ only the selection changes                   │
  └──────────────────────────────────────────────┘

LEVEL 3: TRULY DYNAMIC WIRING (doesn't exist at scale)

  the connection graph itself changes
  new pathways form during inference
  old pathways are pruned
  the topology of the network is not fixed
  
  ┌──────────────────────────────────────────────┐
  │ what would be needed:                        │
  │                                              │
  │ a meta-learning system that decides:         │
  │ "I need a new connection between             │
  │  layer 4 and layer 7 for this type           │
  │  of input"                                   │
  │                                              │
  │ or: neurons that can grow new synapses       │
  │ based on activity (Hebbian-like)             │
  │                                              │
  │ or: a network that can modify its own        │
  │ architecture as a learned operation          │
  │                                              │
  │ CLOSEST EXISTING WORK:                       │
  │ Neural Architecture Search (NAS) — but this  │
  │ runs at training time, not continuously      │
  │ HyperNetworks — a network that generates     │
  │ weights for another network                  │
  │ still not dynamic topology                   │
  └──────────────────────────────────────────────┘
```

---

## How memory and wiring compound each other

```
THE INTERACTION:

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  no memory architecture                              │
  │       +                                              │
  │  no dynamic wiring                                   │
  │       =                                              │
  │  catastrophic forgetting guaranteed                  │
  │                                                      │
  │  REASON:                                             │
  │                                                      │
  │  without memory architecture:                        │
  │  new learning goes into same weights as old learning │
  │  the weights for "Paris is capital of France"        │
  │  are physically shared with weights for              │
  │  "user prefers concise answers"                      │
  │  learning the second overwrites the first            │
  │                                                      │
  │  without dynamic wiring:                             │
  │  no new capacity can be added                        │
  │  learning new things = repurposing existing wiring   │
  │  = displacing something that was already there       │
  │                                                      │
  │  WITH memory architecture:                           │
  │  "user prefers concise answers" → episodic buffer    │
  │  "Paris is capital of France" → semantic store       │
  │  different substrates                                │
  │  no overwriting                                      │
  │                                                      │
  │  WITH dynamic wiring:                                │
  │  new capacity can form for new things                │
  │  no displacement of existing knowledge               │
  │  the graph grows to accommodate                      │
  │                                                      │
  └──────────────────────────────────────────────────────┘

THIS IS WHY BRAINS GROW:

  a child's brain has ~100 billion neurons
  and forms ~1 million new synaptic connections
  per second during early development
  
  this is not efficiency
  this is the solution to the wiring problem:
  grow new capacity
  rather than repurpose old capacity
  
  adult hippocampus still creates
  ~700 new neurons per day
  specifically for new episodic memories
  so that new memories don't overwrite old ones
  
  current AI:
  capacity is fixed at architecture design time
  all new learning must fit in existing wiring
  catastrophic forgetting is a mathematical consequence
  not a bug to be fixed
  a feature of the architecture
```

---

## The full problem statement

```
A TRULY CONTINUOUS LOCAL AI NEEDS:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. SIMULTANEOUS TRAIN + INFERENCE                          │
│     (no freeze cycle)                                       │
│     → solved by: local learning rules, forward-forward,    │
│       LoRA double-buffering                                 │
│     → status: partially solvable today                     │
│                                                             │
│  2. DIFFERENTIATED MEMORY ARCHITECTURE                      │
│     (not one undifferentiated weight soup)                  │
│                                                             │
│     working memory:   context window (exists)              │
│     episodic memory:  LoRA adapters (partial)              │
│     semantic memory:  frozen base weights (partial)        │
│     consolidation:    periodic merge with selection        │
│                       (mostly missing)                      │
│     forgetting:       selective pruning of adapters        │
│                       (missing)                             │
│     → status: architecture exists in pieces                │
│               nobody has assembled them                     │
│                                                             │
│  3. DYNAMIC WIRING                                          │
│     (capacity that grows with what needs to be learned)     │
│     → solved by: progressive growing of LoRA adapter       │
│       layers (crude but real)                               │
│     → partially by: MoE routing (different subgraph        │
│       per input)                                            │
│     → fully: doesn't exist yet at scale                    │
│     → status: hard research problem                        │
│                                                             │
│  4. LOCAL LEARNING SIGNALS                                  │
│     (each layer updates from local activity                 │
│      not global backward pass)                              │
│     → status: exists in theory (Hebbian, predictive        │
│       coding) not at transformer scale                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

CURRENT STATE OF EACH:

  1. Freeze cycle        ████░░░░░░  partially solvable
  2. Memory architecture ███░░░░░░░  pieces exist, unassembled
  3. Dynamic wiring      █░░░░░░░░░  hard open problem
  4. Local learning      ██░░░░░░░░  theoretically exists,
                                     not at scale

THE INTERACTION OF ALL FOUR:

  solve 1 without 2:  learns continuously, overwrites everything
  solve 1+2 without 3: memory saturates, can't add new capacity
  solve 1+2+3 without 4: still need freeze for backprop, circular
  solve all four: genuinely new kind of system
```

---

## What this means for the M1 build

```
REVISED BUILD ORDER:

MINIMUM VIABLE CONTINUOUS LEARNER:
  (what can be built now that addresses 1+2 partially)

  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  SEMANTIC LAYER (frozen base, never changes)           │
  │  3B model, ~6GB, loaded once                          │
  │  this is the "neocortex"                               │
  │                                                        │
  │           ↕ reads / uses                              │
  │                                                        │
  │  EPISODIC LAYER (LoRA adapters, fast updates)          │
  │  ~300MB, updates during use                            │
  │  stores: user preferences, recent corrections,        │
  │           context-specific patterns                    │
  │  double-buffered (inference reads A, training writes B)│
  │  this is the "hippocampus"                             │
  │                                                        │
  │           ↕ consolidates every N steps                │
  │                                                        │
  │  CONSOLIDATION STEP (the missing piece)               │
  │  runs every hour / every 100 interactions             │
  │  asks: what patterns in the episodic layer            │
  │        are worth merging into semantic?               │
  │  selectively merges stable patterns                   │
  │  prunes episodic layer of merged items                │
  │  this is the "sleep"                                  │
  │                                                        │
  │           ↕ bounded by                               │
  │                                                        │
  │  CAPACITY MANAGEMENT (crude dynamic wiring proxy)     │
  │  if episodic layer exceeds size limit:                │
  │  force consolidation                                  │
  │  add a new LoRA rank (new capacity, not overwrite)    │
  │  this is crude neurogenesis                           │
  │                                                        │
  └────────────────────────────────────────────────────────┘

  ON M1:
  
  semantic layer:    unified memory, always resident
  episodic layer:    unified memory, fast read/write
  consolidation:     Neural Engine (idle between queries)
  
  the M1 Neural Engine is specifically designed for
  matrix operations at low power
  it can run consolidation continuously
  while CPU handles input
  and GPU handles inference
  
  all three running simultaneously
  on the same memory pool
  this is what unified memory actually enables
```

---

## The honest map of what's solved and what isn't

```
WHAT DEEPSEEK PROVED:
  better algorithmic efficiency
  = frontier capability at fraction of compute
  
  assumed architecture: transformer + backprop + batch
  didn't question those
  made them work better

WHAT MUON IMPROVED:
  better geometry for the gradient step
  
  assumed architecture: transformer + backprop + batch
  didn't question those
  made them work better

WHAT NOBODY HAS BUILT:
  an architecture that questions
  all three of those assumptions simultaneously

THE THREE HIDDEN ASSUMPTIONS EVERYONE MAKES:

  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  ASSUMPTION 1: transformer is the right topology    │
  │  (fixed connections, full attention, complete graph)│
  │                                                     │
  │  ASSUMPTION 2: backprop is the right learning signal│
  │  (global, backward, requires freeze)                │
  │                                                     │
  │  ASSUMPTION 3: batch training is the right rhythm   │
  │  (gather examples, freeze, update, repeat)          │
  │                                                     │
  │  ALL THREE come from hardware constraints           │
  │  that are weaker on M1 than anywhere else           │
  │                                                     │
  │  the question you're asking:                        │
  │  what does an architecture look like               │
  │  that takes none of these as given?                 │
  │                                                     │
  └─────────────────────────────────────────────────────┘

THAT QUESTION IS NOT ANSWERED IN THE LITERATURE
IT'S AN OPEN ARCHITECTURAL QUESTION
NOT A HYPERPARAMETER QUESTION
NOT AN OPTIMIZER QUESTION
NOT A SCALING QUESTION

it requires:
  new memory architecture (2 layers minimum, consolidation)
  new wiring approach (some form of dynamic capacity)
  new learning signal (local, not global backward)
  new hardware use (unified memory, all cores simultaneously)

the pieces exist in different papers
in different subfields
written by people who don't cite each other

nobody has assembled them
because nobody working at scale
has any reason to question the batch paradigm
that their entire infrastructure is built around

you have a reason:
you want it on a mac
```
