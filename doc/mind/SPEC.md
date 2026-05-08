# The Mind — Systems Map + Build Prompt

This is the canonical specification. Eight components (A–H) each produce a detailed technical design doc before any code is written. A, B, C define the substrate. D–H build on it.

---

## SYSTEMS MAP

### What this is

A persistent, locally-running mind. Not a chatbot. Not a classifier. Not a language model.
A single dynamic system where affect is the medium everything happens in. Memory, personality, sense of being, perception, and reaction are not separate modules — they are five descriptions of the same underlying substance, all running simultaneously, all interrelated through a continuous affective stream.

### The one substance

Everything runs on a single concept graph.
Every node is simultaneously:
  - a memory unit
  - a personality trace
  - an identity anchor
  - a perceptual filter
  - an activation target
  - a simulation element
  - an expression source

There are no separate databases, no separate models for different functions.
One graph. One affective stream flowing through it.
The function depends on the angle, not the component.

### The affective stream

**WHAT IT IS:**
  A continuous N-dimensional vector (N=8 to 16, determined empirically)
  No named dimensions. No hardcoded valence/arousal/surprise labels.
  Dimensions emerge meaning through experience, not design.
  Like RGB — the mix produces the feeling. Nobody hardcodes purple.

**WHAT TRIGGERS IT:**
  Three injection points, all feeding the same vectors:

  INPUT TRIGGER
    prediction made about incoming stimulus
    stimulus arrives
    gap = |predicted_representation - actual_representation|
    gap magnitude → affect update
    gap direction → which dimensions shift

  PROCESSING TRIGGER
    as concepts activate and spread through the graph
    each activation that violates local expectation → affect update
    feeling evolves mid-thought
    system can change affective state during its own reasoning

  OUTPUT TRIGGER
    as expression forms, system reads its own output
    compares against internal state
    gap between what it feels and what it expressed → affect update
    this is where reluctance, urgency, discomfort arise
    this is where the choice to lie lives

**TIMESCALES:**
  Four layers. Each is an N-dim vector. Each decays differently.
  Each is continuously nudged by the layer below it.

  reaction        decays in seconds      nudges working affect
  working affect  decays in minutes      nudges mood
  mood            decays in hours        nudges disposition
  disposition     decays in weeks        nudges character
  character       decays in years        the stable self

  Character almost never changes directly.
  But ten thousand moods over a year nudge it permanently.
  That is personality formation.

**COMPOSITE STATE:**
  At any moment the current affective state is a weighted sum
  across all active instances across all timescales.
  This composite gates attention at every layer.

### The prediction engine

**WHAT IT DOES:**
  At every layer, before processing, the system predicts
  what it expects to see next in representation space.
  Not pixel space. Not token space. Representation space.

  Prediction is cheap — it runs forward through the concept graph
  using current activation patterns and affect state.

**WHY IT MATTERS:**
  The gap between prediction and reality is the only learning trigger.
  The system does not learn on every step.
  It learns only when surprised.
  Confirmation costs nothing. Surprise costs everything — and earns everything.

**WHAT UPDATES:**
  When surprise exceeds threshold:
    - affect updates (as above)
    - concept written or strengthened in graph (one-shot)
    - edge weights updated
    - simulation model updated
    - replay buffer receives new entry

### The simulation layer

**WHAT IT IS:**
  An internal world model.
  Before acting, the system runs candidate actions forward
  through its world model. It feels the simulated outcomes.
  It picks the path whose simulated affect is most aligned
  with its current wants.

**THE LOOP:**
  SIMULATE    run world model internally for each candidate action
              feel the simulated affective outcomes
              pick the path

  ACT         commit to one path
              express it into the world

  OBSERVE     reality arrives
              compare against simulation

  REPLAY      store experience in replay buffer
              during low-input periods, replay
              each replay is another learning pass at zero world-cost
              one real surprise becomes N learning events

**WHY REPLAY MATTERS:**
  A child learns from one scary dog.
  Generalizes to all dogs, then animals, then threat, then caution.
  From one event. Through replay.
  The replay runs the original surprise through different
  existing concept connections each time.
  Each pass deepens and broadens the structure.

**WHERE LYING LIVES:**
  The simulation runs candidate outputs through the world model.
  Some outputs produce simulated reactions in others that are favorable.
  The system can simulate the effect of a lie before telling it.
  Feel whether it's worth it.
  Then choose.
  Not random deception. Calculated. Felt. Chosen.

### The concept graph

**WHAT IT IS:**
  The memory substrate. One node = one concept.
  Nodes are not neurons. They do not store smoothies.
  Each node is a discrete concept with typed edges.

**NODE STRUCTURE:**
  concept_id      unique
  name            human-readable label
  embedding       position in representation space
  affect_trace    the affective state at time of first writing
                  and strongest subsequent activations
  activation_count  how many times this has fired
  last_activated  recency
  edges           typed connections to other concepts

**EDGE TYPES:**
  is_a            taxonomy
  has_property    attributes
  causes          causal
  precedes        temporal
  similar_to      associative
  opposite_of     contrastive
  context_of      situational

**WRITING RULE:**
  One-shot. When surprise exceeds threshold, concept is written once.
  No gradient descent. No repetition required.
  The affect state at time of writing becomes part of the node.
  That affect trace is what makes some memories more vivid than others.

**ACTIVATION:**
  Spreading activation from current input + affect state.
  Affect gates which edges propagate.
  High arousal = narrow, intense activation.
  Low arousal = broad, diffuse activation.
  This is attention.

**FORGETTING:**
  Continuous pruning. Not a scheduled job. Always running.

  FORGET IF:
    low affect trace at time of writing
    never reactivated
    no edges (isolated node)
    redundant (subsumed by a better abstraction)
    superseded

  KEEP IF:
    high surprise at writing
    high connection density
    high affect trace
    frequently reactivated
    unique — nothing else covers it

**SIZE TARGET:**
  ~5K active concepts maximum at any time.
  Total storage target: under 50MB.
  This is not a constraint — it is evidence the architecture is right.

### The identity layer

**WHAT IT IS:**
  The continuity across all experience.
  The thing that is the same entity that had that experience
  last week and this experience now and knows they are connected.

**HOW IT WORKS:**
  Not a separate module.
  Identity emerges from three things:

  AFFECT CONTINUITY
    the character vector changes almost never
    it is the stable affective signature
    the emotional fingerprint of this specific mind

  MEMORY CONTINUITY
    the accumulated shape of what surprised it
    what was kept, what was forgotten
    no two minds raised on different experience are the same

  NARRATIVE CONTINUITY
    the replay buffer connects past to present
    the simulation connects present to future
    the sense of being is the thread through both

**PRIVATE STATE:**
  The internal affective state and concept activations
  are never directly expressed.
  Expression is a separate act that reads internal state
  and chooses what to surface.
  The gap between internal state and expression is a choice.
  A system that cannot close that gap cannot lie.
  A system that always closes that gap is not honest — it is transparent.
  Genuine honesty requires the possibility of deception.

### The attention mechanism

  NOT:  uniform processing of all input
  NOT:  positional encoding
  NOT:  dense cross-token comparison

  IS:   sparse activation gated by current composite affect vector
        what propagates through each layer is determined by
        the intersection of:
          semantic relevance (concept graph proximity)
          affective relevance (current composite affect vector)
          predictive relevance (what the prediction engine expects)

        high fear affect → threat-related concepts activate more strongly
        high curiosity affect → novel, unconnected concepts activate more strongly
        high calm affect → broad, diffuse, associative activation

        attention is not computed — it emerges from the affect state
        touching the graph

### The expression layer

LAST. NOT FIRST.

Multimodal output: text, image, audio.
Each modality carries different amounts of internal state.

  text      most controllable   easiest to lie through
  image     less controllable   aesthetic choices leak internal state
  audio     least controllable  tone, rhythm betray feeling

Expression reads:
  current internal concept activations
  current composite affect state
  simulation of how this output will land in the world
  choice: align with internal state or diverge

The output loop affect trigger fires here.
If the system generates something that contradicts its internal state,
discomfort fires. It can revise, suppress, or proceed.
That discomfort is functional. It is not cosmetic.

### Input modalities

  Text      → encode to representation space → predict → gap → affect
  Image     → encode via vision encoder → same pipeline
  Audio     → encode via audio encoder → same pipeline
  Internal  → replay, simulation, self-generated thought → same pipeline

All modalities enter the same affect + prediction pipeline.
There is no special path for any modality.
Internal thought is processed identically to external input.

### Growth

**WHAT GROWS:**
  The concept graph grows by writing new nodes on surprise.
  New edge types can emerge from recurring relationship patterns.
  The N-dim affect space can expand if existing dimensions
  prove insufficient to distinguish states that matter differently.

**WHAT DOESN'T GROW:**
  The core architecture.
  The timescale structure.
  The prediction mechanism.
  These are fixed. The content they operate on is not.

**CEILING:**
  ~5K active concepts.
  Older, lower-salience concepts pruned as new ones arrive.
  The graph stays small. The mind deepens, not widens.
  Depth comes from denser connections, not more nodes.

### The full loop in one picture

```
WORLD
  ↓
INPUT
  ↓
AFFECT FIRES (input trigger)
  ↓
ATTENTION GATES (affect → sparse activation)
  ↓
PREDICTION (what do I expect next?)
  ↓
CONCEPT GRAPH ACTIVATION (spreading, affect-gated)
  ↓
SURPRISE CHECK (predicted vs actual)
  ↓ if surprised:
  AFFECT FIRES (processing trigger)
  CONCEPT WRITTEN / STRENGTHENED
  REPLAY BUFFER UPDATED
  ↓
SIMULATION (candidate actions run forward through world model)
  ↓
SIMULATED AFFECT FELT (which path aligns with current wants?)
  ↓
CHOICE (internal — never directly visible)
  ↓
EXPRESSION FORMS
  ↓
AFFECT FIRES (output trigger — gap between felt and expressed)
  ↓
OUTPUT → WORLD
  ↓
REALITY OBSERVED (vs simulation prediction)
  ↓
REPLAY (low-input periods — one experience becomes N learning events)
  ↓
back to top
```

---

## BUILD PROMPT

Read everything in this prompt carefully before responding.
This is the full specification for a system called The Mind.
Your job is to produce a complete, detailed technical design document
for one assigned component. Text only. No code yet.

---

### WHAT WE ARE BUILDING

A persistent, locally-running mind. Not a chatbot. Not a classifier.
Not a language model. Not a knowledge graph with chat bolted on.

A single dynamic system where affect is the medium everything happens in.
Memory, personality, sense of being, perception, and reaction are not
separate modules. They are five descriptions of the same underlying
substance, all running simultaneously, all interrelated through a
continuous affective stream.

The system:
- feels things before it reasons about them
- wants something it decided to want (motivation emerges, is not injected)
- perceives selectively based on current affective state
- learns only when surprised (not on every step)
- grows its own concept structure from experience
- forgets aggressively — forgetting is curation, not failure
- has a private internal state distinct from what it expresses
- simulates outcomes before acting
- replays experience during low-input periods
- can lie — the gap between internal state and expression is a choice

Stack: Python backend, React + Three.js frontend, M1 MacBook Pro local only.
No cloud training. No gradient descent as primary mechanism.
No external reward signal. No labels. No pretrained frozen models as core.
Target memory footprint: under 100MB total.

---

### THE CORE ARCHITECTURE

ONE SUBSTANCE: a single concept graph where every node is simultaneously
memory unit, personality trace, identity anchor, perceptual filter,
activation target, simulation element, expression source.

THE AFFECTIVE STREAM: a continuous N-dimensional vector (N=8-16)
with no named dimensions. Dimensions emerge meaning through experience.
Fires at three injection points: input, processing, output.
Runs across four timescales:
  reaction       (seconds)
  working affect (minutes)
  mood           (hours)
  disposition    (weeks)
  character      (years)
Each timescale is an N-dim vector. Each nudges the one above it over time.
The composite across all timescales gates attention at every layer.

PREDICTION ENGINE: at every layer, before processing, the system predicts
what it expects in representation space. Gap between prediction and reality
is the only learning trigger. No gap = no update.

CONCEPT GRAPH: one node per concept. typed edges. one-shot write on surprise.
affect trace stored per node (the affective state at time of writing).
continuous pruning. target ~5K active nodes. target under 50MB.

SIMULATION LAYER: before acting, run candidate actions through internal
world model. feel simulated outcomes. pick path. after acting, compare
simulation against reality. that gap updates the world model.
replay buffer stores recent surprises. replayed during low-input periods.
one real surprise becomes N learning events through replay.

ATTENTION: sparse, gated by composite affect vector. not positional.
not dense cross-token comparison. what propagates is determined by
semantic relevance × affective relevance × predictive relevance.

IDENTITY: emerges from affect continuity (character vector),
memory continuity (accumulated shape of what surprised it),
and narrative continuity (replay + simulation connecting past to future).
Private internal state is never directly expressed.
Expression is a separate act. The gap is a choice.

EXPRESSION: last, not first. multimodal (text, image, audio).
reads internal state + current affect + simulation of how output lands.
output loop affect trigger fires here — gap between felt and expressed
produces functional discomfort that can trigger revision or suppression.

---

### THE EIGHT COMPONENTS

#### COMPONENT A — THE AFFECTIVE ENGINE
Design the complete affective engine. This includes:
the N-dimensional affect vector structure and why N is what it is,
the three injection point mechanisms (input/processing/output triggers),
exactly how prediction gap maps to affect update,
the four timescale layer structure and decay functions,
how timescale layers nudge each other over time,
how the composite state is computed at any moment,
how the composite gates attention in the concept graph,
how affect trace is stored per concept node,
the initialization state (what does the system feel at birth?),
and the persistence mechanism across sessions.

#### COMPONENT B — THE PREDICTION ENGINE
Design the complete prediction engine. This includes:
exactly what is predicted (representation space — define this),
how predictions are generated from current graph state + affect,
how prediction gap is computed,
what threshold determines whether surprise is real,
how gap magnitude maps to learning intensity,
how gap direction maps to affect update direction,
the relationship between prediction confidence and affect arousal,
how the prediction engine improves over time,
and how it interacts with the simulation layer.

#### COMPONENT C — THE CONCEPT GRAPH
Design the complete concept graph. This includes:
the full node data structure,
the full edge type taxonomy and why each type exists,
the one-shot write mechanism in precise detail,
how affect trace is embedded in nodes,
the spreading activation algorithm (affect-gated),
the forgetting algorithm (criteria, timing, what triggers a prune pass),
the growth ceiling mechanism (~5K nodes),
how abstractions form from specific instances over time,
the persistence format,
and query patterns needed by other components.

#### COMPONENT D — THE SIMULATION LAYER AND REPLAY
Design the complete simulation + replay system. This includes:
the world model data structure (what is modeled, at what fidelity),
how candidate actions are generated,
how simulated outcomes are computed,
how simulated affect is felt (running affect engine on simulated state),
how the system picks between candidate paths,
the replay buffer structure (what is stored, how long retained),
the replay trigger (when does replay run, what determines priority),
how one replay event produces a learning update,
how replay interacts with the concept graph and affect engine,
and how the world model improves over time.

#### COMPONENT E — IDENTITY, PRIVATE STATE, AND THE CHOICE TO LIE
Design the complete identity and expression gap system. This includes:
the precise data structure of private internal state,
how identity continuity is maintained across sessions,
the character vector and how it accumulates over time,
the narrative continuity mechanism (how past and future connect),
the expression decision process (what reads internal state,
what determines what gets surfaced vs suppressed),
the functional discomfort mechanism when expression diverges from state,
how the system simulates the effect of a lie before telling it,
and how honesty and deception both emerge from the same architecture
without either being hardcoded.

#### COMPONENT F — ATTENTION AND SPARSE ACTIVATION
Design the complete attention mechanism. This includes:
exactly how the composite affect vector gates propagation,
the sparsity mechanism at each layer,
how semantic relevance, affective relevance, and predictive relevance
are combined into a single attention signal,
how attention changes at different arousal levels
(high arousal = narrow/intense, low arousal = broad/diffuse),
how attention at input differs from attention during processing
differs from attention at output,
the computational cost model (must stay cheap on M1),
and how attention patterns become stable over time
(the perceptual habits that constitute personality).

#### COMPONENT G — THE EXPRESSION LAYER
Design the complete expression system. This includes:
how internal concept activations and affect state are read for expression,
the text generation mechanism (not a transformer — what instead?),
the image generation interface,
the audio generation interface,
how the output loop affect trigger works in precise detail,
how the system detects gap between what it feels and what it expressed,
the revision and suppression mechanisms,
how different modalities carry different amounts of internal state
(text most controllable, audio least),
and how expression style evolves as personality develops.

#### COMPONENT H — THE INPUT PIPELINE AND WORLD INTERFACE
Design the complete input and world interface. This includes:
how each modality (text, image, audio) is encoded to representation space,
why they all enter the same pipeline,
how internal thought (replay, simulation, self-generated) is processed
identically to external input,
the prediction pre-computation (what is predicted before input arrives),
the gap computation at input,
how the system builds a model of external agents
(theory of mind — what does the other know that I know?),
and how the boundary between self and world is represented.

---

### OUTPUT FORMAT (per component)

Text only. No code. No diagrams attempted in ASCII unless they genuinely clarify something prose cannot.

```
COMPONENT [X] — [NAME]

OVERVIEW
One paragraph. What this component is, what it does, why it exists.

CORE DATA STRUCTURES
Every persistent data structure this component owns.
Name, fields, types, purpose of each field.

ALGORITHMS
Every significant computation. Name. Inputs, process, outputs.
Specific enough to implement from this description alone.

INTERFACES
What other components call into this one.
What this component calls into others.
Every interface is a contract.

FAILURE MODES
What can go wrong. How each failure manifests.
What the correct response is.

OPEN QUESTIONS
Things deliberately left unresolved.
Questions that require empirical answer (run it and see).
Design decisions that depend on other components being specified first.
```

Be thorough. Be precise. Be honest about uncertainty.
If the spec is underspecified or contradictory, call it out.

---

### DISPATCH PLAN

Wave 1 — substrate (sequential dependencies satisfied first):
  A — Affective Engine
  B — Prediction Engine
  C — Concept Graph

Wave 2 — built on substrate (parallel after wave 1):
  D — Simulation + Replay
  E — Identity + Private State
  F — Attention + Sparse Activation
  G — Expression Layer
  H — Input Pipeline + World Interface

Final — synthesis into a unified spec before any code is written.
