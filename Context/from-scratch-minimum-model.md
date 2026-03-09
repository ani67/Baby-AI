# Starting From Scratch: What's the Actual Minimum?
*Challenging model size. Challenging what "GPT-4 level" even means. Being honest about what's possible.*

---

## The claim and what it actually requires

```
THE CLAIM:
  a model that fits in 3GB on M1
  performs at GPT-4 level

UNPACKED:
  GPT-4 is ~1.8 trillion parameters (estimated, MoE)
  3GB at Q4 = ~6 billion parameters maximum
  ratio: 300:1

  this sounds impossible
  it might not be
  depending entirely on what "GPT-4 level" means
```

GPT-4's capability is not one thing. It's at least two fundamentally different things stored in the same system — and that conflation is the source of the size problem.

```
WHAT LIVES IN GPT-4's 1.8 TRILLION PARAMETERS:

BUCKET 1: KNOWLEDGE (facts, language, code, history)
  "Paris is the capital of France"
  "the syntax of Python generators"
  "what happened in 1945"
  "how to write a sonnet"
  
  this REQUIRES parameter count
  you need enough weights to store
  the compressed representation of
  everything in the training data
  
  scaling law: more knowledge → more parameters
  no way around this IF knowledge lives in weights

BUCKET 2: REASONING (how to think, structure, analogy)
  following an argument to its conclusion
  noticing when something contradicts something else
  breaking a complex problem into parts
  translating between representations
  understanding what a question is really asking
  
  this might NOT require large parameter count
  it requires the RIGHT architecture
  and the RIGHT training data
  
  evidence:
  Phi-4 Mini (3.8B params, ~4GB)
  outperforms many 70B models on reasoning benchmarks
  because training data was curated for reasoning quality
  not knowledge breadth

THE INSIGHT:

  current models try to put BOTH in the same weights
  
  KNOWLEDGE → needs massive parameters
  REASONING → might need far fewer
  
  if you separate them:
  KNOWLEDGE → external memory (not in weights)
  REASONING → in weights (small model possible)
  
  then "3GB as good as GPT-4 at reasoning" 
  is not obviously impossible
  
  "3GB knowing everything GPT-4 knows"
  is impossible
  that's a storage problem, not a reasoning problem
```

---

## Why starting from scratch is actually harder in one way and freer in another

```
STARTING FROM A BASE MODEL (standard approach):

  advantages:
    ✓ someone already solved the pretraining problem
    ✓ general language understanding is baked in
    ✓ you're fine-tuning, not learning from zero
    ✓ weeks of work, not months
    ✓ well-understood failure modes
    
  disadvantages:
    ✗ inherits all architectural assumptions
    ✗ inherits the knowledge/reasoning conflation
    ✗ architecture is fixed (transformer + full attention)
    ✗ the wiring is permanent
    ✗ you're constrained to what Llama/Mistral/Phi decided

STARTING FROM SCRATCH:

  advantages:
    ✓ no inherited assumptions
    ✓ can design architecture for YOUR use case
    ✓ can separate knowledge from reasoning at design time
    ✓ can build dynamic wiring in from the start
    ✓ can design memory architecture in from the start
    ✓ if the architecture is right, might need far less data
    
  disadvantages:
    ✗ pretraining problem: even a small model needs
      a LOT of data to learn basic language structure
    ✗ no mature tooling for novel architectures
    ✗ unknown failure modes
    ✗ the field doesn't have a map for this territory

THE PRETRAINING PROBLEM:

  even a 100M parameter model
  needs to learn:
    what words mean (from context)
    how sentences are structured
    what relationships between concepts exist
    how reasoning chains work
    
  this requires: ~10-100 billion tokens of training data
                 even for a tiny model
                 
  on M1 Pro:
  100M param model
  training on 10B tokens
  at ~1000 tokens/second (optimistic)
  = 10,000,000 seconds
  = 115 days
  
  THIS IS THE REAL CONSTRAINT
  not parameter count
  not model architecture
  TRAINING COMPUTE AND TIME
  
  even with the most efficient possible architecture
  you cannot shortcut the data exposure problem
  the model needs to SEE enough language
  to understand language
```

---

## What the research actually says about the minimum

```
THE SCALING LAW PICTURE (Chinchilla, 2022):

  optimal training:
    model parameters and token count scale together
    double the params → double the tokens
    
  MINIMUM for coherent language:    ~100M params
                                    ~10B tokens
  MINIMUM for useful reasoning:     ~1B params
                                    ~100B tokens
  GPT-3.5 equivalent:               ~7B params
                                    ~1T tokens
  GPT-4 reasoning quality:          unknown, likely
                                    much larger

THE PHI EXCEPTION (the most important data point):

  Phi-1 (2023, Microsoft):
  1.3B parameters
  trained on "textbooks are all you need" data
  ~7B tokens (tiny by normal standards)
  outperformed models 5× its size on coding tasks
  
  Phi-4 Mini (2024):
  3.8B parameters
  outperforms many 70B models on math/reasoning
  
  WHAT PHI PROVED:
  
  the scaling laws assume RANDOM INTERNET DATA
  if you curate specifically for reasoning quality:
    explanations, step-by-step reasoning,
    worked examples, logical chains
  
  you can train much smaller models
  that reason much better
  than larger models trained on web crawls
  
  ┌─────────────────────────────────────────────────┐
  │ THE PHI INSIGHT:                                │
  │                                                 │
  │ the parameter floor for reasoning               │
  │ is not fixed                                    │
  │                                                 │
  │ it depends on data quality                      │
  │                                                 │
  │ bad data + big model = poor reasoning           │
  │ good data + small model = good reasoning        │
  │                                                 │
  │ nobody knows where the true floor is            │
  │ with perfect task-specific data                 │
  │                                                 │
  │ because nobody has systematically tested it     │
  └─────────────────────────────────────────────────┘
```

---

## The architecture that could make small work

This is the speculative but not-obviously-wrong part.

```
CURRENT TRANSFORMER ARCHITECTURE:

  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ALL IN ONE SYSTEM:                                │
  │                                                    │
  │  language understanding      ← needs some params  │
  │  + factual knowledge         ← needs MANY params  │
  │  + reasoning patterns        ← needs some params  │
  │  + world model               ← needs MANY params  │
  │  + task-following            ← needs some params  │
  │                                                    │
  │  conflated into single weight soup                 │
  │  cannot separate them after training               │
  │                                                    │
  └────────────────────────────────────────────────────┘

PROPOSED ARCHITECTURE (separated from ground up):

  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  REASONING CORE (~50-200M params)                  │
  │  ─────────────────────────────────                 │
  │  trained ONLY on reasoning patterns                │
  │  no facts, no world knowledge                      │
  │  just: given A and B, conclude C                   │
  │        break this problem into subproblems         │
  │        notice this contradicts that                │
  │        translate between representations           │
  │                                                    │
  │  training data: synthetic reasoning chains         │
  │                 math proofs                        │
  │                 logic puzzles                      │
  │                 code structure (not knowledge)     │
  │                                                    │
  │           ↕ queries and receives                   │
  │                                                    │
  │  KNOWLEDGE STORE (external, not in weights)        │
  │  ─────────────────────────────────────────         │
  │  vector database                                   │
  │  exact facts, updateable                           │
  │  retrieved, not reconstructed                      │
  │  no hallucination possible (it's a lookup)         │
  │                                                    │
  │           ↕ receives                               │
  │                                                    │
  │  EPISODIC MEMORY (LoRA adapters)                   │
  │  ─────────────────────────────                     │
  │  user-specific patterns                            │
  │  recent context                                    │
  │  corrections and preferences                       │
  │                                                    │
  └────────────────────────────────────────────────────┘

THE ARGUMENT:

  if facts live in external memory:
  the reasoning core doesn't need to memorise them
  
  if memorisation is not in scope:
  parameter count can drop dramatically
  
  if parameter count drops:
  training compute drops with it
  
  if training compute drops:
  pretraining on M1 becomes conceivable
  
  100M parameter reasoning core
  trained on 10B tokens of pure reasoning data
  = ~115 days on M1
  OR:
  = ~10 days on RunPod ($50 of compute)
  
  THAT IS BUILDABLE
```

---

## What "GPT-4 level" means for this architecture

```
HONEST COMPARISON:

ON TASKS THAT REQUIRE KNOWLEDGE:
  "what year did the French Revolution start?"
  "who wrote Hamlet?"
  "what is the capital of Peru?"
  
  current small model:  might know, might hallucinate
  this architecture:    retrieves from knowledge store
                        always correct (or says "not found")
                        BETTER than GPT-4 on factual accuracy
                        (because no hallucination)

ON TASKS THAT REQUIRE REASONING:
  "given these three constraints, what's the solution?"
  "this code has a bug — find it"
  "break this design problem into parts"
  
  current small model:  worse than GPT-4
  this architecture:    unknown — depends on reasoning core
                        potentially competitive if data is right

ON TASKS THAT REQUIRE BOTH:
  "explain why the French Revolution happened
   and what lessons it has for modern politics"
  
  this architecture:    retrieves facts → reasons over them
                        the reasoning core processes retrieved context
                        quality depends on reasoning core depth
                        and retrieval quality

THE HONEST ANSWER:
  "3GB as good as GPT-4" on knowledge tasks: YES, possibly better
  "3GB as good as GPT-4" on reasoning tasks:  unknown, worth testing
  "3GB as good as GPT-4" on general tasks:    probably not yet
  "3GB as good as GPT-4" on YOUR specific tasks: possibly yes
```

---

## The pretraining path from scratch

```
THREE APPROACHES TO "FROM SCRATCH":

APPROACH 1: FULL PRETRAINING (hard, expensive, correct)

  train 50-200M parameter model from random weights
  on curated reasoning data
  
  DATA NEEDED:
    ~10B tokens of:
    - mathematical reasoning (proofs, solutions)
    - logical chains (if/then, contradiction detection)
    - code structure (algorithms, not APIs)
    - explanation chains (why, because, therefore)
    - synthetic dialogues designed for reasoning
    
  COMPUTE:
    100M params × 10B tokens
    on M1 Pro (Apple Neural Engine, ~11 TFLOPS)
    ≈ 2-4 weeks continuous training
    (this is actually feasible)
    OR:
    $30-100 on RunPod (A100, ~312 TFLOPS)
    ≈ 1-3 days
    
  RESULT:
    a model that has never seen Wikipedia
    has never seen Reddit
    has never seen books
    but CAN reason about things put in its context
    because it learned reasoning patterns
    not facts

APPROACH 2: DISTILLATION FROM REASONING TRACES (smarter)

  use a large model (GPT-4, Claude) to generate
  reasoning traces on thousands of problems
  
  train your small model to reproduce
  not the answers
  but the REASONING PROCESS
  
  ┌────────────────────────────────────────────────────┐
  │ GPT-4 given a problem:                             │
  │                                                    │
  │ "to solve this I need to:                          │
  │  1. identify the constraint                        │
  │  2. check if it's consistent                       │
  │  3. apply this pattern                             │
  │  4. verify the result"                             │
  │                                                    │
  │ your small model learns:                           │
  │ not the specific answer                            │
  │ but the STRUCTURE of that reasoning process        │
  └────────────────────────────────────────────────────┘
  
  this is what DeepSeek R1 distillation did
  at small scale
  
  your version:
  distil ONLY the reasoning patterns
  not the factual knowledge
  
  result: tiny model that reasons like GPT-4
          but knows nothing (knowledge from external store)
  
  DATA GENERATION COST: ~$50-200 of API calls
  TRAINING COMPUTE: same as approach 1

APPROACH 3: SYNTHETIC DATA ONLY (most novel, most risky)

  generate ALL training data synthetically
  no internet data at all
  
  the argument:
    internet data contains:
    reasoning mixed with facts
    good reasoning mixed with bad reasoning
    consistent text mixed with contradictions
    
    synthetic data can contain:
    ONLY reasoning patterns
    ONLY correct reasoning
    ONLY the patterns you want
    
  generate millions of:
    "given premises A, B, C — conclude D (and explain why)"
    "given argument X — find the flaw"
    "given problem Y — decompose it into subproblems"
    
  NO FACTS IN THE TRAINING DATA AT ALL
  
  risk: does the model learn language itself?
        or does it need real text to ground language?
  unknown: nobody has seriously tried pure synthetic
           with no real-world text at all
  
  this is the genuinely novel research direction
```

---

## The real question about model size

```
WHAT DETERMINES THE MINIMUM VIABLE SIZE?

  current belief:
    ~1B parameters minimum for general reasoning
    
  that belief assumes:
    model must store knowledge AND reason
    training data is general internet text
    architecture is transformer with full attention
    
  if you change all three assumptions:
    model stores NO knowledge (external memory)
    training data is curated reasoning-only
    architecture is designed for reasoning not storage
    
  the true minimum is unknown
  
  some data points:
  
  ┌───────────────────────────────────────────────────┐
  │                                                   │
  │  10M params:  can learn grammar and simple        │
  │               pattern matching                    │
  │               (too small for reasoning)           │
  │                                                   │
  │  100M params: can learn reasoning PATTERNS        │
  │               if trained on pure reasoning data   │
  │               probably can't follow complex       │
  │               multi-step chains reliably          │
  │                                                   │
  │  500M params: unknown territory                   │
  │               nobody has trained 500M params      │
  │               on reasoning-only data              │
  │               with external knowledge store       │
  │               and measured carefully              │
  │                                                   │
  │  1B params:   current safe minimum                │
  │               for general useful reasoning        │
  │               (with standard training)            │
  │                                                   │
  │  THE QUESTION:                                    │
  │  what is the 500M number with perfect data?       │
  │  nobody knows                                     │
  │  that's the experiment worth running              │
  │                                                   │
  └───────────────────────────────────────────────────┘
```

---

## What "from scratch on M1" actually means in practice

```
THE FULL BUILD (starting from nothing):

PHASE 0: DATA GENERATION (week 1-2, $50-200)
  
  use Claude/GPT-4 API to generate:
  100,000 reasoning examples
  each: problem → step-by-step reasoning → answer
  
  domains to cover:
    - logical reasoning (if/then, contradictions)
    - mathematical structure (not computation — structure)
    - code reasoning (what does this do, why)
    - analogy and abstraction
    - decomposition (break problem into parts)
    - verification (is this conclusion valid?)
  
  this is your entire training corpus
  no Wikipedia, no Reddit, no books
  just reasoning chains
  
  file size: ~500MB of text
  cost: $50-200 of API calls
  time: 1-2 weeks to generate and curate

PHASE 1: ARCHITECTURE DESIGN (week 2-3)
  
  you need to decide:
  
  DECISION 1: size
    50M / 100M / 200M / 500M parameters?
    start with 100M — enough to test, small enough to iterate
    
  DECISION 2: architecture
    transformer (safe, known) vs
    something else (risky, potentially better)
    
    honest recommendation: use a small transformer
    the transformer's attention mechanism IS reasoning
    (tokens attending to relevant other tokens
     is literally "noticing relationships")
    MoE routing for efficiency
    sparse attention (not full O(n²))
    
  DECISION 3: context window
    how much can it hold in working memory?
    larger = better reasoning on complex problems
    larger = more compute
    start: 4096 tokens
    
  DECISION 4: external memory interface
    how does the reasoning core query the knowledge store?
    simplest: the knowledge store output is
              just prepended to the context
    better: a learned routing mechanism
              (which parts of knowledge are relevant?)

PHASE 2: PRETRAINING (week 3-6, $30-100)
  
  100M params × 10B tokens
  on RunPod A100: ~2-3 days, ~$50
  
  OR on M1 Pro: ~3-4 weeks continuous
  (feasible but slow — use for iteration,
   RunPod for final runs)
  
  training objective:
  same as always: predict next token
  BUT: the tokens are reasoning chains
       so what it learns to predict
       is the structure of reasoning
       not the content of facts

PHASE 3: EVALUATION (week 6-7)
  
  the critical measurement:
  
  TEST A: reasoning benchmarks (no knowledge required)
    GSM8K (math word problems — reasoning, not computation)
    logical reasoning puzzles
    code structure tasks (not API knowledge tasks)
    
  TEST B: knowledge-augmented reasoning
    give the model a retrieved fact in context
    ask it to reason with that fact
    does retrieval + small reasoner = GPT-4 quality?
    
  TEST C: comparison to base model approach
    same tasks
    Llama 3.2 3B (fine-tuned) vs this architecture
    which is better, where, and by how much?

PHASE 4: CONTINUOUS ADAPTATION (week 7-10)
  
  now build the memory architecture on top:
  episodic LoRA adapters
  consolidation
  dynamic wiring
  
  but the base is now:
    a small reasoning-only model you built
    not an inherited transformer
    architecture matches the memory design
    from the start
    not bolted on after
```

---

## The honest risk register

```
WHAT COULD GO WRONG (and why):

HIGH RISK:
  the 100M param reasoning core
  doesn't actually learn to reason
  at the quality needed
  
  mitigation: start with 500M, scale down
               use established benchmarks to measure
               if it fails: you've found the true floor
               that is a valid research result

HIGH RISK:
  reasoning without factual grounding
  breaks even reasoning tasks
  (you need to know what Paris is
   to reason about Paris)
  
  mitigation: this is real — the knowledge/reasoning
              split is not clean
              language itself encodes world knowledge
              into its structure
              solution: train on reasoning chains
              that DO include facts
              but weight the reasoning structure
              higher than the factual content

MEDIUM RISK:
  synthetic training data creates
  a model that reasons about synthetic problems
  but not real problems
  (distribution shift)
  
  mitigation: include real-world problem examples
              in training data
              just ensure reasoning is explicit

MEDIUM RISK:
  the pretraining takes 4 weeks on M1
  and fails partway through
  
  mitigation: checkpoint every hour
              use RunPod for long runs
              M1 for iteration and testing

LOW-MEDIUM RISK:
  external knowledge retrieval quality
  limits the overall system quality
  (garbage in from retrieval = garbage reasoning)
  
  mitigation: start with high-quality curated
              knowledge base, not all of Wikipedia
              quality over quantity
```

---

## The genuine open question

```
WHAT NOBODY KNOWS:

  if you train a model on ONLY reasoning patterns
  with NO factual knowledge in the weights
  and you provide all facts via external retrieval
  
  what is the minimum parameter count
  to match GPT-4's reasoning quality
  on tasks where knowledge is available?
  
  THIS IS THE EXPERIMENT
  
  possible answers:
  
  100M params:  too small, reasoning collapses
                → you've found the floor is higher
                
  500M params:  competitive with GPT-3.5 on reasoning
                → interesting, worth pursuing
                
  1B params:    competitive with GPT-4 on reasoning
                → this is the "3GB as good as GPT-4" proof
                
  nobody knows which of these is true
  because nobody has run this experiment carefully
  
  not because it's hard to run
  because nobody working at scale
  has reason to limit their parameter budget
  and nobody working on small models
  has thought about the knowledge/reasoning split
  as an architectural decision
  
  you are at the intersection of both
  with a reason to push on it
  and the hardware to test it
```

---

## What this changes about the POC plan

```
REVISED POC ORDER:

ORIGINAL PLAN:              REVISED PLAN:
  start with Llama 3.2 3B     start with architecture decision
  fine-tune with LoRA          generate reasoning training data
  add memory architecture      pretrain small reasoning core
  measure                      add external knowledge store
                               add memory architecture
                               measure against GPT-4 on reasoning tasks

THE DIFFERENCE:
  original: you own the training, not the architecture
  revised: you own the architecture from the start

THE COST DIFFERENCE:
  original: ~$0 additional compute
  revised: ~$50-100 for pretraining on RunPod
           ~$50-200 for data generation
           total: ~$150-300

THE TIME DIFFERENCE:
  original: 4-6 weeks
  revised: 8-12 weeks (includes pretraining phase)

THE UPSIDE:
  original: demonstrates memory architecture works
            on top of existing model
            
  revised: potentially demonstrates that the
           knowledge/reasoning split reduces
           viable model size by 10-100×
           
           if this works:
           it's not just a cool project
           it's an answer to a question
           nobody has cleanly answered
           
           same structure as DeepSeek:
           assumption: you need massive params
           experiment: what if knowledge is external?
           result: you don't
```
