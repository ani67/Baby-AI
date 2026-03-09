# On-Demand Training Data From Search
*Why this is a paradigm shift, what it actually requires, and where it breaks*

---

## What you're actually proposing

There are two versions of this idea. The shallow version and the deep version.

```
SHALLOW VERSION:
  instead of compiling a static training dataset
  use search results as training data
  
  ┌─────────────────────────────────────────┐
  │  static dataset         search-based    │
  │  ─────────────          ───────────     │
  │  compiled once          acquired live   │
  │  fixed forever          grows with use  │
  │  you choose what's in   model encounters│
  │  it                     what's needed   │
  └─────────────────────────────────────────┘
  
  this is interesting but not revolutionary
  it's just a different data pipeline

DEEP VERSION:
  the model's OWN FAILURES drive what it learns next
  
  ┌────────────────────────────────────────────────┐
  │                                                │
  │  model encounters something it can't handle   │
  │           │                                   │
  │           ▼                                   │
  │  failure detected (high uncertainty,          │
  │  wrong answer, "I don't know")                │
  │           │                                   │
  │           ▼                                   │
  │  model formulates a search query              │
  │  based on WHAT IT DOESN'T KNOW               │
  │           │                                   │
  │           ▼                                   │
  │  retrieves relevant information               │
  │           │                                   │
  │           ▼                                   │
  │  updates from the retrieved content           │
  │  weights shift toward correct behavior        │
  │           │                                   │
  │           ▼                                   │
  │  next encounter with similar input:           │
  │  model handles it better                      │
  │                                                │
  └────────────────────────────────────────────────┘
  
  the data acquisition IS the learning
  the learning IS the data acquisition
  they are the same loop
  
  THIS IS WHAT YOU'RE ACTUALLY POINTING AT
  and it doesn't exist as a working system anywhere
```

---

## Why this is a different paradigm

```
ALL CURRENT TRAINING PARADIGMS:

  DATA → MODEL (data drives learning)
  
  you compile a dataset
  you train on it
  the model is a compression of that dataset
  
  if the dataset is missing something:
  the model doesn't know it
  and cannot acquire it
  
  ──────────────────────────────────────────
  
  WHAT YOU'RE PROPOSING:

  MODEL GAPS → DATA (gaps drive acquisition)
  
  the model decides what it needs to learn
  based on where it's failing
  and acquires it on demand
  
  the model is not a compression of a fixed dataset
  it's a LEARNER that grows toward competence
  in the domain it's actually used in

THIS IS CALLED (in research):
  Active Learning:     model decides what to label next
  Curriculum Learning: model controls its own training order
  Online Learning:     model updates from live data stream
  Self-directed Learning: model generates its own curriculum
  
  none of these are the same as your idea
  your idea combines elements of all four
  and adds: search as the data acquisition mechanism
  + failure as the trigger
  + continuous as the mode
  
  that specific combination doesn't have a name
  because it doesn't exist as a working system yet
```

---

## The five real problems

### Problem 1: The Training Signal Problem (hardest)

```
THIS IS THE ONE THAT'S GENUINELY UNSOLVED

to train from a search result
you need a TARGET
something the model is trying to match
a loss function
something to measure "was this update good?"

CASE A: the model answers wrong
  you search for the right answer
  you have a clear target: the correct answer
  you can compute loss
  you can update
  
  THIS WORKS (in principle)
  
  ┌──────────────────────────────────────────┐
  │  model: "the capital of Australia is     │
  │          Sydney"                         │
  │  signal: WRONG                           │
  │  search: "capital of Australia"          │
  │  result: "Canberra"                      │
  │  target: "Canberra"                      │
  │  loss: model said Sydney, target Canberra│
  │  update: push weights away from Sydney,  │
  │           toward Canberra                │
  └──────────────────────────────────────────┘
  
  but wait — the model should NOT have
  facts in its weights at all
  (architecture from previous doc)
  facts go in the knowledge store
  
  so the update isn't a weight update
  it's: add "capital of Australia = Canberra"
  to the knowledge store
  and the reasoning core learns:
  "when I don't have this fact, retrieve it"

CASE B: the model reasons wrong
  it has the facts
  but draws the wrong conclusion
  
  THIS IS HARDER
  
  what does "wrong reasoning" look like as a gradient?
  the model said: "A and B therefore C"
  the right answer is: "A and B therefore D"
  
  search can give you D
  but why did the model conclude C?
  which weights caused the wrong inference?
  how do you update them without breaking
  all the other things those weights do?
  
  ┌──────────────────────────────────────────┐
  │ this is the interpretability problem     │
  │ again                                    │
  │                                          │
  │ if we can't read what we built           │
  │ we can't know which weights caused       │
  │ the wrong reasoning                      │
  │ so we can't target the update precisely  │
  │                                          │
  │ gradient descent will update all weights │
  │ that contributed to the wrong output     │
  │ some of those weights also do correct    │
  │ things                                   │
  │ you'll degrade some good behavior        │
  │ while fixing the bad behavior            │
  │ the usual catastrophic forgetting risk   │
  └──────────────────────────────────────────┘

CASE C: the model says "I don't know"
  this is actually the cleanest case
  
  model: "I don't know the answer to this"
  search: retrieves relevant content
  the retrieved content IS the answer
  
  now what?
  
  option 1: add to knowledge store (correct for facts)
  option 2: train reasoning core on the example
             (correct for reasoning patterns)
  option 3: just use it in context this time
             (no learning, just retrieval)
  
  option 3 is RAG (already exists)
  option 1 is knowledge store update (manageable)
  option 2 requires knowing what to do with
           a search result as a training example
```

### Problem 2: The Quality Filter Problem

```
SEARCH RESULTS ARE NOT TRAINING DATA
they are raw text from the web

quality spectrum:
  ████████  authoritative, accurate, well-structured
  ████░░░░  mostly accurate, some noise
  ██░░░░░░  variable quality, opinions mixed with facts
  █░░░░░░░  SEO spam, wrong, misleading
  ░░░░░░░░  actively harmful, factually wrong

a human reading search results knows intuitively:
  "this Wikipedia article is reliable"
  "this forum post might be wrong"
  "this news article has a perspective"
  "this is an ad disguised as content"

the model training on raw search results:
  learns from authoritative AND bad sources equally
  unless you build quality filtering
  
  quality filtering requires:
  knowing what quality is
  which requires... knowing the correct answer
  which is what you're trying to learn
  
  CIRCULAR DEPENDENCY:
  to know if a source is good
  you need to know the answer
  to know the answer
  you need a good source
  
  REAL SOLUTIONS:

  SOLUTION A: source weighting
    Wikipedia > academic papers > news > forums > random
    crude but real
    
  SOLUTION B: cross-source validation
    if 5 independent sources agree → high confidence
    if sources disagree → flag, don't train on it
    
  SOLUTION C: model self-consistency check
    generate answer from multiple retrieved sources
    if model gives consistent answers → update
    if inconsistent → don't update, flag for review
    
  SOLUTION D: human feedback loop
    uncertain updates → human confirms
    confirmed → train
    rejected → discard
    this is RLHF but for data acquisition
    not just for response quality
```

### Problem 3: The Bootstrapping Problem

```
BEFORE THE MODEL KNOWS ANYTHING:
  how does it know what to search for?
  how does it formulate good queries?
  how does it recognize its own failures?
  
  ┌────────────────────────────────────────────────┐
  │  to search well you need to know:              │
  │    what question to ask                        │
  │    what vocabulary to use                      │
  │    how to interpret the results                │
  │    whether the result is relevant              │
  │                                                │
  │  all of this requires some prior knowledge     │
  │                                                │
  │  a model that knows nothing                    │
  │  cannot formulate meaningful search queries    │
  │  about the things it doesn't know              │
  │                                                │
  │  "I don't know X" requires knowing enough      │
  │  to recognise X as a category of knowledge     │
  └────────────────────────────────────────────────┘
  
  THIS IS NOT A FATAL PROBLEM
  but it means you can't start from zero
  
  you need a bootstrapping corpus:
    enough language to understand language
    enough reasoning to recognize failures
    enough world model to formulate queries
    
  this is smaller than a general pretraining corpus
  but it's not nothing
  
  estimate: ~1B tokens of bootstrapping data
  to get to "can formulate search queries"
  and "can recognize its own failures"
  
  after that: search takes over
```

### Problem 4: The Loop Speed Problem

```
TRAINING FROM SEARCH IN REAL TIME:

  user asks question
  model doesn't know
  search takes: 0.5-2 seconds
  quality filter: 0.5-1 second
  extract relevant content: 0.5 second
  compute gradient: 1-3 seconds (on M1 for small model)
  update weights: 0.5 second
  generate answer: 1-2 seconds
  
  TOTAL: 4-10 seconds per unknown question
  
  vs standard RAG:
  search: 0.5-2 seconds
  generate answer with context: 1-2 seconds
  TOTAL: 2-4 seconds
  
  the training step adds latency
  
  SOLUTION: decouple the update from the response
  
  ┌────────────────────────────────────────────────┐
  │                                                │
  │  user asks → model responds (with retrieval)  │
  │              │                                 │
  │              │ response goes to user           │
  │              │ SIMULTANEOUSLY                  │
  │              ▼                                 │
  │              training queue                    │
  │              (background process)              │
  │              model trains on this interaction  │
  │              AFTER user has their answer       │
  │                                                │
  │  user never waits for training                 │
  │  training happens in background                │
  │  unified memory makes this possible on M1      │
  │  (inference and training share same pool)      │
  │                                                │
  └────────────────────────────────────────────────┘
```

### Problem 5: The Legal / ToS Problem

```
TRAINING ON SCRAPED WEB CONTENT:

  using search results for INFERENCE (RAG):
    generally accepted
    fair use argument
    no persistent copy
    
  using search results for TRAINING (weights update):
    grey area at best
    model learns from content permanently
    copyright holders have argued this is infringement
    (the current AI training lawsuit landscape)
    
  PRACTICAL MITIGATIONS:
  
  use only:
    explicitly licensed content (Creative Commons)
    content where training is permitted (arxiv, Wikipedia)
    content you generate yourself (synthetic)
    content user provides themselves
    
  or:
    use search for KNOWLEDGE STORE only
    (not for weight updates)
    knowledge store = exact retrieval, not training
    this sidesteps the copyright issue entirely
    
  THE CLEANEST VERSION:
  
  weights ← trained on synthetic reasoning data (yours)
  knowledge store ← populated from search (retrieval only)
  
  training (weight updates): synthetic or user-generated only
  factual knowledge: retrieved live, never trained into weights
  
  this is legally cleaner AND architecturally correct
```

---

## The architecture that actually works

```
COMBINING EVERYTHING:

  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  USER INPUT                                              │
  │       │                                                  │
  │       ▼                                                  │
  │  REASONING CORE (small, ~100-500M params)                │
  │  "what do I need to answer this?"                        │
  │       │                                                  │
  │       ├──── I have the reasoning pattern                 │
  │       │     answer directly                              │
  │       │                                                  │
  │       ├──── I need a fact                                │
  │       │          │                                       │
  │       │          ▼                                       │
  │       │     KNOWLEDGE STORE QUERY                        │
  │       │     is it in local store?                        │
  │       │          │                                       │
  │       │     YES: retrieve and use                        │
  │       │          │                                       │
  │       │     NO:  SEARCH                                  │
  │       │          retrieve from web                       │
  │       │          quality filter                          │
  │       │          add to knowledge store                  │
  │       │          use it                                  │
  │       │                                                  │
  │       └──── I don't know how to reason about this       │
  │                  │                                       │
  │                  ▼                                       │
  │             SEARCH FOR REASONING EXAMPLES                │
  │             "how do people reason about X?"              │
  │             retrieve worked examples                     │
  │             add to TRAINING QUEUE (background)           │
  │             use retrieved example in context             │
  │             later: train on the example                  │
  │                                                          │
  │  ─────────────────────────────────────────────────────   │
  │                                                          │
  │  BACKGROUND (happening continuously, M1 Neural Engine):  │
  │                                                          │
  │  TRAINING QUEUE → reasoning examples from search         │
  │                    user corrections                       │
  │                    confirmed good responses               │
  │                                                          │
  │  CONSOLIDATION → knowledge store grows and prunes        │
  │                   reasoning core updates slowly          │
  │                   via LoRA on queue items                │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

THE KEY SEPARATIONS:

  SEARCH → knowledge store    (not weights)
  SEARCH → training queue     (eventually weights, for reasoning)
  WEIGHTS ← reasoning only    (not facts)
  WEIGHTS ← high quality only (not raw search)
  
  weights update:  slow, careful, high quality gate
  knowledge store: fast, live, search-populated
  
  this is the architecture that makes
  on-demand training data safe and useful
```

---

## What this means for the model size question

```
REVISITING: "3GB as good as GPT-4"

WITH STATIC TRAINING DATA:
  model must store reasoning AND knowledge
  → large parameter count needed for knowledge
  → 3GB probably insufficient for GPT-4 level

WITH ON-DEMAND SEARCH + KNOWLEDGE STORE:
  model stores ONLY reasoning patterns
  all factual knowledge: retrieved live
  
  KNOWLEDGE PERFORMANCE:
    model + search > GPT-4 on factual accuracy
    because GPT-4 hallucinates
    this architecture retrieves exact text
    no hallucination possible on facts
    
  REASONING PERFORMANCE:
    depends entirely on reasoning core quality
    100-500M params trained on pure reasoning data
    unknown whether this matches GPT-4
    THAT is the experiment
    
  BUT:
    on the tasks that matter most day-to-day:
    "help me think through this problem"
    "what does this concept mean"
    "how should I approach this decision"
    
    these are reasoning tasks with facts provided
    by the user or retrieved from search
    
    a 500M param reasoning core
    with live search
    might genuinely match GPT-4
    on these specific tasks
    
    nobody has tested this
    because nobody has built this
```

---

## The new POC design

```
WHAT CHANGES WITH SEARCH-DRIVEN TRAINING:

PREVIOUS POC:
  1. pretrain small reasoning core (static data)
  2. add memory architecture
  3. measure

REVISED POC:
  1. pretrain minimal bootstrapping core
     (~100M params, ~1B synthetic tokens)
     just enough to: understand language
                     recognise failures
                     formulate queries
     
  2. add knowledge store (empty at start)
  
  3. add search integration:
     unknown fact → search → add to knowledge store
     
  4. add training queue:
     reasoning failures → search for examples
     → queue for background LoRA update
     
  5. add quality filter:
     cross-source validation before anything
     enters training queue
     
  6. measure:
     does the system get better over time?
     at what rate?
     where does quality plateau?
     how does it compare to GPT-4 on target tasks?

TIMELINE CHANGE:
  previous: 8-12 weeks
  revised: 10-14 weeks (search integration adds ~2 weeks)
  
  cost change:
  previous: ~$150-300 (pretraining + data generation)
  revised: ~$200-400 (same + search API costs)
           Google Search API: ~$5 per 1000 queries
           for testing: probably <$50

THE MOST IMPORTANT DIFFERENCE:

  previous POC: capability fixed after pretraining
                (only personalization grows)
                
  revised POC:  capability grows continuously
                from user's actual usage
                
  a system that gets better
  the more it's used
  in ways specific to how YOU use it
  
  not fine-tuned on your preferences
  actually learning the domain you work in
  from the searches your usage triggers
  
  this is a qualitatively different kind of system
  than anything currently available
  at any price
  at any scale
```

---

## The one-sentence version

```
CURRENT AI:   here is everything I learned
              before we talked
              I cannot learn anything new
              from our conversation
              
THIS SYSTEM:  I start with the ability to reason
              every conversation teaches me
              what I don't know
              I go find it
              I remember it
              next time I know it
              
THE DIFFERENCE:
  current AI is a very smart book
  that was printed once
  
  this is something closer to
  a mind that's actually learning
  not from a training set
  but from the world
  via search
  in real time
  
  the training set is unbounded
  because the world is unbounded
  and search is the interface to the world
```
