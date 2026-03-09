# POC Time Estimate: Continuous Local AI on M1
*Honest breakdown. Claude Code does the coding. You do the thinking.*

---

## First: what "from scratch" actually means here

```
FROM SCRATCH ON CODE:
  you write almost nothing
  Claude Code writes it
  you review, redirect, test
  this is fast

FROM SCRATCH ON ARCHITECTURE:
  nobody has assembled these pieces before
  in this specific combination
  you will hit unknown failure modes
  this is slow and unpredictable

FROM SCRATCH ON THEORY:
  the local learning signal without backprop
  is a research problem
  not an engineering problem
  this is NOT a POC timescale

THE ESTIMATE BELOW ASSUMES:
  Claude Code handles all implementation
  you handle all architectural decisions
  you have ~2-3 focused hours per day
  (not full-time, realistic for someone with a job)
```

---

## The four components and their honest difficulty

```
COMPONENT         WHAT IT IS              DIFFICULTY    WHY
──────────────────────────────────────────────────────────────────────
1. Double-buffered known approach,         MEDIUM        the double-buffer
   LoRA adaptation not assembled           (engineering)  swap logic is
                  this way yet                           fiddly to get right
                                                         without race conditions

2. Two-layer      architecturally          MEDIUM-HARD   the consolidation
   memory arch    clear, the              (architecture)  decision — what
   with           consolidation step                      merges, what prunes —
   consolidation  is the unknown                          requires defining
                                                          a quality metric

3. Crude dynamic  growing LoRA rank        HARD          MLX doesn't natively
   wiring         dynamically,            (systems)      support dynamic
                  managing memory                         model modification
                  on M1                                   you'll fight the
                                                          framework

4. No backprop /  genuinely new           RESEARCH       not a POC
   local learning  learning signal         NOT            weeks → years
                  at transformer scale    ENGINEERING     depending on approach
```

---

## What to actually build (scoped honestly)

```
FULL VISION:                           REALISTIC POC:

simultaneous train + inference    →    LoRA double-buffer (achieves this)
two-layer memory + consolidation  →    semantic base + episodic LoRA
                                       + consolidation step (simplified)
dynamic wiring                    →    growing LoRA rank on capacity trigger
                                       (crude but demonstrates the idea)
no backprop                       →    NOT IN THIS POC
                                       still use backprop for LoRA updates
                                       (this is the concession)

THE CONCESSION EXPLAINED:

  dropping "no backprop" means:
  you're still using the freeze cycle
  BUT: only for the tiny LoRA adapters
       not for the full model
       
  freeze cycle on 300MB LoRA adapters:
  ~milliseconds
  inference barely interrupted
  
  vs freeze cycle on 6GB full model:
  seconds
  inference blocked
  
  this is not the full vision
  but it demonstrates:
    ✓ continuous adaptation from interactions
    ✓ memory that persists across conversations
    ✓ no catastrophic forgetting (base frozen)
    ✓ consolidation (patterns merge over time)
    ✓ capacity growth (new LoRA rank when full)
    ✓ runs on M1, uses unified memory properly
    ✓ inference and training simultaneously (effectively)

  the local learning signal is the PhD problem
  everything else is the engineering problem
```

---

## Week by week estimate

```
WEEK 1 — FOUNDATION (5-8 hours total)
─────────────────────────────────────
goal: base model running + LoRA adapter loading in MLX
      basic inference working
      nothing continuous yet, just the substrate

day 1 (1-2 hrs):
  Claude Code task:
  "set up MLX project, load Llama 3.2 3B,
   load a LoRA adapter, run inference,
   measure tokens/sec"
  
  expected output: working inference with adapter
  expected blockers: MLX version conflicts,
                     model download, memory fit
  
day 2 (1-2 hrs):
  Claude Code task:
  "add LoRA training loop in MLX,
   train on 10 example corrections,
   measure if adapter actually changed"
  
  expected output: adapter that updates
  expected blockers: MLX training API is less
                     documented than inference API

day 3-4 (2-3 hrs):
  Claude Code task:
  "implement double-buffer for LoRA:
   buffer A served during inference
   buffer B being trained
   atomic swap every 100 steps"
  
  expected output: inference never blocks on training
  expected blockers: this is the hardest part of week 1
                     thread safety, atomic operations
                     in Python/MLX is genuinely fiddly

day 5 (1 hr):
  measure and test
  does inference quality degrade during training?
  does the swap cause any visible artifacts?

WEEK 1 DELIVERABLE:
  a model that serves inference continuously
  while updating a LoRA adapter in the background
  this alone is novel enough to write about

─────────────────────────────────────────────────────

WEEK 2 — MEMORY ARCHITECTURE (5-8 hours total)
───────────────────────────────────────────────
goal: episodic buffer with persistence
      conversations remembered across sessions
      simple consolidation trigger

day 1 (1-2 hrs):
  Claude Code task:
  "add persistent episodic store:
   after each conversation, save:
   - what user corrected
   - what user preferred
   - what patterns repeated
   to a JSON file that survives restart"
  
  this is simpler than it sounds
  it's just structured logging + LoRA fine-tune triggers

day 2 (2 hrs):
  Claude Code task:
  "implement simple consolidation:
   every 50 interactions,
   fine-tune the base LoRA on the episodic store
   with selective weighting
   (recent corrections weighted higher)"
  
  expected blocker: what counts as "worth consolidating"
  simplest answer: everything in episodic store
  better answer: only things the user confirmed/repeated

day 3-4 (2-3 hrs):
  Claude Code task:
  "implement forgetting:
   if episodic store exceeds 500 items,
   score each item by:
   - recency (newer = higher score)
   - repetition (appeared multiple times = higher)
   - user confirmation (explicitly corrected = higher)
   prune lowest-scoring 20%"
  
  this is the consolidation quality metric
  it's a heuristic, not a mathematical solution
  but it's a real one

day 5 (1 hr):
  test: does the system remember things
        across conversation restarts?
  test: does the base model stay unaffected?

WEEK 2 DELIVERABLE:
  memory that persists
  consolidation that runs automatically
  selective forgetting
  truly a different kind of system than anything
  currently in mainstream use

─────────────────────────────────────────────────────

WEEK 3 — CRUDE DYNAMIC WIRING (5-8 hours total)
────────────────────────────────────────────────
goal: LoRA rank grows when episodic store saturates
      new capacity, not overwriting

day 1-2 (3 hrs):
  Claude Code task:
  "implement LoRA rank expansion:
   when episodic store consolidation runs,
   check: is the adapter close to capacity?
   (proxy: is loss on new corrections
    higher than it was 100 steps ago?)
   if yes: add a new LoRA layer with rank 4
   stack it with existing adapter"
  
  expected blocker: MLX doesn't natively support
                    stacking LoRA adapters dynamically
                    you'll need custom layer wrapper
  
  this is the hardest engineering in the whole POC

day 3 (2 hrs):
  Claude Code task:
  "memory management for growing adapters:
   on M1 with 16GB,
   semantic base: ~6GB
   episodic adapters: budget 2GB max
   if adapters exceed 2GB: trigger hard consolidation
   merge adapters into base (careful — this modifies base)
   or prune oldest adapter layers"
  
  this is where unified memory actually matters:
  you can watch memory usage in real time
  and the OS won't evict your model mid-inference

day 4-5 (2 hrs):
  test: does quality improve over time?
  test: does it degrade on old things?
  test: does memory stay bounded?

WEEK 3 DELIVERABLE:
  a system whose capacity grows with use
  rather than saturating and overwriting
  crude, but it proves the principle

─────────────────────────────────────────────────────

WEEK 4 — MEASUREMENT + WRITE-UP (4-5 hours total)
──────────────────────────────────────────────────
goal: actual numbers, honest evaluation, write-up

the three measurements that matter:

  MEASUREMENT 1: does it actually remember?
  
    test A: tell it a fact in conversation 1
            close everything, restart
            ask the same fact in conversation 2
            does it know?
    
    test B: correct it 5 times on same error
            does the correction stick?
            does it generalise to similar cases?

  MEASUREMENT 2: does old knowledge survive?
  
    test C: benchmark on standard tasks BEFORE any training
            run 100 interactions with personal corrections
            benchmark on SAME standard tasks
            has performance degraded?
    
    (this is the catastrophic forgetting test)

  MEASUREMENT 3: does inference stay continuous?
  
    test D: measure tokens/sec during idle
            measure tokens/sec while LoRA is updating
            what is the degradation?
            is it noticeable?

WEEK 4 DELIVERABLE:
  a table with actual numbers
  honest about what worked and what didn't
  this is the thing worth publishing
```

---

## Total time summary

```
             CODING TIME        THINKING TIME
             (Claude Code)      (you, architectural decisions)

Week 1       ~2 hours           ~4-6 hours (debugging, decisions)
Week 2       ~2 hours           ~4-6 hours
Week 3       ~3 hours           ~5-8 hours (hardest week)
Week 4       ~1 hour            ~3-4 hours (measurement design)
─────────────────────────────────────────────────────────────────
TOTAL        ~8 hours coding    ~16-24 hours thinking/debugging

CALENDAR TIME: 4-6 weeks at 2-3hrs/day
               OR 2-3 weeks if you have more focused time

THE RATIO IS IMPORTANT:
  Claude Code compresses coding to ~⅓ normal time
  it does NOT compress:
    debugging novel failure modes
    making architectural decisions
    figuring out what "consolidation quality" means
    measuring whether it actually works
    
  the thinking time is yours
  and it's most of the time
```

---

## The things that will go wrong (honest)

```
KNOWN UNKNOWNS:

1. MLX training stability
   MLX is optimised for inference on M1
   training support exists but is less mature
   you will hit undocumented edge cases
   
   mitigation: start with smallest possible model
               validate training works before building on it

2. Double-buffer race conditions
   Python threading + MLX memory management
   + atomic weight swaps is genuinely tricky
   you will probably get subtle corruption bugs
   before you get it right
   
   mitigation: add aggressive assertions
               log every swap
               test with memory monitoring

3. Consolidation quality
   the hardest non-engineering decision:
   what makes an episodic memory worth consolidating?
   your heuristic will be wrong at first
   
   mitigation: start with the simplest possible rule
               "consolidate everything"
               measure, then improve

4. The base model modification risk
   in week 3, if you merge adapters into base
   you are modifying the semantic layer
   this can catastrophically degrade it
   
   mitigation: ALWAYS keep a copy of original base
               test on standard benchmarks before and after
               if base degrades: you found the exact failure
               mode the architecture is meant to prevent

5. Memory pressure on 16GB
   base model 6GB
   + adapters ~2GB
   + KV cache (grows with context) ~1-3GB
   + MLX framework overhead ~1GB
   + OS ~2GB
   = 12-14GB in use
   leaves 2-4GB headroom
   
   this is tight
   long conversations will push you over
   
   mitigation: use 3B model not 7B
               limit context window
               monitor memory in real time (Activity Monitor)
```

---

## What success actually looks like

```
MINIMUM SUCCESS (4 weeks):
  
  a system where:
  - you correct it on something in session 1
  - you restart completely
  - it doesn't make the same mistake in session 2
  
  that's it
  that's genuinely new
  no local model does this today
  no consumer product does this today
  (Claude.ai's memory is cloud-side, not local, not in-weights)
  
  ONE DEMONSTRATION OF THIS:
  is worth more than a perfect architecture
  that doesn't work yet

MEDIUM SUCCESS (6 weeks):
  
  + it gets measurably better at your specific tasks
    over 2 weeks of use
  + base model quality doesn't degrade
  + memory stays bounded (doesn't grow forever)

FULL SUCCESS (8-10 weeks):
  
  + everything above
  + measurable numbers comparing
    baseline / standard RAG / this system
  + honest write-up of what worked, what didn't
  + the failure modes are documented
    (these are often more interesting than successes)

THE DEEPSEEK FRAME:
  DeepSeek's contribution was not a perfect system
  it was a rigorous experiment that proved
  an assumption wrong
  
  your contribution is the same:
  prove that memory architecture + continuous adaptation
  is achievable on consumer hardware
  with acceptable quality degradation
  
  or prove it isn't
  and document exactly where it breaks
  that is equally valuable
```

---

## Claude Code's actual role

```
WHAT CLAUDE CODE DOES WELL HERE:

  ✓ MLX boilerplate (model loading, tokenisation)
  ✓ LoRA implementation from scratch
  ✓ double-buffer threading logic
  ✓ episodic store (JSON persistence layer)
  ✓ consolidation loop skeleton
  ✓ measurement harness (benchmark scripts)
  ✓ debugging specific error messages
  ✓ refactoring when architecture changes
  ✓ writing tests

WHAT CLAUDE CODE CANNOT DO:

  ✗ decide what "consolidation quality" means
    (that's your architectural judgment)
  
  ✗ know if the approach is fundamentally flawed
    (that requires running it and observing)
  
  ✗ replace the moments where you watch it fail
    and understand WHY it failed
    (that is the actual research)
  
  ✗ write the write-up
    (it can draft, you make it true)

THE RIGHT WORKFLOW:

  you: "here's the architectural decision I made
       and why — implement this component"
  
  Claude Code: implements, often gets 80% right
  
  you: "this specific thing is wrong — fix it"
       (repeat 3-5 times per component)
  
  you: "now measure if it works"
  
  Claude Code: writes measurement scripts
  
  you: run them, observe actual behavior
       make next architectural decision
```

---

## The honest starting point for day 1

```
THE FIRST CLAUDE CODE PROMPT:

"I want to build a continuous local AI system on M1.
 The architecture has two layers:
 - a frozen 3B base model (semantic layer)
 - a small LoRA adapter that updates during use (episodic layer)
 
 Start with just getting this working:
 1. Load Llama-3.2-3B in MLX with a LoRA adapter
 2. Run inference through base + adapter
 3. Run a single fine-tuning step on the adapter
 4. Run inference again — adapter should have changed
 5. Measure tokens/sec before and after
 
 Use MLX-LM library. Keep it minimal.
 I want to understand each component before we add complexity."

that's it
that's day 1
one working loop
measure if it works

everything else follows from that
```
