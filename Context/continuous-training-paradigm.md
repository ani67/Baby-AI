# What's Fundamentally Wrong With How We Train AI
*And why M1 unified memory might be exactly the right hardware to fix it*

---

## The assumption nobody questions

Every optimizer ever built — vanilla SGD, Adam, AdamW, Muon, all of them — shares one structural assumption that predates neural networks entirely.

```
THE FREEZE-UPDATE CYCLE:

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │   1. FREEZE MODEL                               │
  │      (weights locked, no inference during this) │
  │                  │                              │
  │                  ▼                              │
  │   2. FORWARD PASS                               │
  │      input → prediction                         │
  │                  │                              │
  │                  ▼                              │
  │   3. COMPUTE LOSS                               │
  │      how wrong was the prediction?              │
  │                  │                              │
  │                  ▼                              │
  │   4. BACKWARD PASS                              │
  │      error signal flows back through all layers │
  │                  │                              │
  │                  ▼                              │
  │   5. UPDATE WEIGHTS                             │
  │      every weight nudges slightly               │
  │                  │                              │
  │                  ▼                              │
  │   6. UNFREEZE MODEL                             │
  │      now you can use it again                   │
  │                  │                              │
  │                  ▼                              │
  │      go back to step 1                          │
  │                                                 │
  └─────────────────────────────────────────────────┘

  Muon changes step 5 (better geometry for the update)
  Adam changes step 5 (better step size)
  AdaGrad changes step 5 (per-parameter rates)

  NOBODY QUESTIONS STEPS 1-4
  NOBODY QUESTIONS THE FREEZE ITSELF
```

This cycle was never a law of nature. It was a convenience that became a paradigm.

Backpropagation — step 4 — *requires* it. You cannot run backprop on a moving target. The backward pass needs the exact state of every layer from the forward pass to compute the gradient. If weights change during the backward pass, the math breaks.

So backprop demands frozen weights. And frozen weights means: training and inference cannot happen simultaneously.

---

## What biology actually does

```
A BIOLOGICAL NEURON:

  ─── receives inputs continuously ───────────────────►
      │
      │  fires (or doesn't) based on current weights
      │  AND simultaneously
      │  adjusts synaptic weights based on local activity
      │
  ─── produces outputs continuously ──────────────────►

  there is no step 1 (freeze)
  there is no step 6 (unfreeze)
  
  learning and inference are not sequential
  they are the SAME process
  happening at the same time
  in the same substrate

THE HEBBIAN RULE (1949):
  "neurons that fire together, wire together"
  
  if neuron A is active when neuron B is active
  the connection between them strengthens
  
  this happens:
    locally (no global signal needed)
    continuously (while the neuron is processing)
    without freezing anything
    without a separate backward pass

THE WEIGHT UPDATE IS LOCAL:

  standard backprop:
    weight update requires knowing the FINAL output
    and propagating error ALL THE WAY BACK
    96 layers back
    through the entire network
    before any weight can change

  Hebbian / biological:
    weight update requires knowing only:
    what was MY input?
    what was MY output?
    that's it
    no global signal
    no backward pass
    no freeze

THIS IS THE FUNDAMENTAL DIFFERENCE:

  ┌──────────────────┬───────────────────────────────┐
  │                  │ BACKPROP        │ BIOLOGY      │
  ├──────────────────┼─────────────────┼──────────────┤
  │ learning signal  │ global          │ local        │
  │ direction        │ backward        │ none needed  │
  │ timing           │ after inference │ during       │
  │ freeze needed?   │ yes             │ no           │
  │ continuous?      │ no              │ yes          │
  │ uses output?     │ needs final out │ uses local   │
  │                  │ to backprop     │ activity only│
  └──────────────────┴─────────────────┴──────────────┘
```

---

## Why the freeze cycle exists (the real reason)

It's not just backprop. It's where backprop came from: hardware.

```
NVIDIA GPU SETUP:

  ┌──────────────────────────────────────────────────┐
  │   SYSTEM RAM (CPU)                               │
  │   ████████████████████████  64GB                 │
  │   (data loading, preprocessing)                  │
  └─────────────────┬────────────────────────────────┘
                    │ PCIe bus (bottleneck)
                    │ data must be explicitly transferred
                    ▼
  ┌──────────────────────────────────────────────────┐
  │   VRAM (GPU)                                     │
  │   ████████████  24GB                             │
  │   (where training happens)                       │
  │   (SEPARATE physical memory)                     │
  └──────────────────────────────────────────────────┘

  CONSEQUENCE:
  during training:   weights live in VRAM
  during inference:  weights live in VRAM
  
  but you can't train AND serve simultaneously
  on the same VRAM because:
  - training needs: weights + gradients + optimizer states
    = 3× model size minimum
  - inference needs: weights only = 1× model size
  - together: 4× model size
  - often exceeds VRAM
  
  so production systems run SEPARATE instances:
  one model for inference (frozen)
  one model for training (updating)
  periodically syncing weights

THE BATCH PARADIGM CAME FROM THIS:

  if you have to transfer data over PCIe
  (slow bottleneck between CPU and GPU)
  you want to transfer a lot at once
  
  batch = "gather 32/64/128/256 examples
           transfer them all at once
           train on all of them in parallel"
  
  the batch is a hardware optimization
  that became a conceptual assumption
  
  "you train on batches" is not a law
  it's a workaround for PCIe bandwidth
  that calcified into orthodoxy
```

### M1 breaks this constraint

```
M1 PRO / M1 MAX / M1 ULTRA:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │              UNIFIED MEMORY                      │
  │              16GB / 32GB / 64GB                  │
  │                                                  │
  │  ┌─────────┐     ┌─────────┐     ┌─────────┐    │
  │  │  CPU    │     │  GPU    │     │  Neural  │    │
  │  │  cores  │     │  cores  │     │  Engine  │    │
  │  └────┬────┘     └────┬────┘     └────┬─────┘    │
  │       │               │               │           │
  │       └───────────────┴───────────────┘           │
  │                       │                           │
  │              SAME PHYSICAL MEMORY                 │
  │              all chips read/write here            │
  │              no transfer bottleneck               │
  │              no separate VRAM                     │
  │                                                   │
  └───────────────────────────────────────────────────┘

  IMPLICATION:
  there is no hardware reason to separate
  training memory from inference memory
  
  they ARE the same memory pool
  already
  by default
  
  a model being trained and a model serving inference
  could literally be the same weight matrix
  in the same memory location
  being read and written simultaneously
  
  the freeze cycle is a SOFTWARE assumption
  carried over from a hardware constraint
  THAT DOESN'T EXIST ON M1

  M1 is the first consumer hardware where
  continuous training is not hardware-blocked
  it's only paradigm-blocked
```

---

## What continuous training would actually require

The freeze cycle exists because backprop requires it. So continuous training requires something other than backprop.

There are four real candidates. They exist. They're just not mainstream.

---

### CANDIDATE 1: Local Learning Rules (Hebbian / STDP)

```
THE IDEA:
  each weight updates based only on
  the activity of the neurons it connects
  no global backward pass
  no freeze
  
  SPIKE-TIMING DEPENDENT PLASTICITY (STDP):
  
  if neuron A fires BEFORE neuron B:
    the A→B connection strengthens
    ("A caused B")
    
  if neuron A fires AFTER neuron B:
    the A→B connection weakens
    ("A did not cause B")
  
  update rule is:
    local (only A and B)
    continuous (happens whenever they fire)
    causal (strengthens predictive connections)
  
  ┌────────────────────────────────────────────────┐
  │ PROS:                                          │
  │   no freeze cycle                              │
  │   naturally continuous                         │
  │   biologically plausible                       │
  │   low memory overhead                          │
  │   can run during inference                     │
  │                                                │
  │ CONS:                                          │
  │   much weaker learning signal than backprop    │
  │   hard to train deep networks                  │
  │   doesn't scale to transformers yet            │
  │   still largely a research primitive           │
  └────────────────────────────────────────────────┘

  STATE: interesting, not ready for production
         active research at DeepMind, Numenta
```

---

### CANDIDATE 2: Predictive Coding / Active Inference

```
KARL FRISTON — free energy principle
  theoretical neuroscientist
  University College London

THE IDEA:
  instead of one big top-down backward pass
  every layer makes a PREDICTION
  about what the layer below it will show
  
  the ERROR between prediction and reality
  is the local learning signal
  
  ARCHITECTURE:
  
  layer 3 ──────────────────────────►
           │                         │
           │ predicts what layer 2   │
           │ "should" look like      │
           ▼                         │
  layer 2 ──prediction───►[compare]─►error
           │                         │
           │ predicts what layer 1   │ updates weights
           │ "should" look like      │ based on error
           ▼                         │
  layer 1 ──prediction───►[compare]─►error
           │                         │
  input ───┘                updates weights
  
  KEY DIFFERENCE FROM BACKPROP:
  
  backprop: wait for final output
            propagate error globally backward
            requires frozen weights
            
  predictive coding: each layer has its own
                     local error signal
                     updates continuously
                     from its own prediction error
                     no global backward pass
                     no freeze needed
  
  ┌────────────────────────────────────────────────┐
  │ PROS:                                          │
  │   no freeze cycle                              │
  │   naturally continuous                         │
  │   theoretically well-grounded                  │
  │   inference IS learning (same process)         │
  │   matches biological evidence closely          │
  │                                                │
  │ CONS:                                          │
  │   slower convergence than backprop             │
  │   harder to implement                          │
  │   no mature ML framework supports it           │
  │   compute overhead per layer                   │
  └────────────────────────────────────────────────┘

  STATE: theoretically very promising
         Friston's group + a few others
         not yet at transformer scale
         
  CONNECTION TO YOUR INSIGHTS:
  "building intuition not storage" from POC 1
  is what predictive coding naturally does —
  each layer builds a model of what to expect
  not a lookup table of what it saw
```

---

### CANDIDATE 3: Forward-Forward Algorithm

```
HINTON — 2022
  Geoffrey Hinton (yes, the backprop Hinton)
  after leaving Google
  
  "The Forward-Forward Algorithm:
   Some Preliminary Investigations"

THE IDEA:
  instead of: forward pass + backward pass
  do: two forward passes
  
  PASS 1 (positive):  real data → maximize "goodness"
  PASS 2 (negative):  fake data → minimize "goodness"
  
  each layer has its own "goodness" score
  each layer updates its weights locally
  to maximize goodness on real / minimize on fake
  
  no backward pass
  no global error propagation
  no freeze required (in principle)
  
  ┌────────────────────────────────────────────────┐
  │                                                │
  │  LAYER 1:  sees real data                      │
  │            computes goodness                   │
  │            updates weights to maximize it      │
  │            passes result to layer 2            │
  │                                                │
  │  LAYER 2:  sees layer 1 output                 │
  │            computes its own goodness           │
  │            updates its own weights             │
  │            passes to layer 3                   │
  │                                                │
  │  each layer is a small independent learner     │
  │  no signal needs to travel backward            │
  │                                                │
  └────────────────────────────────────────────────┘
  
  WHY THIS IS INTERESTING FOR M1:
  
  the two forward passes can run simultaneously
  on M1's separate CPU and GPU cores
  (unified memory means both see the same weights)
  
  while GPU runs positive pass (real data)
  CPU runs negative pass (fake/noisy data)
  both update the same weight matrix in shared memory
  
  this is architecturally possible on M1
  in a way that's awkward on NVIDIA
  
  ┌────────────────────────────────────────────────┐
  │ PROS:                                          │
  │   potentially no freeze cycle                  │
  │   Hinton's name means people take it seriously │
  │   parallelizable across M1's compute units     │
  │   still early — high research upside           │
  │                                                │
  │ CONS:                                          │
  │   underperforms backprop on standard tasks     │
  │   still experimental                           │
  │   "goodness" function is handwavy              │
  │   nobody has scaled it to LLM size yet         │
  └────────────────────────────────────────────────┘

  STATE: 2 years old, still mostly unexplored
         Hinton intended it as "here's a direction"
         not "here's a production system"
```

---

### CANDIDATE 4: LoRA Continuous Adaptation (practical now)

```
THIS ONE CAN ACTUALLY BE BUILT TODAY
ON YOUR M1

THE IDEA:
  base model weights = FROZEN FOREVER
  (they never change after initial download)
  
  LoRA adapters = thin layers on top
  (tiny, ~1-5% of model size)
  (these update continuously)
  
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  BASE MODEL (frozen, 6GB for 7B)                   │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  all inference reads from here                     │
  │  never changes                                     │
  │  always safe to query                              │
  │                                                    │
  │         +                                          │
  │                                                    │
  │  LORA ADAPTERS (continuously updating, 0.3GB)      │
  │  ████ ← can be updated while inference runs        │
  │  (small enough to write atomically)                │
  │  (inference reads old version while               │
  │   new version is being computed)                   │
  │                                                    │
  │  combined output = base + adapter(input)           │
  │                                                    │
  └────────────────────────────────────────────────────┘

  WHY THIS WORKS FOR SIMULTANEOUS TRAIN+INFERENCE:
  
  reading the frozen base weights = safe always
  
  writing new LoRA adapters = small operation
  can be done with double-buffering:
  
    BUFFER A: current adapters (inference reads this)
    BUFFER B: being trained (gradient descent here)
    
    every N steps: atomic swap A↔B
    inference barely notices (one inference step
    uses old adapters, then switches to new)
  
  THIS IS NOT PERFECT CONTINUOUS LEARNING
  but it is:
    practically buildable today
    works on M1 (unified memory helps)
    continuous enough for most purposes
    no catastrophic forgetting of base
    only the adapter updates

  ┌────────────────────────────────────────────────┐
  │ WEEK 1 BUILDABLE:                              │
  │                                                │
  │ 3B base model frozen in unified memory         │
  │ LoRA adapter being trained on feedback         │
  │ inference serving from base + current adapter  │
  │ double-buffered adapter swap every 100 steps   │
  │ training signal = corrections to wrong answers │
  │                                                │
  │ this is a primitive continuous learner         │
  │ crude but real                                 │
  └────────────────────────────────────────────────┘
```

---

## The mathematical reason this is all hard

```
CATASTROPHIC FORGETTING (the real enemy):

  every continuous learning approach
  hits the same wall
  
  IF you keep training on new things:
  
  step 1:  model learns A        ████ (A strongly encoded)
  step 2:  model learns B        ████ (B overwrites some of A)
  step 3:  model learns C        ████ (C overwrites some of A and B)
  step N:  model has mostly      ░░░░ (A and B nearly gone)
           only recent stuff
  
  the gradients for new things
  overwrite the weights for old things
  because the weights are SHARED
  
  this doesn't happen in biology because:
  
  new memories → hippocampus (temporary, fast)
  old memories → neocortex (consolidated, slow-changing)
  
  the CONSOLIDATION happens during sleep
  hippocampus replays memories to neocortex
  neocortex updates slowly
  doesn't overwrite old structure
  
  ┌──────────────────────────────────────────────────┐
  │ THE BIOLOGICAL SOLUTION TO CATASTROPHIC          │
  │ FORGETTING IS ARCHITECTURAL:                     │
  │                                                  │
  │ fast-changing system (hippocampus / LoRA):       │
  │   learns new things quickly                      │
  │   can be overwritten                             │
  │   temporary                                      │
  │                                                  │
  │ slow-changing system (neocortex / base model):   │
  │   accumulates consolidated knowledge             │
  │   updates very slowly                            │
  │   not overwritten by new experiences             │
  │                                                  │
  │ LoRA continuous adaptation IS this architecture  │
  │ (accidentally, not by design)                    │
  │                                                  │
  │ frozen base = neocortex                          │
  │ updating adapters = hippocampus                  │
  │                                                  │
  │ you still need a "sleep" step:                   │
  │ periodically merge adapters into base model      │
  │ = consolidation                                  │
  └──────────────────────────────────────────────────┘
```

---

## The unified picture of what's wrong

```
CURRENT AI TRAINING                 WHAT WOULD ACTUALLY WORK

assumes:                            needs:
  separate train/inference            simultaneous train/inference
  global backward pass                local learning signals
  frozen weights during learning      weights always live
  batch updates (hardware legacy)     continuous updates
  flat parameter space                manifold-aware updates
  learning from data only             learning from its own operation

the MUON fix addresses:             none of these other things
  flat parameter space (partially)

the ADAM fix addressed:             none of these other things
  per-parameter step sizes

NOBODY HAS ADDRESSED:
  the freeze cycle
  the global backward pass requirement
  the separation of train and inference
  because these are built into backprop
  and backprop is built into everything

THE MATHEMATICAL FRAMING:

  backprop requires: a static computational graph
                     with frozen weights
                     to apply the chain rule
  
  the chain rule (Leibniz 1670s) requires: 
                     each step in the chain fixed
                     before you multiply through
  
  continuous learning requires:
                     the chain to be changing
                     while you're computing through it
  
  these are mathematically incompatible
  
  to do continuous learning
  you need a learning rule
  that does NOT require a static chain
  
  that is: Hebbian rules, predictive coding,
           forward-forward, or LoRA double-buffering
  
  all four exist
  none are mature
  the field is essentially ignoring this
  because backprop works well enough
  for the batch paradigm
  that everyone already has infrastructure for
```

---

## What's specifically buildable on M1 now

```
THE M1 ADVANTAGE MAP:

feature                    M1 advantage    connection to continuous learning
──────────────────────────────────────────────────────────────────────────
unified memory             HIGH            no train/inference memory split
                                           train and serve from same pool

CPU + GPU + Neural Engine  HIGH            run different tasks simultaneously
on same die                                positive pass on GPU
                                           negative pass on CPU
                                           (Forward-Forward architecture)

memory bandwidth           HIGH            fast weight reads for inference
(400GB/s on M1 Max)                        fast weight writes for training
                                           same pool, no PCIe bottleneck

no forced batch size        MEDIUM         can update on single examples
(no PCIe to optimize)                      truly online learning possible

MLX framework               MEDIUM         Apple's ML framework
                                           optimized for unified memory
                                           supports training + inference
                                           in same process

─────────────────────────────────────────────────────────────────────────

BUILD ORDER (least to most novel):

LEVEL 1 (weeks): LoRA continuous adaptation
  frozen 3B base + updating adapters
  double-buffered swap
  train from corrections while serving
  
LEVEL 2 (months): Forward-Forward exploration
  implement Hinton's 2022 algorithm on MLX
  use CPU+GPU split for positive/negative passes
  measure: does this actually work for small models?
  
LEVEL 3 (research): Predictive coding layer
  replace one transformer layer with predictive coding
  does the local error signal work?
  does it learn without backprop?
  
LEVEL 4 (novel): Full continuous architecture
  no backprop at all
  local learning rules throughout
  genuinely simultaneous train + inference
  would be genuinely new
```

---

## The honest framing

```
WHAT MUON IS:
  a better shovel
  digging in the same direction
  everyone else is digging

WHAT YOU'RE POINTING AT:
  a question about whether we're digging
  in the right place at all

THE RIGHT QUESTION IS NOT:
  "what is the best optimizer?"
  
THE RIGHT QUESTION IS:
  "does training need to be separate from inference?"
  "does learning need a global backward pass?"
  "does the freeze cycle serve the problem
   or just the infrastructure we inherited?"

DEEPSEEK'S INSIGHT WAS:
  the assumption (you need massive compute)
  was wrong

YOUR INSIGHT IS:
  the assumption (you need to freeze to learn)
  might also be wrong

DIFFERENCE:
  DeepSeek could test their insight with existing tools
  Your insight requires different tools to test
  
  which is harder
  and more interesting
  and more worth doing
```
