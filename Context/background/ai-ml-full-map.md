# AI/ML — The Full Illustrated Map
*Everything we covered, compressed with ASCII diagrams. Built bottom-up from first principles.*

---

## 1. What a Neural Network Actually Is

A function with billions of knobs. Text in, prediction out. Knobs start random. Training sets them.

```
SIMPLEST POSSIBLE NEURAL NETWORK:

input        hidden layer       output
                                
"3"    ──►  [neuron]  ──►  [neuron]  ──►  "6"
              × 0.5            × 4
              
each number on the arrows = a weight (a knob)
the network learned: multiply by 2
by adjusting those knobs through training

DEEP NETWORK (what LLMs actually are):

input    layer1   layer2   layer3  ... layer96   output

token ──► [■■■] ──► [■■■] ──► [■■■] ──► [■■■] ──► next token
          ↕↕↕       ↕↕↕       ↕↕↕       ↕↕↕
        weights   weights   weights   weights
        
        each ■ = one neuron
        each ↕ = one weight (parameter)
        a 7B model has 7 billion of these weights
```

---

## 2. Gradient Descent — The Universal Training Algorithm

Invented 1847. Backpropagation formalised 1986. Still powering everything in 2025.

```
THE LANDSCAPE:

error
  │
  │  START HERE (random weights = maximum confusion)
  │  *
  │   *
  │    *
  │     **
  │       ***
  │          ****
  │              ********
  │                      ***************──── GOAL (minimum error)
  └──────────────────────────────────────────────► training steps

THE BLINDFOLDED PERSON ANALOGY:

  you are blindfolded on hilly terrain
  goal = find the lowest point (valley)
  
  what you can do:
    feel which way the ground slopes
    take one small step downhill
    feel again
    step again
    repeat
  
  that's gradient descent
  
  gradient = which way is downhill
  descent  = taking the step

ZOOMED IN — it's not smooth:

error
  │     *  *
  │   *      *    *
  │  *          *    *  *
  │ *               *      * *
  │*                           * * *  * *
  └──────────────────────────────────────►
  
  each * = one batch update
  smooth curve is an illusion of zooming out
  the wobble is actually useful —
  helps escape local traps

THE TRAP PROBLEM:

        ╭──╮           ╭──╮
       ╭╯  ╰──╮     ╭──╯  ╰╮
      ╭╯       ╰─────╯      ╰╮
     ╭╯    *                  ╰╮
    ╭╯     │                   ╰╮
   ╭╯      ↓                    ╰╮
  ─╯   local min            global min
             (trap)         (want this)
```

### The Training Loop

```
repeat billions of times:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. FORWARD PASS                                    │
│     input tokens → through all layers → prediction │
│                                                     │
│  2. LOSS                                            │
│     how wrong was the prediction?                   │
│     correct answer = "mat"                          │
│     model said    = "refrigerator"                  │
│     loss          = very high                       │
│                                                     │
│  3. BACKWARD PASS                                   │
│     gradient flows back through every layer        │
│     each weight gets: nudge this way by this much  │
│                                                     │
│  4. UPDATE                                          │
│     every weight moves slightly toward              │
│     "refrigerator less likely, mat more likely"    │
│                                                     │
└─────────────────────────────────────────────────────┘

forward pass  = water flowing down
backward pass = signal flowing back up
```

### Why Adam Optimizer Matters

```
VANILLA GRADIENT DESCENT:
  step = gradient × fixed_learning_rate
  every parameter treated the same
  learning rate is a guess

ADAM (2014):
  tracks gradient history per parameter
  
  parameter changing direction often:
    ──► bumpy terrain ──► take smaller steps
    
  parameter consistently same direction:
    ──► smooth slope  ──► take larger steps

  this is a cheap approximation
  of "seeing" the landscape shape
  better than fully blind
  not fully sighted
```

### Your Question: What If the Blindfolded Person Could SEE?

```
CURRENT (first order — feel the slope):

  gradient tells you:
    which direction is downhill
    RIGHT NOW at this exact point
  
  result: tiny steps, billions of them

SIGHTED (second order — see the curvature):

  hessian tells you:
    which direction is downhill
    AND how the slope is changing
    AND where the valley roughly is
  
  result: bigger informed jumps, fewer steps

WHY WE DON'T DO THIS:

  gradient  = 1 number per parameter
              175B params = 175B numbers ✓ manageable

  hessian   = 1 number per PAIR of parameters
              175B × 175B = 30,000,000,000,000,000,000 numbers
              thirty quintillion
              not storable
              not computable

META-LEARNING (the deeper version you were pointing at):

  instead of better gradient information
  train a model to REASON about what it's learning
  
  not: react to gradient signal
  but: understand the structure of the problem
       navigate toward solution intelligently
  
  exists in research (MAML)
  hasn't scaled to frontier models yet
  requires solving interpretability first —
  to reason about what you're learning
  you need to read what you've already learned
```

---

## 3. The History of Architectures — What Got Thrown Out

```
TIMELINE:

1986        1997        2014        2015        2017        2020+
  │           │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼           ▼
 RNN        LSTM      Seq2Seq      ResNets    Transformer   Hybrids
            GRU       +Attention   CNNs on    "Attention    Mamba
                                   text       is all        RWKV
                                              you need"     MoE
```

### RNNs — The First Approach

```
input:  "The"    "cat"    "sat"    "on"     "the"    "mat"
         │        │        │        │        │        │
         ▼        ▼        ▼        ▼        ▼        ▼
        [A]─────►[A]─────►[A]─────►[A]─────►[A]─────►[A]
                                                       │
                                                    output

A = same cell, reused at every step
──► = hidden state (the "memory") passed forward

THE PROBLEM:

hidden state is fixed size
doesn't matter if sequence is 10 or 10,000 words

"The" gets progressively overwritten
by the time you reach "mat"
the beginning is faint or gone

this is the VANISHING GRADIENT problem:
training signal trying to reach early layers
gets weaker at every step backward

  gradient at step 6:  0.9   strong
  gradient at step 4:  0.4   weaker
  gradient at step 2:  0.1   faint
  gradient at step 1:  0.001 nearly gone

early layers barely learn
```

### LSTMs — The First Fix

```
input:  "The"    "cat"    "sat"    "on"     "the"    "mat"
         │        │        │        │        │        │
         ▼        ▼        ▼        ▼        ▼        ▼
        [A]─────►[A]─────►[A]─────►[A]─────►[A]─────►[A]
         ║        ║        ║        ║        ║        ║
    ═════════════════════════════════════════════════════►
              cell state (long term memory highway)

INSIDE EACH LSTM CELL:

    ┌─────────────────────────────────────┐
    │                                     │
    │  FORGET GATE  ──► erase old memory? │
    │                                     │
    │  INPUT GATE   ──► write new memory? │
    │                                     │
    │  OUTPUT GATE  ──► what to pass on?  │
    │                                     │
    └─────────────────────────────────────┘

upgraded from notepad to filing system with folders
can choose what to keep, toss, or look up later

POWERED:
  Google Translate (before 2017)
  Siri speech recognition
  Gmail Smart Reply
  Early text generation

STILL SEQUENTIAL — couldn't parallelise
training on massive datasets = painfully slow
```

### GRUs

```
Simpler version of LSTM
fewer gates, slightly faster, similar quality
some people used these instead
also thrown out by transformers
```

### CNNs on Text

```
"The"  "cat"  "sat"  "on"  "the"  "mat"

filter slides across windows of 3 words:

step 1: ["The" "cat" "sat"] ──► feature detected
step 2:       ["cat" "sat" "on"] ──► feature detected
step 3:             ["sat" "on" "the"] ──► feature detected
step 4:                   ["on" "the" "mat"] ──► feature detected

then pool features ──► compressed representation

FAST because no sequential dependency
BAD at long range — "The" can never directly see "mat"
local window only

Facebook Research pushed this ~2014-2016
largely abandoned after transformers
```

### Seq2Seq — Direct Ancestor of Attention

```
TRANSLATION TASK: English → French

ENCODER (reads input):

"The" "cat" "sat" ──► [RNN] ──► [RNN] ──► [RNN] ──► [single vector]
                                                      compressed summary
                                                      of entire sentence

DECODER (generates output):

[single vector] ──► [RNN] ──► [RNN] ──► [RNN]
                     │         │         │
                    "Le"      "chat"    "est assis"

THE BOTTLENECK PROBLEM:

compressing "The quick brown fox jumped over 
the lazy dog near the river bank at sunset
while the birds were singing" 
into ONE vector = catastrophic information loss

THIS IS EXACTLY WHAT MOTIVATED ATTENTION:

researchers said:
  what if instead of one compressed vector
  the decoder could look back at ALL
  the encoder's hidden states
  and pick which ones are relevant
  at each step?

first attention mechanism (2014-2015):
  still built on top of RNNs
  but the seed of what would become transformers
```

---

## 4. The Transformer — What Replaced Everything

```
"Attention Is All You Need" — 2017
title was deliberately provocative
saying: throw away the RNN scaffolding entirely

KEY INSIGHT:
  attention doesn't need sequential processing
  every position can attend to every other position
  SIMULTANEOUSLY
  
  this means: GPUs can be fully utilised
  GPUs are designed for parallel computation
  RNNs wasted this (sequential dependency)
  transformers exploited it completely

TRANSFORMER STRUCTURE:

         INPUT TOKENS
         [the] [cat] [sat]
           │     │     │
           ▼     ▼     ▼
        ┌─────────────────────────────┐
        │        EMBEDDING            │  tokens → vectors
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │         ATTENTION           │  tokens look at each other
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  ADD & NORMALISE            │  stabilise signal
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │       FEEDFORWARD           │  process each token
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  ADD & NORMALISE            │
        └──────────────┬──────────────┘
                       │
                (repeat 96 times)
                       │
        ┌──────────────▼──────────────┐
        │        OUTPUT HEAD          │  → probability over all tokens
        └─────────────────────────────┘
```

### Attention — The Core Idea

```
EVERY TOKEN TALKS TO EVERY OTHER TOKEN SIMULTANEOUSLY:

input:  "The"  "cat"  "sat"  "on"  "the"  "mat"
          │      │      │      │      │      │
          ▼      ▼      ▼      ▼      ▼      ▼
        [tok]  [tok]  [tok]  [tok]  [tok]  [tok]
          │╲    │╲    │╲    │╲    │╲    │
          │ ╲   │ ╲   │ ╲   │ ╲   │ ╲   │
          │  ╲──┼──╲──┼──╲──┼──╲──┼──╲──┤
          │   ╲─┼───╲─┼───╲─┼───╲─┼───╲─┤
          ▼    ▼▼    ▼▼    ▼▼    ▼▼    ▼▼
        [tok]  [tok]  [tok]  [tok]  [tok]  [tok]

every token has a direct line to every other token
no left-to-right constraint
all at once

ATTENTION WEIGHTS — what "it" attends to:

sentence: "The animal didn't cross the street because it was tired"

        The  animal  cross  street  because  tired
   it  [0.02  0.71   0.01   0.12    0.05    0.09]

visualised:
        The  animal  cross  street  because  tired
   it  │░░░  ████   ░░░░   ███░    ░░░░    ██░░│

████ = high attention (animal)
███░ = medium attention (street)
░░░░ = low attention

model learned: "it" refers to "animal" not "street"
because "tired" collocates with animals in training data
nobody told it this — it emerged
```

### The Cost of Attention

```
ATTENTION SCALING:

sequence length 100 tokens:
  attention computations = 100 × 100 = 10,000

sequence length 1,000 tokens:
  attention computations = 1,000 × 1,000 = 1,000,000

sequence length 10,000 tokens:
  attention computations = 10,000 × 10,000 = 100,000,000

this is O(n²) — quadratic scaling
double the context = quadruple the compute
this is the fundamental expense
baked in at the architectural level
not an implementation detail you can fix
```

### What Came After Transformers

```
CHALLENGERS TO FULL ATTENTION:

MAMBA (SSMs — State Space Models):
  replaces attention with linear-scaling state
  O(n) not O(n²)
  cheaper but less good at long-range recall
  
  ┌────────────────────────────────┐
  │ transformer: every token       │
  │ looks at every other token     │
  │ cost: n²                       │
  │                                │
  │ mamba: running state updates   │
  │ token by token                 │
  │ cost: n                        │
  └────────────────────────────────┘

RWKV:
  recurrent at inference (constant memory)
  transformer-like at training (parallelisable)
  best of both worlds in theory

MoE (Mixture of Experts):
  model has many "expert" subnetworks
  each token routed to only a few experts
  
  ┌─────────────────────────────────────┐
  │ standard transformer:               │
  │ every token → every parameter       │
  │ wasteful — most params irrelevant   │
  │                                     │
  │ MoE:                                │
  │ every token → top 2-6 experts only  │
  │ large on disk, small at runtime     │
  │ Qwen 3 30B behaves like 8B          │
  └─────────────────────────────────────┘

HYBRIDS (where frontier is heading):
  attention where it matters (complex relationships)
  cheaper operations where it doesn't
  not pure transformer, not transformer-free
```

---

## 5. From Text to Vectors — The Universal Language

```
EVERYTHING BECOMES NUMBERS:

TEXT:
  "Paris" ──► token ID 15238
          ──► vector [0.2, -0.8, 0.4, 0.1, 0.9, -0.3 ...]
                      768 to 12,288 numbers
                      
  the vector encodes MEANING, not just identity
  similar concepts → nearby vectors
  
  king  - man  + woman  ≈  queen
  [vec]   [vec]  [vec]     [vec]
  
  this geometry emerges from training
  nobody designed it

IMAGES:
  pixel grid ──► numbers
  
  grayscale 100×100 = 10,000 numbers (0-255 each)
  colour 100×100    = 30,000 numbers (R, G, B each)
  
  raw pixels fed directly = bad (too much noise)
  
  CNN approach:
    slide filter across image
    detect edges → shapes → parts → objects
    layer by layer, local to global
    
  ViT approach (vision transformer):
    cut image into patches (like tokens)
    each patch → vector
    run attention across patches
    patch of sky attends to other sky patches
    patch with eye attends to patch with nose

VIDEO:
  video = images over time
  10 seconds at 24fps = 240 images
  
  hard problem = temporal coherence
  frame 47 must follow naturally from frame 46
  
  solution: attention across frames AND space
  
  cost: O(n²) across patches × frames
  this is why video models are so expensive

THE SHARED VECTOR SPACE (CLIP):

  photo of dog ────────────────────► [0.2, -0.3, 0.9...]
                                            ▲
                                     same region
                                            ▼
  words "a dog" ──────────────────► [0.2, -0.4, 0.8...]
  
  this shared space is what enables
  text-controlled image generation
  the prompt steers the image
  because they speak the same vector language
```

### How Diffusion (Image Generation) Works

```
TRAINING:

real image ──► add noise ──► add noise ──► ... ──► pure static
     │                                               │
     └──────── train model to reverse this ──────────┘
                "given noisy image, predict less noisy"

GENERATION:

pure noise ──► denoise ──► denoise ──► ... ──► image

text prompt "red cat on mat"
     │
     ▼
CLIP encodes to vector
     │
     ▼
steers denoising toward
red + cat-shaped + mat-textured structures
```

### Your Question: Does Text Hold More Info Than Images?

```
TEXT is denser for:             IMAGE is denser for:

abstract concepts               spatial relationships
logical relationships           texture, material, light
causality                       emotion, expression
temporal sequences              scene composition
precise facts                   things language hasn't named
instructions

"the battle caused economic      a photograph of a face contains:
 collapse which led to           age, mood, health, micro-expressions,
 revolution"                     lighting direction, skin texture...
                                 most of which has no text equivalent

TEXT IS LOSSY COMPRESSION OF REALITY:

reality ──► humans observe ──► humans name things ──► text
           (saw everything)   (kept only what      (model learns
                               was worth naming)    from this)

two lossy compressions between model and world
which is why LLMs fail at:
  spatial reasoning
  physical intuition
  "which cup is bigger"
  things a 3-year-old handles trivially

IMPLICATION:
  text-as-images ──► mostly redundant, adds steps
  the right direction ──► learn from reality directly
  Genie, robotics, embodied AI are all this question
```

---

## 6. Genie — A Different Category Entirely

```
VIDEO GENERATION:         WORLD MODEL (Genie):

prompt ──► video clip     prompt ──► environment you live in
you watch it              you walk around it
it's done                 it responds to what you do
                          it generates what's AHEAD OF YOU
                          as you move

GENIE LINEAGE:

Genie 1 (early 2024):
  simple 2D interactive environments
  rough but proved it was possible

Genie 2 (Dec 2024):
  3D worlds from a single image
  consistent for ~1 minute
  responds to keyboard/mouse

Genie 3 (Aug 2025):
  24fps real-time
  720p
  consistent for several minutes
  text prompt → walkable world

WHAT'S HAPPENING INSIDE:

  standard video model:
    predict what comes next
    in a passive sequence
    
  Genie adds:
    predict what comes next
    GIVEN WHAT THE USER DID
    
  ┌────────────────────────────────────────┐
  │  frame N                               │
  │  + user action (pressed forward key)  │
  │       │                               │
  │       ▼                               │
  │  frame N+1 (world moved toward you)   │
  └────────────────────────────────────────┘
  
  learned cause-and-effect entirely from
  watching gameplay footage
  nobody told it that arrow keys move character
  it inferred this from millions of examples

WHY DEEPMIND IS BUILDING THIS:
  not primarily for entertainment
  for training AI agents
  
  generate unlimited varied training environments
  from text prompts
  train robots/agents in simulated worlds
  before deploying in real world
  
  Genie 3 → stepping stone toward AGI
  world models = prerequisite for real-world AI
```

---

## 7. Model Size, RAM, Compute

### The Formula

```
MODEL SIZE:

parameter count × bits per parameter ÷ 8 = bytes

PRECISION TABLE:

┌──────────┬──────────────┬────────────┬────────────┐
│ Format   │ Bits/param   │ 7B model   │ 70B model  │
├──────────┼──────────────┼────────────┼────────────┤
│ FP32     │ 32           │ ~28 GB     │ ~280 GB    │
│ FP16     │ 16           │ ~14 GB     │ ~140 GB    │
│ Q8       │ 8            │ ~7 GB      │ ~70 GB     │
│ Q4       │ 4            │ ~4 GB      │ ~40 GB     │
│ Q2       │ 2            │ ~2 GB      │ ~20 GB     │
└──────────┴──────────────┴────────────┴────────────┘

Q4 is the sweet spot
4x smaller than FP16
minimal quality loss

RUNTIME RAM = model size + 20-30% overhead

overhead includes:
  KV cache (grows with context length)
  activations (intermediate calculations)
  framework (Ollama, llama.cpp themselves)

so a 4GB Q4 model needs ~5-6GB to run comfortably
```

### Apple Silicon Advantage

```
NVIDIA GPU:                    APPLE SILICON:

┌────────────────────┐         ┌───────────────────────────┐
│  System RAM        │         │                           │
│  32GB              │         │   Unified Memory Pool     │
│  (CPU only)        │         │   16GB                    │
│                    │         │   shared by CPU and GPU   │
├────────────────────┤         │                           │
│  VRAM              │         │   full 16GB usable        │
│  8GB               │         │   for model weights       │
│  (GPU only)        │         │                           │
│  HARD LIMIT        │         └───────────────────────────┘
└────────────────────┘
                               M1 Pro 16GB often beats
model must fit in 8GB VRAM     16GB VRAM NVIDIA card
or it fails entirely           for local inference
```

### LLMs on M1 Pro 16GB

```
┌──────────────────┬──────────┬─────────────────────────┬───────────────┐
│ Model            │ RAM (Q4) │ Good at                  │ Speed         │
├──────────────────┼──────────┼─────────────────────────┼───────────────┤
│ Llama 3.2 3B     │ ~3GB     │ Fast tasks, prototyping  │ 60-80 tok/s   │
│ Phi-4 Mini       │ ~4GB     │ Surprisingly capable     │ 50-70 tok/s   │
│ Mistral 7B       │ ~5GB     │ Instruction following    │ 35-50 tok/s   │
│ Llama 3.1 8B     │ ~6GB     │ General all-rounder      │ 30-45 tok/s   │
│ DeepSeek-R1 8B   │ ~6GB     │ Step-by-step reasoning   │ 20-30 tok/s*  │
│ Gemma 3 12B      │ ~9GB     │ Reasoning, writing       │ 20-30 tok/s   │
│ Qwen 2.5 14B     │ ~10GB    │ Coding, math             │ 15-25 tok/s   │
│ DeepSeek-R1 14B  │ ~10GB    │ Best local reasoning     │ 10-18 tok/s*  │
│ Llama 3.1 70B Q2 │ ~14GB    │ Best local quality       │ 4-8 tok/s     │
└──────────────────┴──────────┴─────────────────────────┴───────────────┘
*R1 generates reasoning tokens before answering — slower, quality justifies it
```

### Local Image Gen on M1 Pro

```
FLUX family (Black Forest Labs — go-to in 2025):
  FLUX.1 Schnell ──► fast, Apache 2.0, ~12GB
  FLUX.2 klein   ──► sub-second, ~13GB
  
Stable Diffusion ecosystem:
  SD 1.5 / SDXL fine-tunes
  massive library of fine-tuned models
  very well optimised for Apple Silicon

Interfaces:
  ComfyUI ──► node-based, best for Mac, most flexible
  Draw Things ──► native Mac app, good for quick experiments

Expect: 30-90 seconds per image on M1 Pro
```

### Local Video Gen on M1 Pro

```
REALITY CHECK:
  video models are CUDA-first
  MPS (Apple's GPU backend) support is patchy
  expect slower than equivalent NVIDIA
  some models fail to load entirely

WHAT CAN REALISTICALLY RUN:

  LTX-Video     ──► 12GB min, best bet for M1 Pro
  Wan 2.1 1.3B  ──► ~8GB, surprisingly capable
  CogVideoX 2B  ──► most accessible, short clips

  HunyuanVideo 1.5 ──► technically fits at 14GB
                       leaves almost no headroom
                       MPS support experimental

Expect: MINUTES per clip, not seconds
For Frameo-type production work: cloud inference
(fal.ai, Replicate) more practical at this hardware tier
```

---

## 8. The Full Training Pipeline

```
RAW TEXT
    │
    ▼
TOKENISER
"Hello world" ──► [15496, 995]  (token IDs)
                  roughly word-sized chunks
    │
    ▼
EMBEDDING LAYER
[15496] ──► [0.2, -0.8, 0.4, 0.1, ...]  (vector, 768-12288 dims)
    │
    ▼
TRANSFORMER LAYERS × 96
    ├── Attention (every token looks at every other)
    ├── Add & Normalise
    ├── Feedforward (each token processed independently)
    └── Add & Normalise
    │
    ▼
OUTPUT HEAD
vector ──► probability distribution over 50,000 possible next tokens
    │
    │    "mat"    : 0.847  ◄── most likely
    │    "floor"  : 0.091
    │    "ground" : 0.034
    │    "ceiling": 0.001
    │    ...
    ▼
SAMPLE
pick token ──► add to sequence ──► repeat from top
```

### The Blocks Affecting Starting Slope

```
TRAINING HAS THREE PHASES:

error
  │
1.0│ *               ① EARLY — rapid drop
  │  *                  basic patterns, grammar,
  │   *                 common words
  │    **
  │      ***
  │         ****     ② MIDDLE — slowing
  │             ****    harder patterns,
  │                 **  reasoning, rare knowledge
  │                   **
  │                    *────────────────
  │                                     ③ PLATEAU
  │                                        diminishing returns
  └─────────────────────────────────────────────────────────►

WHAT CONTROLS THE STARTING SLOPE:

                         impact
                         on slope
block                    ████████████████
─────────────────────────────────────────
pretraining              ████████████████  massive
data quality/ordering    ████████████      large
weight init (Xavier/He)  ████████          significant
lr warmup schedule       ███████           significant
normalisation            ██████            moderate
residual connections     █████             moderate
batch size               ████              moderate
optimizer (Adam)         ███               moderate

LEARNING RATE SHAPES:

CONSTANT (naive):
lr │────────────────────────────
   └────────────────────────────►
   often too aggressive early

WARMUP THEN DECAY (standard):
lr │        ╭─╮
   │       ╭╯ ╰╮
   │      ╭╯   ╰──╮
   │     ╭╯        ╰──────────
   │────╭╯
   └────────────────────────────►
        ▲       ▲
     warmup   decay
     (find    (don't
     footing) overshoot)

RESIDUAL CONNECTIONS (why deep networks train):

WITHOUT:                    WITH:
                            
gradient at layer 96: 0.9   gradient at layer 96: 0.9
gradient at layer 50: 0.1   gradient at layer 50: 0.85
gradient at layer 10: 0.001 gradient at layer 10: 0.79
gradient at layer 1:  0.00001 gradient at layer 1: 0.76

          input ──► [layer] ──►[+]──► output
                        │      ▲
                        └──────┘
                     skip connection
                     (gradient highway)
```

---

## 9. Three Types of Training — Critical Distinction

```
┌─────────────────────────────────────────────────────────────────┐
│ PRETRAINING FROM SCRATCH                                        │
│                                                                 │
│ random weights → train on billions of tokens → capable model   │
│                                                                 │
│ cost:   $10,000 – $1,000,000+                                   │
│ time:   weeks on many GPUs                                      │
│ who:    only big labs                                           │
│ on M1:  decades. not viable.                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ FINE TUNING                                                     │
│                                                                 │
│ pretrained model → adapt to specific task                       │
│                                                                 │
│ ┌──────────────────────────────────────┐                        │
│ │ pretrained knowledge landscape       │                        │
│ │                                      │                        │
│ │  ~~~general knowledge~~~             │                        │
│ │  ~~reasoning capability~~            │                        │
│ │         ╔════════════╗               │                        │
│ │         ║ your task  ║ ◄── carved   │                        │
│ │         ║  carved    ║    deeper    │                        │
│ │         ╚════════════╝              │                        │
│ └──────────────────────────────────────┘                        │
│                                                                 │
│ cost:   $10 – $1,000                                            │
│ time:   hours to days, single GPU                               │
│ on M1:  YES (with LoRA/QLoRA)                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DISTILLATION                                                    │
│                                                                 │
│ large model (teacher) → train small model to imitate it         │
│                                                                 │
│ RAW TEXT TRAINING:              DISTILLATION:                   │
│                                                                 │
│ target: "Paris"                 teacher says:                   │
│ signal: right or wrong          Paris    : 0.94                 │
│ binary, harsh                   Lyon     : 0.03                 │
│                                 Marseille: 0.02                 │
│                                 London   : 0.005                │
│                                                                 │
│                                 student learns the DISTRIBUTION  │
│                                 not just the top answer          │
│                                                                 │
│ the soft probabilities contain hidden knowledge:                │
│ "Lyon is more Paris-like than London"                           │
│ "French cities relate differently to Paris than foreign ones"   │
│ this knowledge never existed in the raw text "Paris"            │
│                                                                 │
│ result: small model punches above its weight                    │
│ this is how DeepSeek distills work                              │
└─────────────────────────────────────────────────────────────────┘
```

### LoRA — Why Fine Tuning is Cheap

```
FULL FINE TUNING:
  update all 175B parameters
  like renovating every room in a skyscraper
  expensive, risks destroying existing knowledge

LORA INSIGHT:
  the CHANGE needed to adapt a model = low rank
  
  you don't need 16 million numbers to describe the change
  you can approximate it as A × B
  
  FULL UPDATE:              LORA:
  
  ████████████████          ██      ████████████████
  ████████████████    ≈     ██  ×   ████████████████
  ████████████████          ██
  ████████████████          ██
  (4096×4096 matrix)    (4096×8) × (8×4096)
  16 million numbers    65,536 numbers
  
  99.6% fewer parameters to train
  original weights completely untouched
  
  at inference: original weights + A×B = adapted model

WHAT'S IN MEMORY DURING LORA FINE TUNE (M1 Pro 16GB):

┌──────────────────────────────────┐
│ model weights (frozen)  ~6 GB   │
│ LoRA adapters           ~0.5 GB │
│ optimizer states        ~1 GB   │
│ activations             ~2 GB   │
│ gradients               ~1 GB   │
│ batch data              ~0.1 GB │
│ OS + framework          ~2 GB   │
├──────────────────────────────────┤
│ total                   ~12.6GB │
│ headroom                ~3.4 GB │
│ TIGHT BUT POSSIBLE              │
└──────────────────────────────────┘
```

### Can You Train on a Mac?

```
PRETRAINING 7B FROM SCRATCH on M1 Pro:

  frontier training: ~2 weeks on 1000 H100s
  M1 Pro equivalent: ~decades
  verdict: no

FULL FINE TUNING 7B on M1 Pro:

  needs ~40-50GB for weights + gradients
  M1 Pro has 16GB
  verdict: no

QLORA FINE TUNING 7B on M1 Pro:

  7B model: ~10-12GB ✓
  3B model: ~6-8GB   ✓
  tools: MLX (Apple's framework, fastest on Mac)
         Unsloth, LLaMA Factory
  verdict: YES — hours to days

PRACTICAL WORKFLOW:

  prototype + experiment ──► M1 Pro (free)
  actual training run    ──► RunPod / Colab (~$0.50-2/hr)
  inference              ──► back to M1 Pro (free)
```

---

## 10. Training at Scale — Parallelised Pipelines

```
SINGLE GPU (what your M1 Pro does):

┌──────────────────────────────┐
│  data batch                  │
│     │                        │
│     ▼                        │
│  forward pass                │
│     │                        │
│     ▼                        │
│  loss                        │
│     │                        │
│     ▼                        │
│  backward pass               │
│     │                        │
│     ▼                        │
│  weight update               │
└──────────────────────────────┘
sequential, simple, slow

FRONTIER TRAINING (three types of parallelism simultaneously):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA PARALLELISM — same model, different data:

  GPU 1: [model copy] ← batch A → gradient A ─┐
  GPU 2: [model copy] ← batch B → gradient B ─┼──► average ──► update all
  GPU 3: [model copy] ← batch C → gradient C ─┘
  GPU 4: [model copy] ← batch D → gradient D ─┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL PARALLELISM — model too big for one GPU:

  GPU 1: [layers 1-24]  ──activations──►
  GPU 2:                [layers 25-48] ──activations──►
  GPU 3:                               [layers 49-72] ──activations──►
  GPU 4:                                              [layers 73-96]
  
  like an assembly line
  each GPU works on its section
  passes result to next GPU

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TENSOR PARALLELISM — single operation split across GPUs:

  weight matrix W (too big for one GPU)
  split into W1, W2, W3, W4
  
  GPU 1: input × W1 ──► partial result 1 ─┐
  GPU 2: input × W2 ──► partial result 2 ─┼──► combine ──► full result
  GPU 3: input × W3 ──► partial result 3 ─┘
  GPU 4: input × W4 ──► partial result 4 ─┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL THREE RUNNING SIMULTANEOUSLY:
  thousands of GPUs
  coordinated across data centres
  the cost isn't just the GPUs —
  it's the networking, synchronisation, coordination overhead
  this is why cloud compute is expensive even if you own the hardware
```

### Compute vs Capability Landscape

```
capability
    │
    │                                        ▲ GPT-4 class
    │                                   ─────┤ (trillion+ tokens)
    │                                ────    │ (thousands of GPUs)
    │                             ────       │ ($100M+)
    │                          ────          │
    │                       ────             │ DeepSeek R1 full
    │                    ────                │ (~$6M, still massive)
    │                 ────                   │
    │              ────                      │ DeepSeek distills (8B, 14B)
    │           ────                         │ (fine tune, cheap)
    │        ────                            │
    │     ────                               │ Phi-4 Mini, Gemma 3 small
    │  ────                                  │ (small + quality data)
    │────                                    │
    └────────────────────────────────────────┼──────────────► compute
    $0      $10     $100    $1k    $10k    $1M+

THE INTERESTING ZONE ($0-$100, consumer hardware):

  current: fine tune existing models for narrow tasks
  
  your question: could pretraining happen here?
  
  general model pretraining: NO
    parameter floor is real
    below ~1B general capability collapses
    DeepSeek distills are near the practical minimum
    
  specialist model pretraining: POSSIBLY YES
    this is the underexplored space
    (see section 14)
```

---

## 11. Why We Can't Read What We Built

```
THE SITUATION:

every weight = accessible
nothing hidden
every number available

┌─────────────────────────────────┐
│ weight 1:        0.002341       │
│ weight 2:       -0.007821       │
│ weight 3:        0.015634       │
│ ...                             │
│ weight 175,000,000,000: 0.004521│
└─────────────────────────────────┘

THE PROBLEM: not access. comprehension.

THE ANALOGY:

  you have a book
  written in a language nobody has seen before
  every character perfectly visible
  you can count them, cluster them, run statistics
  
  but you cannot READ it
  
  the numbers are the characters
  the language is unknown
  the meaning is in there somewhere
  we don't have the decoder yet
```

### Four Reasons We Can't Read It

```
REASON 1 — SCALE:

  you could print every weight
  fill every library on earth
  and still not be able to read what's happening
  
  175B parameters × connections × interactions
  the sheer size defeats human comprehension

REASON 2 — DISTRIBUTED KNOWLEDGE:

  "Paris is the capital of France" is encoded as...
  
  weight 4,521,847   = 0.0023
  weight 891,234,102 = -0.0071
  weight 2,341,998   = 0.0156
  ... across millions of weights
  each contributing a tiny fragment
  none meaning anything alone
  
  like asking:
  which specific water molecules in the ocean
  are responsible for this specific wave?
  
  the wave is real
  the question has no clean answer

REASON 3 — SUPERPOSITION:

  WHAT WE HOPED:
  neuron 4521: responds to "France"
  neuron 4522: responds to "capitals"
  neuron 4523: responds to "geography"
  
  clean, one concept per neuron
  
  WHAT ACTUALLY HAPPENS:
  neuron 4521: responds to "France" AND
               "certain verb tenses" AND
               "concept of distance" AND
               "something about music" AND
               17 other unrelated things
  
  model packs more concepts than it has neurons
  by overlapping representations
  efficient but completely unreadable
  
  there is no "Paris neuron"
  Paris is a PATTERN across thousands of neurons
  that only manifests in certain contexts

REASON 4 — EMERGENCE:

  trained on: predict next token
              predict next token
              (billions of times)
  
  what emerged that nobody put there:
  ┌─────────────────────────────────────┐
  │ arithmetic                          │
  │ cause and effect understanding      │
  │ theory of mind                      │
  │ code writing                        │
  │ analogical reasoning                │
  │ detecting irony                     │
  └─────────────────────────────────────┘
  
  nobody designed these in
  nobody knows exactly when they appeared
  nobody knows exactly where they live
```

### Your Insight: It's a Visualisation Problem

```
YOU SAID:
  "it feels like a visualisation problem —
   clustering and more layers"

THIS IS EXACTLY WHAT RESEARCHERS ARE WORKING ON:

WRONG UNIT:     individual weights
                too small, meaningless alone

BETTER UNIT:    features
                patterns that activate together
                
BETTER STILL:   circuits
                features connecting to features

BEST:           algorithms
                what the model is actually doing

THE HIERARCHY:

weights
  └──► features    (what concepts exist)
         └──► circuits    (how concepts interact)
                └──► motifs    (recurring patterns)
                       └──► algorithms   (what it's doing)
                              └──► reasoning  (legible story)

each level legible at its own scale
like zoom levels on a map

SPARSE AUTOENCODERS (Anthropic's key work):

BEFORE:
  neuron 4521 = 0.72   ← what does this mean???

AFTER sparse autoencoder decomposition:
  france-related:      0.71  ← readable
  capital-city:        0.43  ← readable
  geography:           0.38  ← readable
  music-related:       0.01  ← noise
  
  suddenly legible

THE FULL MULTI-LEVEL VISUALISATION (what's needed):

INPUT: "The Eiffel Tower is in ___"
                │
                ▼
┌───────────────────────────────────────────────┐
│ LIVE FEATURE MAP                              │
│ token: "Eiffel Tower"                         │
│ ████ france-geography    0.89                 │
│ ███░ landmark            0.71                 │
│ ██░░ europe              0.54                 │
└───────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│ CIRCUIT TRACE                                 │
│ france-geography                              │
│      │                                        │
│      ├──► capital-lookup-circuit              │
│      │         │                              │
│      │         ▼                              │
│      │    suppresses other cities             │
│      │         │                              │
│      └──────── outputs "Paris"               │
└───────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│ CONFIDENCE MAP                                │
│ Paris      ████████  0.94                    │
│ Lyon       █░░░░░░░  0.08                    │
│ Marseille  ░░░░░░░░  0.02                    │
└───────────────────────────────────────────────┘

this doesn't fully exist yet
the pieces are being assembled
```

---

## 12. What DeepSeek Proved

```
CONVENTIONAL WISDOM BEFORE DEEPSEEK:

  frontier capability = f(money + compute + data)
  
  only players with massive resources
  can be at the frontier
  
  chip export controls = capability controls
  no H100s = no frontier AI

WHAT DEEPSEEK DID:

  DeepSeek V2 (mid 2024):
    competitive with GPT-4
    ~1/10th the training cost
    via fine-grained MoE
    
  DeepSeek R1 (early 2025):
    matched OpenAI o1 on reasoning benchmarks
    trained on ~$6M of compute
    o1 estimated: $100M+
    open weights released publicly
    
    trained on H800s (restricted export chips)
    lower memory bandwidth than H100s
    STILL MATCHED O1

THE SPECIFIC INNOVATIONS:

MULTI-HEAD LATENT ATTENTION (MLA):

  standard attention KV cache:
    100 units stored per token
    
  MLA:
    compress to latent vector
    4 units stored per token
    
    96% memory reduction
    essentially no quality cost

FINE-GRAINED MoE:

  standard MoE:     8 experts, pick top 2
  DeepSeek MoE:     64 fine-grained experts
                    + shared experts always active
                    pick top 6 + shared
                    
  more specialisation, less redundancy
  lower compute per token

R1'S REASONING EMERGED FROM ALMOST NOTHING:

  training signal: is the answer correct?
  (maths has verifiable answers)
  (code either runs or doesn't)
  
  nobody told it how to reason
  these emerged spontaneously:
  ┌────────────────────────────────────────┐
  │ slow down on hard problems             │
  │ check its own work                     │
  │ try alternative approaches             │
  │ backtrack when stuck                   │
  │ express uncertainty                    │
  └────────────────────────────────────────┘
  reasoning as emergent behaviour
  from a simple correctness signal

WHAT IT PROVED:

BEFORE:                         AFTER:
                                
frontier capability             frontier capability
      │                               │
   requires                       reachable via
      │                         ┌────┴──────────────┐
      ▼                         ▼                   ▼
massive compute          massive compute    algorithmic efficiency
                         (brute force)      (being clever)

two paths exist
the second is much cheaper
doesn't require export-controlled hardware
frontier more democratically accessible
than anyone assumed
```

---

## 13. The Scale Question — Can We Go Smaller Than DeepSeek?

```
CURRENT SMALL MODEL LANDSCAPE:

DeepSeek R1 distill 8B   ──► ~6GB,  good reasoning
DeepSeek R1 distill 1.5B ──► ~1.5GB, surprisingly capable
Llama 3.2 1B             ──► ~1GB,  basic tasks
Qwen 2.5 0.5B            ──► ~500MB, narrow use

below ~1B parameters
general capability collapses

THE PARAMETER FLOOR:

capability
    │
    │                    ●  7B  (good general reasoning)
    │                ●  3B  (decent, useful)
    │            ●  1B  (basic, limited)
    │        ●  500M  (narrow tasks only)
    │    ●  100M  (very specific tasks only)
    │●  10M  (barely useful)
    └─────────────────────────────────────────► parameters

the floor for general capability ≈ 1B parameters
below that: only useful for narrow specific tasks

SCALING LAWS (Chinchilla, 2022):

optimal training = model size and data scale together

  if you double parameters ──► double training tokens

  GPT-3:    175B params,   300B tokens  ← undertrained
  Llama 1:  7B params,     1T tokens    ← better
  Llama 3:  8B params,     15T tokens   ← even better

KEY INSIGHT:
  smaller model + MORE data
  often beats larger model + less data
  
  Phi proved this
  Gemma proved this
  DeepSeek refined this

BUT there's a floor:
  below a certain parameter count
  no amount of data helps
  the model lacks CAPACITY
  to store the patterns
  regardless of how many times you show them
```

### The Specialist Model Insight

```
GENERAL MODEL (what big labs build):

  must learn:
    grammar of 100 languages
    history of human civilisation
    all of science, code, literature
    reasoning in general
    
  requires: billions of parameters
  requires: trillions of tokens  
  requires: millions of dollars
  cost: $$$$$

SPECIALIST MODEL (the underexplored space):

  must learn ONE thing extremely well
  
  example: natural language → SQL queries
  
  vocabulary needed:   ~500 words
  patterns needed:     ~50 core structures
  edge cases:          ~200 variations
  
  minimum viable model: ~100M parameters
  training data:        ~10M examples
  RAM:                  ~100MB
  training time:        hours on a laptop
  training cost:        nearly zero
  
  AND: outperforms 7B general model
       on this specific task

THE COMPARISON:

GENERAL 7B MODEL on SQL task:
  knows everything
  SQL is a small fraction of its knowledge
  might get it right, might confabulate
  
SPECIALIST 100M MODEL on SQL task:
  knows ONLY SQL
  100% of capacity dedicated to this
  more reliable, more accurate, way smaller

THIS IS THE UNDEREXPLORED SPACE:

  nobody is asking:
  "what is the MINIMUM model
   that solves THIS specific problem?"
  
  not minimum general model
  minimum TASK model

PHI PROVED THE PRINCIPLE:
  quality of data > quantity of parameters
  
  Phi-4 Mini at 4GB
  outperforms many 7B models
  because training data was
  extremely carefully curated
  
  nobody has systematically pushed
  how small you can go
  on specific tasks
  with perfect task-specific data

THE COMPUTE PICTURE FOR SPECIALIST MODELS:

  frontier general model:    $6M – $100M+
  DeepSeek distill (8B):     $10k – $100k
  standard fine tune (7B):   $10 – $1,000
  specialist model (100M):   $0 – $10
  trains on M1 Pro:          hours
  
  this zone is almost completely unexplored
  by serious researchers
```

---

## 14. The Three Major Unsolved Problems

```
PROBLEM 1 — WE CAN'T READ WHAT WE BUILT

  what we have:    models that work remarkably well
  what we lack:    any reliable way to verify WHY
  
  blocker:
  ┌────────────────────────────────────────────┐
  │ can't verify reasoning ──► trust problem   │
  │ can't find bias        ──► fairness problem│
  │ can't confirm honesty  ──► safety problem  │
  │ can't predict failure  ──► reliability     │
  └────────────────────────────────────────────┘
  
  state: sparse autoencoders promising
         years from practical deployment safety

PROBLEM 2 — LEARNING IS BROKEN VS NATURE

  model:                  human child:
  
  15 trillion tokens      100 million words by age 10
  to train                100x less data
  then FROZEN forever     keeps learning every day
                          for free, from experience
  
  CATASTROPHIC FORGETTING:
  
  teach model new thing ──► overwrites old things
  can't update a model like you update a human
  
  blocker: no persistent experience
           every conversation starts from zero
           knowledge without a life lived
  
  state: mostly unsolved
         no convincing solution at scale

PROBLEM 3 — INTELLIGENCE WITHOUT REALITY

  everything models know came through:
  
  reality ──► humans observed ──► humans wrote ──► model trained
  
  TWO LOSSY COMPRESSIONS between model and world
  
  what gets lost:
    physical intuition ──► which way does water flow?
    causal grounding   ──► does fire cause burns
                           or do fire and burn just co-occur?
    spatial reasoning  ──► which shape fits which hole?
    
  blocker: hallucination at its root
           not a bug to fix
           consequence of learning from descriptions
           rather than from reality itself
  
  state: multimodal helps at edges
         embodied AI interesting but nascent
         fundamental gap remains

HOW THE THREE CONNECT:

         can't read the model (P1)
                │
    ┌───────────┴──────────────┐
    │                          │
    ▼                          ▼
learning is frozen (P2)    no grounding (P3)
    │                          │
    └──────────────┬───────────┘
                   │
         capable but not reliable
         capable but not trustworthy
         capable but not truly intelligent

most current AI progress = scaling existing systems
improves capability
doesn't touch these roots
which is why new models feel simultaneously
more impressive and somehow the same
```

---

## 15. Your Three Original Insights

*These emerged unprompted from the conversation. They're genuinely interesting.*

```
INSIGHT 1 — REPETITION BUILDS INTUITION, NOT STORAGE

  what you noticed:
    humans repeat understanding from many angles
    until it becomes intuition
    then let go of the facts
    keep the shape of the understanding
    
  what models do:
    see fact once ──► nudge weights
    see it 1000 times ──► nudge weights 1000 times
    no qualitative reorganisation
    no "this is now intuition not storage"
    
  the gap:
  
    SHALLOW PASS:
    "Paris is the capital of France"
    ──► stored as statistical association
    
    REPEATED DEEP PASSES:
    Paris in history
    Paris in geography
    Paris in culture
    Paris in politics
    Paris as a concept with texture and weight
    ──► becomes INTUITION
        fact can be let go
        understanding stays
    
    models have the first
    not the second
    they can tell you everything about Paris
    without understanding what Paris IS

────────────────────────────────────────────────────────

INSIGHT 2 — REFERENCEABLE KNOWLEDGE SHOULDN'T BE MEMORISED

  what you noticed:
    brain separates what to memorise vs reference
    
    memorise:    intuitions, patterns, judgment, skills
    reference:   dates, exact quotes, precise numbers
    
  current LLMs:
    memorise everything into weights
    AND are bad at retrieval
    because they reconstruct from weights
    (which can go wrong = hallucination)
    rather than look up
    
  THE RIGHT ARCHITECTURE:
  
  ┌───────────────────────────────────────────────┐
  │                                               │
  │   REASONING CORE (in weights)                 │
  │   intuitions, patterns, how to think          │
  │                                               │
  │           ▲               │                  │
  │           │               ▼                  │
  │       query           retrieve               │
  │           │               │                  │
  │           │               ▼                  │
  │   KNOWLEDGE STORE (external)                  │
  │   facts, figures, exact information           │
  │   always accurate, always updatable           │
  │   without retraining                          │
  │                                               │
  └───────────────────────────────────────────────┘
  
  RAG exists but is BOLTED ON:
    model still tries to answer from weights first
    retrieval is a fallback
    weights and retrieval compete
    
  your insight:
    flip the default
    model DISTRUSTS its own parametric memory
    ALWAYS retrieves before stating facts
    weights used only for reasoning around facts

────────────────────────────────────────────────────────

INSIGHT 3 — SIGHTED GRADIENT DESCENT

  what you noticed:
    blindfolded person feeling slope = inefficient
    sighted person seeing landscape = strictly better
    
  second order optimisation:
    provably better
    costs 30 quintillion numbers to compute
    not feasible at scale
    
  the deeper version you were pointing at:
  
    current: training process reacts to gradient signal
    yours:   training process REASONS about what it's learning
    
  this requires solving interpretability first:
    to reason about what you're learning
    you need to read what you've already learned
    which is problem 1
    
  these three insights are connected:
  
  intuition not storage ──────────────────► model knows
  referenceable facts outside ─────────────  what KIND of
  training that can see itself ────────────►  thing it's learning

THE UNIFIED INSIGHT:

  a model that:
    builds intuition not storage
    references facts externally  
    can see what it's learning
    
  would be:
    interpretable by design (solves P1)
    continuously learnable (solves P2)
    grounded in retrievable reality (solves P3)
    
  suspiciously close to how human cognition works:
    prefrontal cortex ──► reasoning
    hippocampus ───────── episodic memory (referenceable)
    neocortex ─────────── consolidated intuitions
```

---

## 16. The Three POCs

*Context: zero code, Claude Code as implementation partner, M1 Pro 16GB.*
*Goal: find a small thing that was wrong about an assumption everyone made.*

---

### POC 1 — Depth of Repetition Changes Reasoning Quality

```
THE ASSUMPTION TO CHALLENGE:
  training data volume and diversity matter
  depth of repetition on the same concept
  doesn't produce qualitatively different reasoning

THE HYPOTHESIS:
  a model trained on the same facts
  repeated from many angles
  will REASON around those facts better
  than a model trained on many facts stated once
  even at the same total token count

WHAT YOU BUILD:

Dataset A (depth):
  "Paris is the capital of France"
  × 100 different contexts:
    Paris in history
    Paris in geography
    Paris in literature
    Paris in architecture
    Paris as political centre
    ... (all same fact, different angles)

Dataset B (breadth):
  100 different facts
  each stated once

SAME token count. DIFFERENT structure.

Fine tune same base model on both.

TEST (this is the key — NOT retrieval, REASONING):

  ✗ wrong test: "what is the capital of France?"
    both models answer correctly — measures retrieval
    
  ✓ right test: "if France moved its government to Lyon,
                  what would change about Paris's role
                  in European politics?"
    requires USING the knowledge, not retrieving it
    model A (deep) vs model B (broad) — which reasons better?

WHY IT MATTERS IF TRUE:
  current training pipelines optimise for breadth
  "cover more ground" is the implicit goal
  if depth on fewer things produces more genuine understanding
  implications for how training data is curated at scale

HARDWARE: M1 Pro, MLX for fine tuning, ~2-4 hours per run
NUANCE: defining "reasoning quality" vs "retrieval accuracy"
         is the measurement challenge
         need tasks that fail on surface pattern matching
         but succeed on genuine understanding
```

---

### POC 2 — Train the Model to Distrust Itself *(recommended starting point)*

```
THE ASSUMPTION TO CHALLENGE:
  RAG works by adding retrieval to a model that reasons
  model answers, retrieval helps when model is uncertain
  retrieval is a supplement to parametric memory

THE HYPOTHESIS:
  a model trained to ALWAYS retrieve before answering —
  to distrust its own parametric memory by default —
  will hallucinate less than standard RAG
  where retrieval is bolted on as fallback

THREE-WAY COMPARISON:

┌──────────────────────────────────────────────────────┐
│ BASELINE                                             │
│ model with no retrieval                              │
│ answers from weights only                            │
│ reconstructs facts, can go wrong                     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ STANDARD RAG (bolted on)                             │
│ model still tries weights first                      │
│ retrieval as fallback / supplement                   │
│ weights and retrieval sometimes compete              │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ YOUR VERSION (trained to retrieve)                   │
│                                                      │
│  reasoning model (3B)                                │
│  trained to always say "let me check"                │
│  before stating any fact                             │
│         │                                            │
│         ▼                                            │
│  vector database                                     │
│  (all facts live here)                               │
│  (exact, retrievable, updateable)                    │
│         │                                            │
│         ▼                                            │
│  model reasons around retrieved facts               │
│  weights used ONLY for reasoning                     │
│  not for fact storage                                │
└──────────────────────────────────────────────────────┘

MEASURE:
  hallucination rate
  answer accuracy
  latency (retrieval adds time)

ORIGINAL CONTRIBUTION:
  not RAG — RAG exists
  training the model to have a different
  DEFAULT BEHAVIOUR toward its own memory
  
  small shift in assumption
  potentially large shift in results

WHY INTERESTING REGARDLESS OF OUTCOME:

  your version wins ──► default behaviour matters
                        not the retrieval mechanism
                        
  no difference    ──► training vs prompting retrieval
                        makes no difference (useful to know)
                        
  standard RAG wins ──► competition between weights and
                         retrieval is productive somehow
                         new hypothesis generated

HARDWARE: M1 Pro, Ollama, ChromaDB (local vector store), MLX
NUANCE: measurement must distinguish between
         "retrieved correctly but reasoned wrong"
         vs "ignored retrieval, used weights instead"
         these are different failure modes → different fixes

WEEK BY WEEK:

  Week 1: baseline — 50 factual questions, measure hallucination rate
  Week 2: add standard RAG, measure again, compare
  Week 3: fine tune model to always retrieve (your novel part)
  Week 4: three-way comparison table + chart
```

---

### POC 3 — A Model That Watches Itself Learn

```
THE ASSUMPTION TO CHALLENGE:
  training is a blind optimisation process
  only signal: were the outputs correct?
  training has no awareness of WHAT is being learned
  (memorising surface patterns vs building intuitions)

THE HYPOTHESIS:
  a small observer model watching a student model's
  activations during training — and adjusting what
  training examples to show based on what the student
  appears to be learning — will produce better
  generalisation than standard fixed-dataset training

WHAT YOU BUILD:

┌──────────────────────────────────────────────────────────┐
│                                                          │
│  STUDENT MODEL (1.5B, being trained)                     │
│                                                          │
│  after every N training steps:                           │
│         │                                                │
│         │ sample activations                             │
│         │ test on paraphrased versions of training data  │
│         │ (memorisation fails, intuition succeeds)       │
│         ▼                                                │
│  OBSERVER MODEL (3B, frozen)                             │
│                                                          │
│  "is the student memorising surface patterns             │
│   or building generalisable intuitions?"                 │
│         │                                                │
│         │ feedback signal                                │
│         ▼                                                │
│  ADJUST TRAINING:                                        │
│  if memorising ──► show more varied examples             │
│  if intuiting  ──► show deeper examples                  │
│                                                          │
└──────────────────────────────────────────────────────────┘

THIS CONNECTS ALL THREE INSIGHTS:

  insight 1 (intuition not storage):
    observer detects whether student is memorising or intuiting
    
  insight 2 (reference not memorise):
    student trained on reasoning, not fact storage
    
  insight 3 (sighted training):
    observer IS the sight for gradient descent
    training process that can see what it's learning

WHY GENUINELY NOVEL:
  current RLAIF (AI feedback):
    model generates ──► second model evaluates OUTPUT quality
    signal: was the output good?
    
  your version:
    model trains ──► second model evaluates LEARNING PROGRESS
    signal: is the student learning the right KIND of thing?
    
  this doesn't exist in this form
  it's a primitive version of
  "training processes that are aware of their own progress"

HARDWARE: tight on M1 Pro (two models simultaneously)
          1.5B student + 3B observer = edge of feasible
          prototype locally, train on RunPod

NUANCE: defining "memorising vs intuiting" is the crux
         proposed proxy:
           after N steps, test on paraphrased versions of training data
           exact paraphrase: model that memorised answers correctly
           semantic paraphrase: model that intuited answers correctly
           observer uses this signal to adjust training curriculum
```

---

## The Disruption Frame

```
DEEPSEEK DISRUPTED NOT BY BUILDING SOMETHING NOBODY HAD
BUT BY PROVING AN ASSUMPTION WAS WRONG:

  assumption: you need massive compute for frontier capability
  proof:      here's the experiment, here's the result
  
YOUR ASSUMPTIONS TO CHALLENGE:

POC 1: depth of repetition matters as much as breadth
POC 2: model's default behaviour toward its own memory
        matters more than the retrieval mechanism
POC 3: training can be made aware of what it's learning

EACH IS:
  small enough to test on consumer hardware
  clear enough to measure
  interesting enough that both outcomes matter
  novel enough to be worth writing up

THE PATH:

POC 2, Week 1: one number
  baseline hallucination rate on 50 factual questions
  
that number is everything
it's your starting point
it makes everything else concrete

the field gets disrupted by people who:
  pick one small clear question
  answer it rigorously
  share it openly
  
not by people who:
  try to solve everything at once
  build in secret
  announce when perfect

you already have the questions
the code is just making them physical
```

---

*This document covers: neural networks, gradient descent, the full architecture history (RNNs → LSTMs → CNNs → Seq2Seq → Transformers → hybrids), attention mechanism, images/video/Genie, model sizing and RAM, local hardware (LLM + image + video on M1 Pro 16GB), the three types of training, LoRA, parallelised training at scale, the compute/capability landscape, why we can't read models, sparse autoencoders, DeepSeek's proof, the specialist model opportunity, the three unsolved problems, your three original insights, and the three POCs.*
