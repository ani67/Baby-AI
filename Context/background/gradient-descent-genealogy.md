# The Family Tree of Gradient Descent
*Explained like you're 10. With pictures made of letters.*

---

## First, what even IS gradient descent?

Imagine you're blindfolded on a hilly field.
Your job: find the lowest point.

You can't see anything.
But you can *feel* the ground under your feet.

So you do this:

```
feel which way the ground slopes down
take one small step that way
feel again
step again
repeat until ground feels flat

that's it
that's gradient descent
```

Someone invented this in 1847.
For a completely different reason.
And we're still using it today to train AI.

Here's how we got here.

---

## THE FAMILY TREE

```
                        ┌──────────────────────────────┐
                        │         CALCULUS             │
                        │    Newton + Leibniz          │
                        │         ~1700s               │
                        │                              │
                        │  invented the idea of a      │
                        │  "gradient" — which way      │
                        │  does a hill slope?          │
                        └──────────────┬───────────────┘
                                       │
                                       │ 150 years pass
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   AUGUSTIN-LOUIS CAUCHY      │
                        │         1847                 │
                        │      Paris, France           │
                        │                              │
                        │  a very famous               │
                        │  mathematician               │
                        │  not building AI             │
                        │  trying to solve equations   │
                        │                              │
                        │  had an idea:                │
                        │  "if I walk downhill         │
                        │   step by step               │
                        │   I'll find the bottom"      │
                        │                              │
                        │  wrote it in a 2-page paper  │
                        │  nobody thought it was       │
                        │  a big deal                  │
                        └──────────────┬───────────────┘
                                       │
                          ┌────────────┴─────────────┐
                          │                          │
                          │  Cauchy himself knew      │
                          │  it had a problem:        │
                          │                          │
                          │     🏔️  zig              │
                          │    ╱╲  ╲  zag            │
                          │   ╱  ╲  ╲                │
                          │  ╱    ╲  ╲  zig          │
                          │        ╲  ╲              │
                          │         ╲  ╲ zag         │
                          │          ╲  ●  zig...    │
                          │                          │
                          │  walk one way            │
                          │  overshoot               │
                          │  walk back               │
                          │  overshoot               │
                          │  takes forever           │
                          │                          │
                          │  known to be bad         │
                          │  from day one            │
                          └──────────────────────────┘
                                       │
                                       │ 60 years pass
                                       │ nobody does much with it
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │         1907                 │
                        │   HADAMARD independently     │
                        │   invents the exact same     │
                        │   thing                      │
                        │                              │
                        │   meaning: it's such an      │
                        │   obvious idea that two      │
                        │   brilliant people           │
                        │   thought of it separately   │
                        │   60 years apart             │
                        └──────────────┬───────────────┘
                                       │
                                       │ 44 more years pass
                                       │ computers are invented
                                       │ WWII happens
                                       │ Operations Research is born
                                       │ (how do you move soldiers?
                                       │  optimize factory output?)
                                       │
                                       ▼
              ┌────────────────────────────────────────────┐
              │         ROBBINS & MONRO — 1951             │
              │         STATISTICIANS, not AI people       │
              │                                            │
              │  problem they were solving:                │
              │  "I can only see blurry noisy              │
              │   versions of the hill                     │
              │   how do I still walk downhill?"           │
              │                                            │
              │  this is the RANDOM version               │
              │  of Cauchy's walk                          │
              │  called Stochastic Gradient Descent        │
              │  (stochastic = random)                     │
              │                                            │
              │  still not for AI                          │
              │  for statistics                            │
              └─────────────────┬──────────────────────────┘
                                │
              ┌─────────────────┴──────────────────────────┐
              │                                            │
              ▼                                            ▼
┌─────────────────────────┐              ┌─────────────────────────┐
│   ROSENBLATT — 1957     │              │  THE DARK AGES          │
│   invents the perceptron│              │  1969 — 1986            │
│   (first "neuron")      │              │                         │
│                         │              │  two mathematicians     │
│   uses a gradient-ish   │              │  prove neural networks  │
│   update rule           │              │  can't do much          │
│   kind of               │              │                         │
│   not quite backprop    │              │  everyone gives up      │
│   yet                   │              │  money disappears       │
│                         │              │  researchers move on    │
│   first time gradient   │              │                         │
│   touches neural nets   │              │  this period called     │
└───────────┬─────────────┘              │  "AI Winter"            │
            │                            └─────────────────────────┘
            │
            │  big problem surfaces:
            │
            │  ┌────────────────────────────────────┐
            │  │ HOW DO YOU COMPUTE THE GRADIENT    │
            │  │ FOR A DEEP NETWORK?                │
            │  │                                    │
            │  │ gradient descent = knows which     │
            │  │ way to step                        │
            │  │                                    │
            │  │ backprop = figures out WHICH WAY   │
            │  │ IS DOWNHILL in the first place     │
            │  │                                    │
            │  │ these are two different things!    │
            │  │                                    │
            │  │ gradient descent = the walk        │
            │  │ backprop = the eyes that see       │
            │  │           which way to walk        │
            │  └────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│                  THE BURIED DISCOVERIES                │
│                                                        │
│  1970 — LINNAINMAA (Finland, master's thesis)          │
│    figured out backprop for general math formulas      │
│    wrote it in Finnish                                 │
│    nobody read it                                      │
│                                                        │
│  1974 — WERBOS (Harvard, PhD thesis)                   │
│    applied backprop to neural networks                 │
│    explicitly                                          │
│    nobody listened                                     │
│                                                        │
│  both of these discoveries sat unused                  │
│  for over a decade                                     │
│  like a cure on a shelf nobody opened                  │
└────────────────────────────────────────────────────────┘
            │
            │  12 more years
            │
            ▼
┌────────────────────────────────────────────────────────┐
│         RUMELHART, HINTON, WILLIAMS — 1986             │
│         published in Nature (the famous journal)       │
│                                                        │
│  "Learning representations by back-propagating errors" │
│                                                        │
│  didn't INVENT backprop (Linnainmaa did, 16 yrs prior) │
│  showed clearly how to USE it for deep networks        │
│  made it understandable                                │
│  changed everything                                    │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │                                                 │  │
│  │  think of a deep network like a game of         │  │
│  │  telephone with 96 people                       │  │
│  │                                                 │  │
│  │  person 1 hears something wrong                 │  │
│  │  tells person 2                                 │  │
│  │  tells person 3                                 │  │
│  │  ...                                            │  │
│  │  person 96 says the wrong answer                │  │
│  │                                                 │  │
│  │  BACKPROP = person 96 whispers back:            │  │
│  │  "hey I was wrong, here's by how much"          │  │
│  │  person 95 passes it back                       │  │
│  │  all the way to person 1                        │  │
│  │  everyone adjusts slightly                      │  │
│  │                                                 │  │
│  │  gradient descent = the adjustment              │  │
│  │  backprop = the whisper going backward          │  │
│  │                                                 │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  BUT: still didn't fully work for very deep networks   │
│  the whisper got quieter with each person              │
│  by person 1, almost silent                           │
│  = the VANISHING GRADIENT problem                      │
└───────────────────────┬────────────────────────────────┘
                        │
          ┌─────────────┴──────────────────────┐
          │                                    │
          ▼                                    ▼
┌──────────────────────────┐      ┌────────────────────────────┐
│  NEW PROBLEMS FOUND      │      │  PEOPLE TRY TO FIX CAUCHY'S│
│                          │      │  ZIG-ZAG PROBLEM           │
│  VANISHING GRADIENT      │      │                            │
│  1990s                   │      │  1964 POLYAK               │
│                          │      │  invented MOMENTUM         │
│  ┌──────────────────┐    │      │                            │
│  │ gradient signal  │    │      │  imagine a ball rolling    │
│  │ going back:      │    │      │  downhill                  │
│  │                  │    │      │  it builds up speed        │
│  │ layer 96:  loud  │    │      │  doesn't zig-zag as much   │
│  │ layer 50:  quiet │    │      │  overshoots less           │
│  │ layer 10:  faint │    │      │                            │
│  │ layer 1:   gone  │    │      │  ┌──────────────────────┐  │
│  │                  │    │      │  │ WITHOUT MOMENTUM:    │  │
│  │ early layers     │    │      │  │   zig                │  │
│  │ learn nothing    │    │      │  │      zag             │  │
│  └──────────────────┘    │      │  │         zig          │  │
│                          │      │  │            zag...    │  │
│  FIX: LSTM — 1997        │      │  │                      │  │
│  Hochreiter & Schmidhuber│      │  │ WITH MOMENTUM:       │  │
│                          │      │  │   ~~~smooth~~~       │  │
│  added a memory highway  │      │  │      ~~~curve~~~     │  │
│  that gradients could    │      │  │           ●          │  │
│  travel without fading   │      │  └──────────────────────┘  │
└──────────────────────────┘      └────────────────────────────┘
                        │
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  THE PROBLEM NOBODY SOLVED:           │
        │  the learning rate                    │
        │                                       │
        │  how BIG a step do you take?          │
        │                                       │
        │  too big:  ┌──────────────────────┐   │
        │            │      🌋              │   │
        │            │  step ──────────────►│   │
        │            │  fly right over      │   │
        │            │  the valley          │   │
        │            └──────────────────────┘   │
        │                                       │
        │  too small: ┌──────────────────────┐  │
        │             │  ● step              │  │
        │             │    ● step            │  │
        │             │      ● step          │  │
        │             │        takes         │  │
        │             │        forever       │  │
        │             └──────────────────────┘  │
        │                                       │
        │  and DIFFERENT parameters need        │
        │  DIFFERENT step sizes                 │
        │  one size fits all = bad              │
        └───────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────────────┐
          │             │                     │
          ▼             ▼                     ▼
┌─────────────┐  ┌────────────┐   ┌────────────────────────┐
│ ADAGRAD     │  │ RMSPROP    │   │ ADAM — 2014            │
│ 2011        │  │ 2012       │   │ Kingma & Ba            │
│             │  │            │   │                        │
│ first time  │  │ Hinton     │   │ combines:              │
│ each param  │  │ (slides,   │   │  ✓ momentum            │
│ gets its    │  │ no paper)  │   │  ✓ per-parameter rates │
│ own step    │  │            │   │  ✓ running history     │
│             │  │ fixed a    │   │                        │
│ problem:    │  │ problem    │   │ became the default     │
│ step gets   │  │ AdaGrad    │   │ for almost everything  │
│ smaller     │  │ had with   │   │ for 10 years           │
│ forever,    │  │ shrinking  │   │                        │
│ eventually  │  │ steps      │   │ still used everywhere  │
│ stops       │  └────────────┘   │ today                  │
└─────────────┘                   └────────────────────────┘
                        │
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  2012 — THE UNLOCK                    │
        │                                       │
        │  AlexNet wins ImageNet competition    │
        │  by a massive margin                  │
        │                                       │
        │  secret ingredient: GPUS              │
        │                                       │
        │  gradient descent was always right    │
        │  we just didn't have enough compute   │
        │  to make it work properly             │
        │                                       │
        │  like having a recipe for a cake      │
        │  but only owning a match              │
        │  not an oven                          │
        │                                       │
        │  GPUs were the oven                   │
        └───────────────────────────────────────┘
                        │
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  2015 — BATCH NORM & RESIDUALS        │
        │                                       │
        │  two tricks that finally killed       │
        │  the vanishing gradient problem       │
        │                                       │
        │  BATCH NORM:                          │
        │  ┌──────────────────────────────────┐ │
        │  │ keeps the signal consistent      │ │
        │  │ as it travels through layers     │ │
        │  │                                  │ │
        │  │ like adjusting the volume        │ │
        │  │ so every layer hears clearly     │ │
        │  └──────────────────────────────────┘ │
        │                                       │
        │  RESIDUAL CONNECTIONS:                │
        │  ┌──────────────────────────────────┐ │
        │  │ add a direct shortcut            │ │
        │  │ that skips some layers           │ │
        │  │                                  │ │
        │  │ input ──►[layer]──►[+]──► out   │ │
        │  │             │       ▲            │ │
        │  │             └───────┘            │ │
        │  │         skip connection          │ │
        │  │                                  │ │
        │  │ gradient can travel the          │ │
        │  │ shortcut instead of fading       │ │
        │  │ through all the layers           │ │
        │  └──────────────────────────────────┘ │
        │                                       │
        │  finally: 100-layer networks learn    │
        └───────────────────────────────────────┘
                        │
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  2017 — ADAMW                         │
        │  Loshchilov & Hutter                  │
        │                                       │
        │  small fix to Adam                    │
        │  separates two things that            │
        │  were incorrectly combined            │
        │                                       │
        │  still the actual default             │
        │  in most training today               │
        └───────────────────────────────────────┘
                        │
                        │  7 years pass
                        │  Adam reigns supreme
                        │  nobody seriously challenges it
                        │
                        ▼
        ┌───────────────────────────────────────────────────┐
        │  2024 — MUON                                      │
        │  Keller Jordan et al.                             │
        │                                                   │
        │  first real challenger to Adam's throne           │
        │                                                   │
        │  what Adam does:                                  │
        │  ┌───────────────────────────────────────────┐   │
        │  │ treats every single weight as a number    │   │
        │  │ adjusts each one independently            │   │
        │  │ doesn't care that weights form a MATRIX   │   │
        │  │                                           │   │
        │  │ like adjusting every pixel in a photo     │   │
        │  │ one by one                                │   │
        │  │ without seeing the whole picture          │   │
        │  └───────────────────────────────────────────┘   │
        │                                                   │
        │  what Muon does:                                  │
        │  ┌───────────────────────────────────────────┐   │
        │  │ looks at the gradient as a MATRIX         │   │
        │  │ asks: what's the best step for this       │   │
        │  │ shape of thing?                           │   │
        │  │                                           │   │
        │  │ "orthogonalises" the gradient             │   │
        │  │ (makes the update point in a direction    │   │
        │  │  that matches the geometry of the matrix) │   │
        │  │                                           │   │
        │  │ like adjusting a photo by understanding   │   │
        │  │ the relationship between pixels           │   │
        │  │ not just each pixel alone                 │   │
        │  └───────────────────────────────────────────┘   │
        │                                                   │
        │  result: trains faster                            │
        │          fewer compute steps to same quality      │
        │          used in real frontier models now         │
        │          (Moonlight 16B, Kimi K2)                 │
        └───────────────────────────────────────────────────┘
```

---

## The characters, all in one place

```
┌──────────────────────────────────────────────────────────────────────┐
│ WHO          │ WHEN │ FIELD              │ ORIGINAL PROBLEM          │
├──────────────────────────────────────────────────────────────────────┤
│ Cauchy       │ 1847 │ Pure math          │ Solve equations           │
│ Hadamard     │ 1907 │ Pure math          │ Same (separately)         │
│ Robbins &    │ 1951 │ Statistics         │ Find things in noisy data │
│ Monro        │      │                    │                           │
│ Rosenblatt   │ 1957 │ Neuroscience/CS    │ Make a fake neuron        │
│ Polyak       │ 1964 │ Math optimization  │ Fix zig-zagging           │
│ Linnainmaa   │ 1970 │ CS (Finland)       │ Automate math on graphs   │
│ Werbos       │ 1974 │ Economics (Harvard)│ Apply to neural nets      │
│ Rumelhart,   │ 1986 │ Cognitive science  │ How do brains learn?      │
│ Hinton,      │      │                    │                           │
│ Williams     │      │                    │                           │
│ Hochreiter & │ 1997 │ CS                 │ Fix fading signal problem │
│ Schmidhuber  │      │                    │                           │
│ Duchi et al. │ 2011 │ ML theory          │ Per-parameter step sizes  │
│ Krizhevsky   │ 2012 │ Computer vision    │ Image recognition         │
│ (AlexNet)    │      │                    │                           │
│ Kingma & Ba  │ 2014 │ ML                 │ Better optimizer for nets │
│ Loshchilov & │ 2017 │ ML                 │ Small fix to Adam         │
│ Hutter       │      │                    │                           │
│ Jordan et al.│ 2024 │ ML optimization    │ Match update to geometry  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Why this whole story matters

Almost nobody in this family tree was trying to build AI.

```
CAUCHY          ──► wanted to solve equations
ROBBINS & MONRO ──► wanted to do statistics
RUMELHART       ──► wanted to understand how brains learn
HINTON          ──► wanted to model human cognition
LINNAINMAA      ──► wanted to automate differentiation

ALL of them      ──► accidentally built the engine
                     that runs every AI in existence
```

The other big thing: this is a tool invented in 1847 that we KNEW was bad from the start. Cauchy knew it zig-zagged. 175 years of patches and improvements later, we're still using it. We never found something fundamentally different. We made it less bad, over and over, until it was good enough.

Muon in 2024 is the first time someone asked a genuinely different question: *what geometry does this problem actually have?* Instead of: *how do we tune the step size better?* Whether that turns into something fundamentally new or is just another patch — unknown.

---

## The one insight that ties everything together

Gradient descent is not a computer science invention. It's not an AI invention. It's a mathematical tool from 1847 for solving equations, borrowed by statisticians in 1951, borrowed again by neuroscientists in the 1980s, handed to computer scientists, who had access to GPUs big enough to make it work, and now it runs on tens of thousands of chips simultaneously training models that write poetry and reason about cancer.

The tool was always there. What changed was the scale we could run it at.

```
1847    2 page paper
        1 mathematician
        solve some equations

2025    thousands of chips
        billions of parameters
        train at a scale Cauchy
        couldn't have imagined
        using the exact same idea

step in direction of negative gradient
repeat
```

Same sentence. Completely different universe.
