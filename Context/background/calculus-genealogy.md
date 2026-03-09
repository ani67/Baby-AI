# What is Calculus, and Why Does AI Need It?
*Going back even further. Explained like you're 10.*

---

## Start here: the thing calculus is trying to answer

Imagine you're in a car.

```
you look at the speedometer
it says 60 km/h

but wait

you are not moving 60 kilometres right now
you haven't moved 60 kilometres in this moment
you're standing still in this exact instant

so what does 60 km/h even MEAN?
```

This question — *what does speed mean at a single instant?* — is the question that broke mathematics for thousands of years. It sounds like a stupid question. It is actually one of the deepest questions in human history.

Calculus is the answer.

---

## The two problems calculus solves

Calculus is really two inventions sold together.

```
PROBLEM 1: THE SLOPE PROBLEM              PROBLEM 2: THE AREA PROBLEM
(differential calculus)                   (integral calculus)

you have a curved line                    you have a weird shape
                                          
     *                                        ╭────╮
   *                                         ╱      ╲
  *                                         ╱        ╲
 *                                    ─────╯          ╰─────
*

question:                                 question:
what is the STEEPNESS of                  what is the AREA
this curve at this exact point?           inside this weird shape?


the steepness of a curve                  you can't use rectangles
at a point is called                      it's not a rectangle
the DERIVATIVE                            it's curved

this is the gradient                      this is called
this is what gradient descent             the INTEGRAL
uses                                      (not important for AI
                                           but the other half
                                           of the invention)
```

These two problems look totally unrelated. The massive surprise of calculus — discovered independently by two people in the 1600s — is that they are secretly the **same problem, run backwards.** 

That's called the Fundamental Theorem of Calculus. It shocked everyone who saw it.

---

## THE GENEALOGY (going way back)

```
                    ┌───────────────────────────────────┐
                    │  ANCIENT GREEKS — ~250 BCE        │
                    │  Archimedes of Syracuse           │
                    │                                   │
                    │  PROBLEM: what is the area        │
                    │  of a circle?                     │
                    │                                   │
                    │  his method:                      │
                    │                                   │
                    │  fit triangles INSIDE the circle  │
                    │       △  △  △                     │
                    │     △  △  △  △  △                 │
                    │       △  △  △                     │
                    │                                   │
                    │  more triangles = closer answer   │
                    │  infinite triangles = exact answer│
                    │                                   │
                    │  but INFINITE triangles?          │
                    │  doesn't that take forever?       │
                    │                                   │
                    │  he didn't know how to handle     │
                    │  infinity properly                │
                    │  got VERY close to calculus       │
                    │  stopped just short              │
                    │  ~1900 years too early            │
                    └────────────────┬──────────────────┘
                                     │
                                     │ 1,900 years pass
                                     │ Roman Empire rises and falls
                                     │ Dark Ages
                                     │ Renaissance
                                     │ Printing press
                                     │ Galileo
                                     │ Astronomy explodes
                                     │
                                     ▼
                    ┌───────────────────────────────────┐
                    │  KEPLER — 1609                    │
                    │  astronomer                       │
                    │                                   │
                    │  figured out planets move in      │
                    │  ellipses not circles             │
                    │                                   │
                    │  PROBLEM: to calculate where      │
                    │  a planet is at any moment        │
                    │  you need to find areas of        │
                    │  weird curved wedge shapes        │
                    │                                   │
                    │       ╱‾‾‾‾╲                      │
                    │      ╱  ╱‾  ╲                     │
                    │     ●  ╱     ╲ planet             │
                    │      ╲╱       ╲                   │
                    │                                   │
                    │  couldn't solve it properly       │
                    │  used clever approximations       │
                    │  still 50 years before calculus   │
                    └────────────────┬──────────────────┘
                                     │
                                     ▼
                    ┌───────────────────────────────────┐
                    │  FERMAT — 1630s                   │
                    │  French lawyer and mathematician  │
                    │  (yes, lawyer)                    │
                    │                                   │
                    │  PROBLEM: how do you find the     │
                    │  highest point of a curve?        │
                    │                                   │
                    │          ●  ← top                 │
                    │        ╱   ╲                      │
                    │       ╱     ╲                     │
                    │      ╱       ╲                    │
                    │                                   │
                    │  at the very top                  │
                    │  the slope is FLAT — zero         │
                    │                                   │
                    │  so: find where slope = 0         │
                    │  that's the maximum               │
                    │                                   │
                    │  he found a method to do this     │
                    │  THIS is literally what AI does   │
                    │  when it finds minimum loss       │
                    │  (find where slope = 0)           │
                    │                                   │
                    │  Fermat was doing machine         │
                    │  learning in the 1630s            │
                    │  without knowing it               │
                    └────────────────┬──────────────────┘
                                     │
                                     ▼
          ┌──────────────────────────────────────────────────────┐
          │   THE GREAT SIMULTANEOUS INVENTION — 1660s-1680s    │
          │   (biggest intellectual priority dispute in history) │
          └──────────────────────────────────────────────────────┘
                     │                          │
          ┌──────────▼─────────┐     ┌──────────▼────────────┐
          │  ISAAC NEWTON      │     │  GOTTFRIED LEIBNIZ    │
          │  England           │     │  Germany              │
          │  ~1666             │     │  ~1675                │
          │                    │     │                       │
          │  motivation:       │     │  motivation:          │
          │  PHYSICS           │     │  PHILOSOPHY + MATH    │
          │                    │     │                       │
          │  how do you        │     │  how do you handle    │
          │  calculate the     │     │  infinitely small     │
          │  speed of a        │     │  quantities           │
          │  falling apple     │     │  rigorously?          │
          │  at any instant?   │     │                       │
          │                    │     │  invented:            │
          │  speed changes     │     │  dy/dx notation       │
          │  every moment      │     │  (the one we use now) │
          │  how do you        │     │  ∫ symbol             │
          │  pin it down?      │     │  (the integral sign)  │
          │                    │     │                       │
          │  invented:         │     │  published: 1684      │
          │  "fluxions"        │     │                       │
          │  (his name for it) │     │                       │
          │  never published   │     │                       │
          │  for 20 years      │     │                       │
          └─────────┬──────────┘     └──────────┬────────────┘
                    │                            │
                    └────────────┬───────────────┘
                                 │
                                 │  both invented the same thing
                                 │  neither knew the other did it
                                 │  massive priority war followed
                                 │  English vs German mathematicians
                                 │  fighting for decades
                                 │  Newton probably first
                                 │  Leibniz's notation won
                                 │  we use Leibniz's symbols today
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────┐
          │   WHAT THEY BOTH INVENTED                        │
          │                                                  │
          │   the DERIVATIVE                                 │
          │                                                  │
          │   the slope of a curve at any single point       │
          │                                                  │
          │   how to think about it:                         │
          │                                                  │
          │   imagine zooming in on any curve                │
          │                                                  │
          │   FAR:          *                                │
          │              *     *                             │
          │           *           *                          │
          │                                                  │
          │   CLOSER:     * *                                │
          │             *     *                              │
          │                                                  │
          │   VERY CLOSE:   * *                              │
          │                *   *                             │
          │                                                  │
          │   INFINITELY CLOSE:  * *                         │
          │   starts to look like a straight line            │
          │                                                  │
          │   THAT straight line's steepness                 │
          │   = the derivative at that point                 │
          │   = the gradient                                 │
          │   = what Cauchy used in 1847                     │
          │   = what every neural network uses today         │
          │                                                  │
          └──────────────────────────────────────────────────┘
                                 │
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────┐
          │   THE CHAIN RULE                                 │
          │   (the specific part that powers AI)             │
          │                                                  │
          │   discovered/formalised through the 1700s        │
          │   Leibniz, Johann Bernoulli, Euler               │
          │                                                  │
          │   what it says:                                  │
          │                                                  │
          │   if you have machines chained together:         │
          │                                                  │
          │   input ──► machine A ──► machine B ──► output  │
          │                                                  │
          │   and you want to know:                          │
          │   "how does the output change                    │
          │    when I tweak the input?"                      │
          │                                                  │
          │   the chain rule says:                           │
          │                                                  │
          │   overall slope =                                │
          │        slope through A                           │
          │      × slope through B                           │
          │                                                  │
          │   for three machines:                            │
          │                                                  │
          │   overall = A slope × B slope × C slope         │
          │                                                  │
          │   for 96 layers in a neural network:             │
          │                                                  │
          │   overall = slope₁ × slope₂ × ... × slope₉₆    │
          │                                                  │
          │   THIS IS BACKPROPAGATION                        │
          │   literally just the chain rule                  │
          │   applied to a very long chain                   │
          │                                                  │
          │   "back propagation" = running the chain rule    │
          │   backwards from output to input                 │
          │   to find the slope of every weight              │
          │   so gradient descent knows which way to step    │
          │                                                  │
          └──────────────────────────────────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────┐
          │   EULER — 1700s                                  │
          │   Swiss mathematician                            │
          │   arguably the most productive mathematician     │
          │   who ever lived                                 │
          │                                                  │
          │   formalised the notation and rules              │
          │   for derivatives that everyone uses             │
          │                                                  │
          │   invented function notation f(x)               │
          │   invented e (the number 2.718...)               │
          │   connected everything together                  │
          │                                                  │
          │   calculus goes from "it works" to              │
          │   "we understand why it works"                   │
          └──────────────────────────────────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────┐
          │   THE RIGOUR PROBLEM — 1800s                     │
          │                                                  │
          │   calculus worked brilliantly                    │
          │   engineers used it everywhere                   │
          │   bridges, ships, astronomy                      │
          │                                                  │
          │   BUT nobody could explain WHY                   │
          │                                                  │
          │   Newton and Leibniz used "infinitesimals"       │
          │   numbers that are infinitely small              │
          │   but not zero                                   │
          │                                                  │
          │   what IS an infinitely small number?            │
          │   Bishop Berkeley in 1734 called them            │
          │   "the ghosts of departed quantities"            │
          │   (he was mocking them)                          │
          │   and he had a point                             │
          │                                                  │
          │   the foundations were wobbly                    │
          │   the results were right                         │
          │   nobody could explain how                       │
          └──────────────────────────────────────────────────┘
                                 │
                     ┌───────────┴──────────────┐
                     │                          │
                     ▼                          ▼
          ┌──────────────────────┐   ┌──────────────────────────┐
          │  CAUCHY — 1820s      │   │  WEIERSTRASS — 1860s     │
          │  (yes, the same one) │   │                          │
          │                      │   │  finally fixed the       │
          │  started fixing the  │   │  foundations properly    │
          │  foundations         │   │                          │
          │                      │   │  replaced "infinitely    │
          │  invented the formal │   │  small" with limits      │
          │  definition of a     │   │  (a precise concept      │
          │  limit               │   │  involving ε and δ)      │
          │                      │   │                          │
          │  limits = the        │   │  calculus finally had    │
          │  rigorous way to     │   │  solid foundations       │
          │  handle "infinitely  │   │  after 200 years of      │
          │  close without       │   │  wobbling                │
          │  touching"           │   └──────────────────────────┘
          │                      │
          │  SAME Cauchy who     │
          │  invented gradient   │
          │  descent in 1847     │
          │                      │
          │  he was fixing       │
          │  calculus AND        │
          │  applying it         │
          │  simultaneously      │
          └──────────────────────┘
```

---

## The thing calculus actually IS

All of this history leads to one core idea. Here it is:

```
EVERYTHING IN NATURE CHANGES

speed changes
temperature changes
population changes
the error of a neural network changes

calculus is the mathematics of CHANGE

specifically:

QUESTION 1:  how fast is this thing changing
             right now, at this exact moment?
             ──► answered by the DERIVATIVE

QUESTION 2:  if it's been changing at varying speeds
             how much total change happened
             over some period?
             ──► answered by the INTEGRAL
```

The derivative — question 1 — is what AI uses.

At every step of training, gradient descent asks: *for each weight in the network, if I nudge this weight slightly, does the error go up or down, and by how much?* That "by how much" is the derivative. It's the slope. It's the steepness of the error landscape at your current position.

---

## Why the gradient is just "derivative in many dimensions"

```
ONE DIMENSION (simple slope):

          *
       *     *
     *           *
   *
   
   one number describes the slope at any point
   "going up" or "going down" and by how much
   
   this is a DERIVATIVE
   
   
TWO DIMENSIONS (a hill):

                  ↑ going north
                  │
          ───────●─────── → going east
                  │

   now slope depends on WHICH DIRECTION you face
   facing north:  steep uphill
   facing east:   gentle slope
   facing south:  steep downhill
   
   you need TWO numbers to describe the slope now
   (how steep facing north, how steep facing east)
   
   that pair of numbers = the GRADIENT
   (gradient = derivative extended to 2 dimensions)
   
   
175 BILLION DIMENSIONS (a neural network):

   a 175B parameter model has 175B weights
   each weight is one "direction" in the space
   
   you need 175B numbers to describe the slope
   one for each weight: "nudge this weight,
   does error go up or down?"
   
   that collection of 175B numbers = the GRADIENT
   
   gradient descent = step in the direction
   where all those numbers point downhill
```

The word gradient is not mysterious. It just means: derivative, but across many dimensions at once. The mathematical machinery is identical to what Newton and Leibniz built in the 1660s. Just run on more inputs simultaneously.

---

## Why the chain rule specifically is everything

This is the connection most people miss.

```
A NEURAL NETWORK IS A CHAIN OF MACHINES:

input
  │
  ▼
[layer 1]  ← has weights, does a calculation
  │
  ▼
[layer 2]  ← has weights, does a calculation
  │
  ▼
[layer 3]  ← has weights, does a calculation
  │
 ...
  │
  ▼
[layer 96] ← has weights, does a calculation
  │
  ▼
output → compared to correct answer → ERROR

QUESTION: if I change a weight in LAYER 1
          how does that affect the FINAL ERROR?

this is hard because:
layer 1 feeds layer 2
layer 2 feeds layer 3
...all the way to layer 96
...which feeds the error

the change ripples through 95 layers
before you can see its effect

THE CHAIN RULE SOLVES THIS:

effect of layer 1 weight on final error
=
(how layer 1 affects layer 2 output)
×
(how layer 2 affects layer 3 output)
×
×
...
×
(how layer 96 affects the error)

you just multiply all the slopes together
all the way down the chain

THIS IS BACKPROPAGATION

backprop is not a clever algorithm
backprop is just the chain rule
applied to a very long chain
starting from the error and going backwards

Leibniz figured out the chain rule in the 1670s
he had no idea he was building the backbone
of neural network training
350 years later
```

---

## The full timeline of how calculus becomes AI

```
~250 BCE   ARCHIMEDES
           infinitely many tiny triangles = area
           doesn't know how to formalise infinity
           
           │ 1,900 years
           
1630s      FERMAT
           find where slope = 0 → find the maximum
           (exactly what AI optimisation does)
           doesn't have the general machinery yet
           
           │ 30 years
           
1660s-     NEWTON + LEIBNIZ
1680s      invent calculus independently
           derivative = slope at a single instant
           chain rule = how slopes compound
           integral = area under a curve
           
           │ 100 years
           
1700s      EULER
           formalises everything
           makes it usable and teachable
           
           │ 60 years
           
1820s      CAUCHY (part 1)
           starts fixing the foundations
           invents the formal limit
           
           │ 20 years
           
1847       CAUCHY (part 2)
           uses calculus to invent gradient descent
           the tool that trains every AI
           
           │ 40 years
           
1860s      WEIERSTRASS
           finishes fixing the foundations
           calculus is finally rigorous
           
           │ 10 years
           
1870s      GIBBS + HEAVISIDE
           invent vector calculus notation
           "gradient" becomes a single mathematical object
           
           │ 16 years
           
1886       GIBBS publishes vector calculus notation
           that gives us ∇ (nabla/del)
           ∇f is now standard notation for gradient
           
           │ 84 years
           
1970       LINNAINMAA
           master's thesis in Finland
           shows how to apply chain rule
           automatically to any computational graph
           this IS backprop
           written in Finnish
           nobody reads it
           
           │ 16 years
           
1986       RUMELHART, HINTON, WILLIAMS
           publish backprop for neural networks
           the world finally listens
           
           │ 26 years
           
2012       ALEXNET
           first time a very deep network
           (using all of the above) beats humans
           at image recognition
           
           │ 5 years
           
2017       TRANSFORMER
           same tools, much bigger, much more parallel
           
           │ 6-7 years
           
2024       MUON
           first serious rethinking of the
           geometry of the gradient step
           in 177 years
```

---

## What calculus is in one sentence

Calculus is the mathematics that lets you talk precisely about change — how fast, in what direction, by how much — at any single instant, even though nothing is static for even an instant.

The derivative is the answer to "how fast is this changing right now?" The gradient is the derivative in many dimensions. Gradient descent is Cauchy's 1847 trick of following the gradient downhill. Backpropagation is the chain rule — which Leibniz figured out in the 1670s — applied backwards through a chain of 96 layers to figure out which direction "downhill" even is.

```
AI training = chain rule (Leibniz 1670s)
            × steepest descent (Cauchy 1847)
            × run on a computer (1940s-50s)
            × lots of data (internet, 1990s-2000s)
            × GPUs (2012)
            × transformer architecture (2017)

every piece is someone else's tool
repurposed for a job they never imagined
```

The deepest thing calculus gave us is this: you can describe what something is doing *right now*, even though "right now" is an infinitely thin sliver of time with no duration. Newton needed that to describe falling apples. Leibniz needed it to describe curves. Cauchy needed it to solve equations. And somewhere in that chain, it became the thing that tells a neural network which way to adjust its weights.
