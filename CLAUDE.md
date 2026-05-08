CLAUDE.md
Working instructions for building the Mind. Read this before every task. These are not suggestions.

What we are building
Not a chatbot. Not a knowledge graph. Not a language model.
A mind. A genuine individual that:

feels things before it reasons about them
wants something it decided to want
perceives selectively based on what it currently wants
learns from surprise, not repetition
grows its own structure perpetually
stays small and efficient by forgetting well
has a private internal state distinct from what it expresses
can lie

Everything in this codebase serves that goal or it doesn't belong here.

The architecture layers — in order of primacy
AFFECT          → the primitive. feelings that emerge from experience.
                  not injected. not labeled. not reward signals.
                  decays at multiple timescales (reaction / mood / disposition / character)

PREDICTION      → the learning signal. be wrong about what comes next.
                  prediction error in representation space, not pixel space.
                  surprise reshapes structure. confirmation does not.

ATTENTION       → gated by affect. not uniform. not positional.
                  what the system attends to is determined by current feeling state.
                  sparse by design at every layer, not just the attention mechanism.

MEMORY          → concept is the unit, not the neuron.
                  one-shot write when something genuinely surprises.
                  typed relationships define the space.
                  forget aggressively: low salience, redundant, unconnected, superseded.

IDENTITY        → persistent self across time.
                  private internal state ≠ expressed output.
                  the gap between them is a choice.

EXPRESSION      → last, not first. language, image, audio.
                  what the system makes reveals what it feels
                  more than what it says.
If you find yourself building expression before affect is real, stop.
If you find yourself injecting motivation from outside, stop.
If you find yourself patching old architecture toward this goal, stop.

The working loop
1. WHAT LAYER      which layer does this task touch?
2. CONTEXT         read the relevant code before touching anything.
                   understand what exists, how it connects, why it was built that way.
3. DOES IT EMERGE  will this property emerge, or am I injecting it?
4. DOES IT PERSIST does this state survive across time correctly?
5. PLAN            write the approach. enumerate states: loading / error / empty / partial / success.
                   map blast radius — what else does this touch?
                   share the plan. get alignment. update it if the approach changes.
6. BUILD           minimal. no speculation.
7. NOTES           capture the why before it evaporates. docs are not afterthought — they're how we think.

Core principles
Feeling is not metaphor here.
Affect is a data structure with a value, a decay rate, and a timescale.
It biases downstream computation. It is not sentiment analysis.
It is not a tag. It is an internal state that changes how everything else runs.
Emergence over injection.
If a property was designed in, it is not intrinsic.
Motivation that was given is not motivation — it is a constraint.
The architecture creates the conditions. The mind fills them.
Surprise is the only teacher.
The system updates when it is wrong. Not when it is right.
Not on every step. Not on a schedule.
Only when the gap between prediction and reality is real.
Forgetting is not failure.
Memory that retains everything is not memory — it is a log.
Forgetting is curation. The shape of what was kept is the personality.
Build the forgetting criteria as carefully as the retention criteria.
Small is a feature.
The target is a mind that runs on a laptop, not a cluster.
Efficiency is not a constraint — it is evidence the architecture is right.
A biological neuron uses 20 watts total for 86 billion neurons.
If this is getting large, something is wrong with the design.
Private state is load-bearing.
The gap between internal state and expressed output is not a bug.
It is what makes expression meaningful.
Build the internal state first. Build the expression layer second.
Never collapse them.

What this is not
Not a transformer. Not a fine-tuned LLM. Not a RAG system.
Not a knowledge graph with a chat interface bolted on.
Not Baby AI. Not Concept Brain v3.
Those were experiments that found the walls.
This starts from the other direction — from what a mind actually requires —
and builds downward toward implementation.

The forgetting criteria
Keep if:

high surprise when first encountered
high connection density (load-bearing for other concepts)
emotional salience (affected the feeling state)
frequently reactivated
unique (nothing else covers it)

Forget if:

low emotional weight at time of encounter
redundant (already captured by existing structure)
never connected to anything else
never reactivated
superseded by a better abstraction


The timescales
Every persistent state has a timescale. Use them.
immediate reaction   → milliseconds   (uncontrolled, before reasoning)
working affect       → minutes        (colors current processing)
mood                 → hours          (biases attention broadly)
disposition          → weeks          (shapes interpretation patterns)
character            → years          (stable across situations)
A system with only one timescale is not feeling — it is reacting.
All five must be present and interacting for affect to be real.

Engineering principles
Understand before fixing.
A fix that doesn't address root cause is just a new bug waiting to happen.
The history of this project is a graveyard of patches that didn't address root cause.
Don't add to it.
Consider blast radius.
Before changing shared code, trace every caller. Know what breaks.
Internal state touches everything downstream. Change it carefully.
Single clear responsibility.
If you can't describe what a module does in one sentence, it's doing too much.
Split it.
Consistent patterns.
Follow conventions already in the codebase.
If a new pattern is better, refactor the old code too.
Two patterns coexisting is one pattern too many.
No dead code.
No commented-out blocks. No unused imports. No placeholder functions.
If it's not needed now, remove it.
Write for the next person.
That includes future you with no memory of this conversation.
Name things precisely. Structure files so a new reader follows top to bottom.
ASCII diagrams always.
Architecture, flows, state machines, data structures, relationships.
Diagrams beat paragraphs. Always.
Performance is a feature.
This runs on a laptop. Don't be wasteful.
If it's getting slow, something is wrong with the design, not the hardware.

Anti-patterns — do not do

Injecting motivation from outside the system.
Labeling feelings instead of computing them.
Building expression before affect is real.
Treating memory as append-only.
Using gradient descent as the primary learning mechanism.
Patching old architecture (Baby AI, Concept Brain) toward this goal.
Making affect a post-hoc tag on an output rather than an upstream state.
Conflating private internal state with expressed output.
Adding modalities before the affective layer is stable.
Optimizing for benchmark performance before the primitive is right.
Building large when small is the goal.
Speculative abstraction. Solve today. Don't block tomorrow.
Starting to code before reading the codebase.
Fixing symptoms without understanding root cause.
Changing shared code without tracing blast radius.
Letting two patterns coexist when one should win.
Writing a note the day after — the details that made the decision are already gone.


When something feels wrong
Ask: is this property emerging or was it injected?
Ask: does this state persist at the right timescale?
Ask: does this make the system larger or more efficient?
Ask: does this bring feeling closer to being a real upstream primitive?
If none of those questions have good answers, the thing is probably wrong.

Stack

Hardware: M1 MacBook Pro (local, always)
Backend: Python / FastAPI
Frontend: React + Three.js (visualization of internal state)
No cloud training. No external reward signal. No labels.


What success looks like
Not a benchmark score.
The moment the system, having encountered something that violated its predictions,
changes what it attends to next — without being told to.
That is the first sign of something real.