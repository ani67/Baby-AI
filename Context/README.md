# Context Folder — Navigation Guide
*This folder contains the design documents for the continuous local AI build.*
*Start with build-spec.md. Everything else is referenced from there.*

---

## File map

```
build-spec.md                    ← MASTER DOCUMENT
                                   contains component specs, session prompts,
                                   interface contracts, and success tests
                                   READ THIS FIRST AND ALWAYS

continuous-training-paradigm.md  ← WHY the freeze cycle is wrong
                                   relevant for: Component 3 (double-buffer)
                                   background for: why this architecture exists

memory-and-wiring-problem.md     ← WHY memory needs two layers
                                   relevant for: Components 4, 6 (episodic store, consolidation)
                                   background for: what problem we're solving

from-scratch-minimum-model.md    ← WHY the model is small + knowledge split
                                   relevant for: Component 9 (knowledge store)
                                   background for: why facts don't live in weights

model-dialogue-training.md       ← WHY teacher ensemble works
                                   relevant for: Component 8 (teacher ensemble)
                                   background for: why this beats search

brain-architecture-scaling.md    ← WHY internal states matter
                                   relevant for: Components 5, 7 (importance scorer, state monitor)
                                   background for: the biological grounding

modality-emergence-self.md       ← WHY self-narrative and multimodality
                                   relevant for: Component 10 (orchestrator, self-narrative)
                                   background for: what the system is pointing toward
```

---

## How to use these files with Claude Code

### At the start of EVERY session, say:

```
Read context/build-spec.md first.
I am building Component [N].
Components 1 through [N-1] are complete and tested.
The spec for Component [N] starts at the heading "COMPONENT [N]".
Build only that component.
```

### If Claude Code needs background on WHY something is designed a certain way:

```
Read context/[relevant file] for the reasoning behind this design decision.
```

### You do NOT need Claude Code to read all files every session.
build-spec.md contains everything needed to build.
The other files are for when you want to understand or challenge a decision.

---

## Component → file reference

| Component | Build spec section | Background file if needed |
|-----------|-------------------|--------------------------|
| 1. Base inference | COMPONENT 1 | — |
| 2. LoRA training | COMPONENT 2 | continuous-training-paradigm.md |
| 3. Double-buffer | COMPONENT 3 | continuous-training-paradigm.md |
| 4. Episodic store | COMPONENT 4 | memory-and-wiring-problem.md |
| 5. Importance scorer | COMPONENT 5 | brain-architecture-scaling.md |
| 6. Consolidation | COMPONENT 6 | memory-and-wiring-problem.md |
| 7. State monitor | COMPONENT 7 | brain-architecture-scaling.md |
| 8. Teacher ensemble | COMPONENT 8 | model-dialogue-training.md |
| 9. Knowledge store | COMPONENT 9 | from-scratch-minimum-model.md |
| 10. Orchestrator | COMPONENT 10 | modality-emergence-self.md |
