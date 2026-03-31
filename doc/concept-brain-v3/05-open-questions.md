# Open Questions

Things we discussed but haven't fully resolved.

## 1. How does self-encoding actually work?

Initially we use CLIP vectors to position concepts. But we want the space to evolve.

Question: when and how does the brain's space diverge from CLIP's?
- Option A: concept vectors are the AVERAGE of all observations → naturally diverges as more modalities added
- Option B: concept vectors are defined purely by edge neighbors → position = mean(connected concepts)
- Option C: hybrid — start with CLIP, gradually blend toward relationship-defined positions

We said "the dictionary defines itself" but HOW exactly?

## 2. Relation type discovery

We said "types emerge from data, not pre-defined." But:
- How does the parser know "apple is red" has a "color" relation?
- Does it start with pre-defined types and discover new ones?
- Or does it start with NO types and discover ALL of them?

Probably: bootstrap with 10-20 common types (is_a, has_property, action, location, color, size), then let new types emerge when patterns don't fit existing ones.

## 3. The generalization question

One-shot storage stores SPECIFIC facts. But intelligence requires GENERALIZATION.
- "This apple is red" = fact
- "Apples are generally red" = generalization
- "Fruit comes in colors" = abstraction

How does the concept brain generalize?
- Answer: meta-edges. When it sees 10 (fruit→color) edges for different fruits, it creates a meta-pattern: "fruits have colors." That IS generalization.
- But: does this actually work in practice? Need to test.

## 4. Scale — 5K concepts enough?

482K training items → ~5K unique concepts (estimate).
Is 5K enough for useful intelligence?
- English has ~170K words in common use
- A child at age 6 knows ~13,000 words
- Basic functional vocabulary: ~3,000 words

5K concepts might be enough for BASIC understanding.
But limited. Need to grow with more data.

## 5. The composition problem in detail

"Purple dog" = compose purple + dog. Simple.
"A dog that is ironically dressed as a cat for Halloween" = ???

Composition gets HARD with complex modifiers, nested relations, abstract concepts. How far can simple vector arithmetic go?

## 6. Generation quality

Templates produce "the apple is red" but not "crisp autumn apples hung like red jewels from gnarled branches."

Is template-based generation ENOUGH for the project goals?
Or do we need something more (learned sequence model)?

The honest answer: templates are the v1. If the concept graph proves useful, a more sophisticated generator can be added later. The KNOWLEDGE is the hard part. Generation is presentation.

## 7. Does slow learning (FF/gradient) still have a role?

We're removing gradient learning for facts. But maybe:
- The PARSER needs to be trained (learns to extract better relations over time)
- The GENERATOR needs to learn templates from data
- The CLUSTERING needs to refine over time
- The ENCODERS still distill from CLIP (gradient-based)

Maybe: one-shot for FACTS, gradient for SKILLS (parsing, generating, encoding).
Facts = what you know. Skills = how you process.

## 8. Integration with existing frontend

The React/Three.js frontend visualizes the brain as a neural graph.
The concept graph is a different structure — fewer nodes, typed edges, hierarchical.
Visualization needs to change to show:
- Concept nodes (sized by confidence, colored by cluster)
- Typed edges (different colors/styles per relation type)
- Clusters (spatial grouping)
- Activation patterns (which concepts light up for a query)

This is actually EASIER to visualize than 10K neurons — 5K labeled concepts with meaningful connections vs anonymous neurons with untyped edges.
