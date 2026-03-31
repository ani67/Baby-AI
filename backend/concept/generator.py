"""Generator for Concept Brain v3 — turns graph activations into text and images."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch import Tensor

from concept.node import ConceptNode, TypedEdge
from concept.graph import ConceptGraph

try:
    import PIL.Image
except ImportError:
    PIL = None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class GeneratorOutput:
    text: str
    confidence: float
    source_edges: list[TypedEdge] = field(default_factory=list)
    image: PIL.Image.Image | None = None


# ---------------------------------------------------------------------------
# Templates and ordering
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    "is_a":     "{subject} is {article} {object}",
    "color":    "{subject} is {object}",
    "size":     "{subject} is {object}",
    "property": "{subject} is {object}",
    "action":   "{subject} {object}s",
    "has":      "{subject} has {object}",
    "spatial":  "{subject} is {prep} {object}",
    "location": "{subject} is in {object}",
    "contains": "{subject} contains {object}",
    "causes":   "{subject} causes {object}",
    "part_of":  "{subject} is part of {object}",
    "before":   "{subject} comes before {object}",
    "after":    "{subject} comes after {object}",
}

TIER_ORDER: list[str] = [
    "is_a", "property", "color", "size", "action", "has",
    "spatial", "location", "contains", "causes", "part_of",
    "before", "after",
]

# Blending weights for image composition by relation type.
_IMAGE_BLEND: dict[str, float] = {
    "color":    0.3,
    "size":     0.15,
    "location": 0.2,
}

# Vowel-start prefixes for _article heuristic.
_VOWEL_SOUNDS = set("aeiouAEIOU")
_AN_EXCEPTIONS = {"uni", "use", "eur", "one", "once"}
_A_EXCEPTIONS = {"hour", "heir", "hono", "herb"}


# ---------------------------------------------------------------------------
# ConceptGenerator
# ---------------------------------------------------------------------------

class ConceptGenerator:
    """Turns activated concept sub-graphs into human-readable text and images."""

    def __init__(self, vision_decoder=None) -> None:
        self._decoder = vision_decoder  # VisionDecoder | None

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate_text(
        self,
        focal: ConceptNode,
        activated: list[ConceptNode],
        graph: ConceptGraph,
        query_edge_filter: str | None = None,
        max_sentences: int = 5,
    ) -> GeneratorOutput:
        """Render the sub-graph around *focal* as natural-language sentences."""

        edges = graph.get_edges(focal.id, direction="outgoing")
        if not edges:
            return GeneratorOutput(
                text=f"{focal.name or 'this concept'} has no known properties.",
                confidence=0.0,
            )

        # Group edges by base relation type (e.g. "spatial:on" -> "spatial").
        groups: dict[str, list[TypedEdge]] = defaultdict(list)
        for e in edges:
            base = e.relation.split(":")[0]
            groups[base].append(e)

        # Filter to a single relation type if requested.
        if query_edge_filter:
            base_filter = query_edge_filter.split(":")[0]
            groups = {k: v for k, v in groups.items() if k == base_filter}

        # Sort groups by tier order.
        ordered_keys = sorted(
            groups.keys(),
            key=lambda k: TIER_ORDER.index(k) if k in TIER_ORDER else len(TIER_ORDER),
        )

        sentences: list[str] = []
        used_edges: list[TypedEdge] = []
        total_confidence = 0.0

        for rel_type in ordered_keys:
            if len(sentences) >= max_sentences:
                break

            group = groups[rel_type]
            # Top-3 by evidence * strength.
            group.sort(key=lambda e: e.evidence * e.strength, reverse=True)
            top = group[:3]

            for edge in top:
                if len(sentences) >= max_sentences:
                    break

                target = graph.get_node(edge.target_id)
                if target is None:
                    continue

                subject = focal.name or "it"
                obj = target.name or "something"

                prefix = self._confidence_prefix(edge.evidence, edge.strength)
                sentence = self._render(edge.relation, subject, obj)
                if prefix:
                    # Capitalize prefix, lowercase sentence start.
                    sentence = prefix + sentence[0].lower() + sentence[1:]
                sentences.append(sentence)
                used_edges.append(edge)
                total_confidence += edge.strength

        text = ". ".join(sentences)
        if text and not text.endswith("."):
            text += "."

        avg_conf = total_confidence / len(used_edges) if used_edges else 0.0

        return GeneratorOutput(
            text=text,
            confidence=avg_conf,
            source_edges=used_edges,
        )

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate_image(
        self,
        focal: ConceptNode,
        activated: list[ConceptNode],
        graph: ConceptGraph,
    ) -> PIL.Image.Image | None:
        """Compose a vector from focal + property edges and decode to an image."""
        if self._decoder is None:
            return None

        # Start with the image modality vector if available, else the main vector.
        base = focal.modality_vectors.get("image", focal.vector).clone().float()

        # Blend in modifier vectors from property edges.
        edges = graph.get_edges(focal.id, direction="outgoing")
        for edge in edges:
            base_rel = edge.relation.split(":")[0]
            alpha = _IMAGE_BLEND.get(base_rel)
            if alpha is None:
                continue
            target = graph.get_node(edge.target_id)
            if target is None:
                continue
            modifier = target.modality_vectors.get("image", target.vector).float()
            base = (1.0 - alpha) * base + alpha * modifier

        # Normalize.
        base = base / (base.norm() + 1e-8)

        return self._decoder.decode(base)

    # ------------------------------------------------------------------
    # Answer generation (convenience)
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        graph: ConceptGraph,
        operations_module,
    ) -> GeneratorOutput:
        """High-level: parse intent, activate, select, generate text.

        Expects *operations_module* to expose:
            encode(text) -> Tensor
            spread_activation(graph, seed_ids, ...) -> list[ConceptNode]
            typed_inference(graph, relation, target_id) -> list[ConceptNode]
            parse_query_intent(question) -> dict with keys:
                'concepts': list[str], 'relation': str|None
        """
        intent = operations_module.parse_query_intent(question)

        concept_names: list[str] = intent.get("concepts", [])
        relation: str | None = intent.get("relation")

        # Encode and find seed concepts.
        seeds: list[ConceptNode] = []
        for name in concept_names:
            matches = graph.get_by_name(name)
            if matches:
                seeds.append(matches[0])
            else:
                vec = operations_module.encode(name)
                similar = graph.find_similar(vec, k=1)
                if similar:
                    seeds.append(similar[0][0])

        if not seeds:
            return GeneratorOutput(
                text="I don't know enough to answer that yet.",
                confidence=0.0,
            )

        focal = seeds[0]

        # Spread activation from seed nodes.
        seed_ids = [s.id for s in seeds]
        activated = operations_module.spread_activation(graph, seed_ids)

        # If a specific relation was asked, use typed inference.
        edge_filter: str | None = None
        if relation:
            edge_filter = relation

        return self.generate_text(
            focal=focal,
            activated=activated,
            graph=graph,
            query_edge_filter=edge_filter,
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render(self, relation: str, subject: str, obj: str) -> str:
        """Apply the template for *relation* to produce a sentence."""
        base = relation.split(":")[0]
        template = TEMPLATES.get(base)

        if template is None:
            return f"{subject} is related to {obj}"

        # Template-specific substitutions.
        if base == "is_a":
            article = self._article(obj)
            return template.format(subject=subject, article=article, object=obj)

        if base == "spatial":
            # Extract preposition from "spatial:on" -> "on"; default "near".
            parts = relation.split(":", 1)
            prep = parts[1] if len(parts) > 1 else "near"
            return template.format(subject=subject, prep=prep, object=obj)

        if base == "action":
            verb = self._pluralize(obj)
            return f"{subject} {verb}"

        return template.format(subject=subject, object=obj)

    @staticmethod
    def _article(word: str) -> str:
        """Return 'a' or 'an' based on the leading sound of *word*."""
        if not word:
            return "a"
        lower = word.lower()
        # Check exceptions first.
        for prefix in _AN_EXCEPTIONS:
            if lower.startswith(prefix):
                return "a"
        for prefix in _A_EXCEPTIONS:
            if lower.startswith(prefix):
                return "an"
        if lower[0] in _VOWEL_SOUNDS:
            return "an"
        return "a"

    @staticmethod
    def _pluralize(word: str) -> str:
        """Naive English pluralization / third-person-singular verb form."""
        if not word:
            return word
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        if word[-1] in "sxz" or word.endswith("ch") or word.endswith("sh"):
            return word + "es"
        return word + "s"

    @staticmethod
    def _confidence_prefix(evidence: int, strength: float) -> str:
        """Map evidence count to a hedging prefix."""
        if evidence >= 20:
            return ""            # Stated as fact.
        if evidence >= 5:
            return "Usually, "
        if evidence >= 2:
            return "Sometimes, "
        return "Possibly, "
