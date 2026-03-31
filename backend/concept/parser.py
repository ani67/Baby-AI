"""Rule-based concept parser for Concept Brain v3.

Extracts structured triples from text and dataset items.
No external NLP dependencies — pure Python + regex.
~1000 items/sec on typical curriculum data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedTriple:
    subject: str
    object: str
    relation: str
    confidence: float = 1.0
    source_text: str = ""


@dataclass
class ParseResult:
    triples: list[ParsedTriple] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    modality: str = "text"
    item_type: str = "general"
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Word sets
# ---------------------------------------------------------------------------

COLOR_WORDS = frozenset({
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "golden", "silver", "tan", "maroon",
    "navy", "teal", "cyan", "magenta", "violet", "indigo", "beige", "ivory",
    "crimson", "scarlet", "turquoise", "coral", "salmon", "lavender",
})

SIZE_WORDS = frozenset({
    "big", "small", "large", "tiny", "huge", "enormous", "little", "giant",
    "massive", "miniature", "tall", "short", "long", "wide", "narrow",
    "thick", "thin", "deep", "shallow",
})

SPATIAL_PREPS = frozenset({
    "above", "below", "behind", "beside", "near", "under", "over", "inside",
    "outside", "between", "around", "through", "across", "along", "toward",
    "against", "beneath", "among", "within", "beyond", "upon", "onto",
    "next to", "in front of", "on top of", "to the left of", "to the right of",
})

# Multi-word spatial preps that must be detected before tokenizing
_MULTI_WORD_SPATIAL = sorted(
    [p for p in SPATIAL_PREPS if " " in p],
    key=len, reverse=True,
)

COMPOUND_WORDS = frozenset({
    "ice cream", "new york", "los angeles", "san francisco", "hot dog",
    "living room", "dining room", "high school", "middle school",
    "polar bear", "teddy bear", "traffic light", "fire truck",
    "cell phone", "remote control", "baseball bat", "tennis racket",
    "parking lot", "swimming pool", "power plant", "real estate",
    "seat belt", "credit card", "post office", "fire station",
    "roller coaster", "washing machine", "christmas tree",
    "birthday cake", "peanut butter", "orange juice", "apple pie",
})

DETERMINERS = frozenset({"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"})

# Common be-verbs for pattern matching
_BE = frozenset({"is", "are", "was", "were"})

# Words that should not be treated as main verbs
_STOP_VERBS = _BE | {"the", "a", "an", "and", "but", "or", "if", "then", "so", "not", "no"}

# Simple irregular plural map (common cases only)
_IRREGULAR_PLURALS = {
    "children": "child", "men": "man", "women": "woman", "mice": "mouse",
    "teeth": "tooth", "feet": "foot", "geese": "goose", "people": "person",
    "leaves": "leaf", "knives": "knife", "wolves": "wolf", "lives": "life",
    "halves": "half", "selves": "self",
}


# ---------------------------------------------------------------------------
# ConceptParser
# ---------------------------------------------------------------------------

class ConceptParser:

    def __init__(self) -> None:
        self.colors = COLOR_WORDS
        self.sizes = SIZE_WORDS
        self.spatial = SPATIAL_PREPS
        self.compounds = COMPOUND_WORDS
        self._compound_pattern = self._build_compound_re()
        self._multi_spatial_pattern = self._build_multi_spatial_re()

    # -- regex builders ----------------------------------------------------

    def _build_compound_re(self) -> re.Pattern:
        escaped = [re.escape(c) for c in sorted(self.compounds, key=len, reverse=True)]
        return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    def _build_multi_spatial_re(self) -> re.Pattern:
        escaped = [re.escape(p) for p in _MULTI_WORD_SPATIAL]
        return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    # -- normalization -----------------------------------------------------

    def _is_valid_concept(self, name: str) -> bool:
        """Filter out garbage concept names."""
        if not name or len(name) < 2:
            return False
        if len(name) > 40:
            return False  # phrases, not concepts
        if not any(c.isalpha() for c in name):
            return False  # pure dashes, numbers, symbols
        if name.startswith("-") or name.startswith("_"):
            return False
        return True

    def _normalize(self, word: str) -> str:
        """Lowercase, strip punctuation, simple depluralize."""
        w = word.lower().strip()
        w = re.sub(r"[^a-z0-9_\-]", "", w)
        if not w:
            return ""
        # Irregular plurals
        if w in _IRREGULAR_PLURALS:
            return _IRREGULAR_PLURALS[w]
        # Regular depluralize (simple heuristics)
        if len(w) > 3 and w.endswith("ies") and w[-4] not in "aeiou":
            return w[:-3] + "y"
        if len(w) > 3 and w.endswith("ses") and w[-4] not in "s":
            return w[:-2]
        if len(w) > 3 and w.endswith("es") and w[-3] in "shxz":
            return w[:-2]
        if len(w) > 3 and w.endswith("s") and w[-2] not in "su":
            return w[:-1]
        return w

    def _detect_compounds(self, text: str) -> str:
        """Replace known compound phrases with underscored versions."""
        text = self._compound_pattern.sub(lambda m: m.group(0).lower().replace(" ", "_"), text)
        return text

    def _detect_multi_spatial(self, text: str) -> str:
        """Replace multi-word spatial preps with underscored versions."""
        text = self._multi_spatial_pattern.sub(lambda m: m.group(0).lower().replace(" ", "_"), text)
        return text

    # -- clause splitting --------------------------------------------------

    @staticmethod
    def _split_clauses(text: str) -> list[str]:
        """Split on sentence boundaries, 'and', 'but'."""
        # Split on period/semicolon/exclamation/question first
        parts = re.split(r"[.;!?]+", text)
        result = []
        for part in parts:
            # Split on coordinating conjunctions (but keep each clause)
            subs = re.split(r"\b(?:and|but)\b", part)
            for s in subs:
                s = s.strip()
                if s:
                    result.append(s)
        return result

    # -- NP extraction helpers ---------------------------------------------

    @staticmethod
    def _strip_determiners(words: list[str]) -> list[str]:
        """Remove leading determiners from a word list."""
        while words and words[0].lower() in DETERMINERS:
            words = words[1:]
        return words

    @staticmethod
    def _extract_np(words: list[str]) -> str:
        """Join remaining words as a noun phrase, underscore-joined if multi-word."""
        words = ConceptParser._strip_determiners(words)
        if not words:
            return ""
        return "_".join(words) if len(words) > 1 else words[0]

    # -- pattern matching --------------------------------------------------

    def _parse_clause(self, clause: str, source_text: str) -> list[ParsedTriple]:
        """Match a single clause against patterns in priority order."""
        triples: list[ParsedTriple] = []
        clause = clause.strip()
        if not clause:
            return triples

        # Preprocess
        processed = self._detect_compounds(clause)
        processed = self._detect_multi_spatial(processed)
        words_raw = processed.split()
        words = [self._normalize(w) for w in words_raw]
        words = [w for w in words if w]  # drop empty

        if not words:
            return triples

        # Find be-verb position
        be_idx = None
        for i, w in enumerate(words):
            if w in _BE:
                be_idx = i
                break

        if be_idx is not None and be_idx > 0:
            subj_words = words[:be_idx]
            rest = words[be_idx + 1:]
            subj = self._extract_np(subj_words)

            if subj and rest:
                # Pattern 1: Comparison — "X is ADJer than Y"
                if len(rest) >= 3 and rest[-2] == "than":
                    adj_word = rest[0]
                    obj = self._extract_np(self._strip_determiners(rest[-1:]))
                    # Strip -er suffix for relation name
                    base_adj = adj_word
                    if base_adj.endswith("er"):
                        base_adj = base_adj[:-2]
                        if base_adj.endswith(base_adj[-1]) and len(base_adj) > 2:
                            # e.g. "bigger" -> "big" (doubled consonant)
                            base_adj = base_adj[:-1]
                    if obj:
                        triples.append(ParsedTriple(subj, obj, f"comparison:{base_adj}", 0.9, source_text))
                        return triples

                # Pattern 2: Spatial — "X is PREP Y"
                if rest:
                    prep_candidate = rest[0]
                    # Handle underscored multi-word preps
                    prep_check = prep_candidate.replace("_", " ")
                    if prep_check in self.spatial and len(rest) > 1:
                        obj = self._extract_np(self._strip_determiners(rest[1:]))
                        if obj:
                            triples.append(ParsedTriple(subj, obj, f"spatial:{prep_candidate}", 0.9, source_text))
                            return triples

                # Pattern 3: Color — "X is COLOR"
                if len(rest) == 1 and rest[0] in self.colors:
                    triples.append(ParsedTriple(subj, rest[0], "color", 0.95, source_text))
                    return triples

                # Also catch "X is a COLOR noun" — still extract color
                # But first check is_a

                # Pattern 4: Is_a — "X is a/an Y"
                if rest[0] in ("a", "an"):
                    obj = self._extract_np(rest[1:])
                    if obj:
                        triples.append(ParsedTriple(subj, obj, "is_a", 0.9, source_text))
                        return triples

                # Pattern 5: Property — "X is ADJ" (including size words)
                if len(rest) == 1:
                    triples.append(ParsedTriple(subj, rest[0], "property", 0.85, source_text))
                    return triples

                # Multi-word predicate after be-verb — treat as property
                if len(rest) <= 3:
                    obj = self._extract_np(rest)
                    if obj:
                        triples.append(ParsedTriple(subj, obj, "property", 0.7, source_text))
                        return triples

        # Pattern 6: Has — "X has Y"
        has_idx = None
        for i, w in enumerate(words):
            if w in ("has", "have", "had"):
                has_idx = i
                break

        if has_idx is not None and has_idx > 0 and has_idx < len(words) - 1:
            subj = self._extract_np(self._strip_determiners(words[:has_idx]))
            obj = self._extract_np(self._strip_determiners(words[has_idx + 1:]))
            if subj and obj:
                triples.append(ParsedTriple(subj, obj, "has", 0.85, source_text))
                return triples

        # Pattern 7 & 8: SVO / SV — "X VERB Y" or "X VERB"
        # Find first verb-like word (not a stop word, not first word if it looks like a noun)
        verb_idx = None
        for i, w in enumerate(words):
            if i == 0:
                continue  # skip subject position
            if w in _STOP_VERBS:
                continue
            if w in DETERMINERS:
                continue
            # Heuristic: treat second word as verb if clause has 2+ words after subject
            verb_idx = i
            break

        if verb_idx is not None:
            subj = self._extract_np(self._strip_determiners(words[:verb_idx]))
            verb = words[verb_idx]
            rest = words[verb_idx + 1:]

            if subj and verb:
                if rest:
                    obj = self._extract_np(self._strip_determiners(rest))
                    if obj:
                        # SVO
                        triples.append(ParsedTriple(subj, verb, "action", 0.8, source_text))
                        triples.append(ParsedTriple(verb, obj, "action_target", 0.8, source_text))
                        return triples
                # SV
                triples.append(ParsedTriple(subj, verb, "action", 0.75, source_text))
                return triples

        return triples

    # -- public API --------------------------------------------------------

    def parse_text(self, text: str) -> ParseResult:
        """Parse free text into triples via rule-based pattern matching."""
        raw = text.strip()
        if not raw:
            return ParseResult(raw_text=raw)

        clauses = self._split_clauses(raw.lower())
        all_triples: list[ParsedTriple] = []
        concepts_set: set[str] = set()

        for clause in clauses:
            triples = self._parse_clause(clause, raw)
            for t in triples:
                # Filter out garbage concepts
                if self._is_valid_concept(t.subject) and self._is_valid_concept(t.object):
                    all_triples.append(t)
                    concepts_set.add(t.subject)
                    concepts_set.add(t.object)

        return ParseResult(
            triples=all_triples,
            concepts=sorted(concepts_set),
            modality="text",
            item_type="general",
            raw_text=raw,
        )

    def _filter_result(self, result: ParseResult) -> ParseResult:
        """Remove triples with invalid concept names."""
        clean = [t for t in result.triples
                 if self._is_valid_concept(t.subject) and self._is_valid_concept(t.object)]
        concepts = sorted({t.subject for t in clean} | {t.object for t in clean})
        return ParseResult(triples=clean, concepts=concepts,
                           modality=result.modality, item_type=result.item_type,
                           raw_text=result.raw_text)

    def parse_item(self, item: dict) -> ParseResult:
        """Auto-detect dataset format from keys, dispatch to specialized parser."""
        keys = set(item.keys())

        # Math: question + answer + final_answer (gsm8k style)
        if {"question", "answer", "final_answer"} <= keys:
            return self._filter_result(self._parse_math(item))

        # Coding: instruction + output
        if {"instruction", "output"} <= keys:
            return self._filter_result(self._parse_coding(item))

        # Passage QA: passage + question (boolq style)
        if {"passage", "question"} <= keys:
            return self._filter_result(self._parse_passage_qa(item))

        # Multiple choice: question + choices
        if "question" in keys and "choices" in keys:
            return self._filter_result(self._parse_multiple_choice(item))

        # Pronoun resolution: sentence + option1
        if "sentence" in keys and "option1" in keys:
            return self._filter_result(self._parse_pronoun(item))

        # Plain text
        if "text" in keys and len(keys) == 1:
            return self.parse_text(item["text"])

        # Default: find a text-ish field and parse it
        for field_name in ("text", "sentence", "question", "input", "content"):
            if field_name in item and isinstance(item[field_name], str):
                result = self.parse_text(item[field_name])
                result.item_type = "general"
                return result

        return ParseResult(raw_text=str(item), item_type="unknown")

    def parse_dataset(
        self,
        items: list[dict],
        progress_callback=None,
    ) -> list[ParseResult]:
        """Parse a list of dataset items, with optional progress callback."""
        results: list[ParseResult] = []
        total = len(items)
        for i, item in enumerate(items):
            results.append(self.parse_item(item))
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, total)
        if progress_callback and total % 100 != 0:
            progress_callback(total, total)
        return results

    # -- specialized parsers -----------------------------------------------

    def _parse_math(self, item: dict) -> ParseResult:
        """Extract concepts from math/gsm8k items."""
        question = item.get("question", "")
        final_answer = str(item.get("final_answer", ""))
        category = item.get("category", "math")

        # Parse the question text for any extractable triples
        result = self.parse_text(question)
        result.item_type = "math"
        result.modality = "text"

        # Add math-specific triples
        result.triples.append(ParsedTriple(
            subject="problem",
            object=final_answer,
            relation="answer",
            confidence=0.95,
            source_text=question,
        ))
        result.triples.append(ParsedTriple(
            subject="problem",
            object=category,
            relation="category",
            confidence=1.0,
            source_text=question,
        ))
        if "problem" not in result.concepts:
            result.concepts.append("problem")
        if final_answer and final_answer not in result.concepts:
            result.concepts.append(final_answer)

        return result

    def _parse_coding(self, item: dict) -> ParseResult:
        """Extract concepts from coding instruction items."""
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        result = self.parse_text(instruction)
        result.item_type = "coding"
        result.modality = "text"

        # Extract task concept from instruction
        task_summary = self._normalize(
            "_".join(instruction.split()[:5])
        ) if instruction else "task"

        result.triples.append(ParsedTriple(
            subject="task",
            object=task_summary,
            relation="description",
            confidence=0.8,
            source_text=instruction,
        ))
        result.triples.append(ParsedTriple(
            subject="task",
            object="coding",
            relation="category",
            confidence=1.0,
            source_text=instruction,
        ))
        if "task" not in result.concepts:
            result.concepts.append("task")
        if "coding" not in result.concepts:
            result.concepts.append("coding")

        return result

    def _parse_passage_qa(self, item: dict) -> ParseResult:
        """Extract concepts from passage + question items (boolq style)."""
        passage = item.get("passage", "")
        question = item.get("question", "")
        answer = item.get("answer", "")

        # Parse the passage for content triples
        result = self.parse_text(passage)
        result.item_type = "passage_qa"

        # Extract topic from first few words of passage
        passage_words = passage.split()[:5]
        topic = self._extract_np([self._normalize(w) for w in passage_words])
        if not topic:
            topic = "topic"

        # Add QA-specific triples
        answer_str = str(answer).lower().strip()
        if answer_str in ("true", "yes", "1"):
            answer_str = "yes"
        elif answer_str in ("false", "no", "0"):
            answer_str = "no"

        result.triples.append(ParsedTriple(
            subject=topic,
            object=question.lower().strip().rstrip("?"),
            relation="questioned_about",
            confidence=0.85,
            source_text=question,
        ))
        result.triples.append(ParsedTriple(
            subject=question.lower().strip().rstrip("?"),
            object=answer_str,
            relation="answer",
            confidence=0.9,
            source_text=f"{question} -> {answer}",
        ))

        if topic not in result.concepts:
            result.concepts.append(topic)

        return result

    def _parse_multiple_choice(self, item: dict) -> ParseResult:
        """Extract concepts from multiple choice items."""
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer_idx = item.get("answer", item.get("label", 0))

        result = self.parse_text(question)
        result.item_type = "multiple_choice"

        # Resolve correct answer
        correct = ""
        if isinstance(answer_idx, int) and isinstance(choices, list) and 0 <= answer_idx < len(choices):
            correct = str(choices[answer_idx])
        elif isinstance(answer_idx, str):
            # Letter-based: "A"->0, "B"->1, etc.
            idx = ord(answer_idx.upper()) - ord("A")
            if isinstance(choices, list) and 0 <= idx < len(choices):
                correct = str(choices[idx])
            else:
                correct = answer_idx

        if correct:
            q_short = self._normalize("_".join(question.split()[:6]))
            result.triples.append(ParsedTriple(
                subject=q_short if q_short else "question",
                object=self._normalize(correct),
                relation="answer",
                confidence=0.9,
                source_text=question,
            ))
            norm_correct = self._normalize(correct)
            if norm_correct and norm_correct not in result.concepts:
                result.concepts.append(norm_correct)

        return result

    def _parse_pronoun(self, item: dict) -> ParseResult:
        """Extract from pronoun resolution items (winogrande style)."""
        sentence = item.get("sentence", "")
        option1 = item.get("option1", "")
        option2 = item.get("option2", "")
        answer = item.get("answer", "")

        # Replace underscore placeholder with correct answer
        correct = option1 if str(answer) == "1" else option2
        filled = sentence.replace("_", correct)

        result = self.parse_text(filled)
        result.item_type = "pronoun"
        return result
