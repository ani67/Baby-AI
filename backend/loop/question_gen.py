import random


class QuestionGenerator:
    """
    Generates a question string from a curriculum item.
    """

    IMAGE_TEMPLATES = [
        "What is this?",
        "What do you see?",
        "Describe this image.",
        "What do you call this?",
        "Name what you see.",
    ]

    TEMPLATES = {
        0: [
            "What is this? [IMAGE: {description}]",
            "What do you call this? [IMAGE: {description}]",
            "Name this: [IMAGE: {description}]",
        ],
        1: [
            "What kind of thing is a {label}?",
            "What category does a {label} belong to?",
            "Is a {label} an animal, a plant, or an object?",
        ],
        2: [
            "What do {label_a} and {label_b} have in common?",
            "How is a {label_a} different from a {label_b}?",
            "Why would you group {label_a} with {label_b}?",
        ],
        3: [
            "What caused {event_b} to happen after {event_a}?",
            "Why did {event_b} follow from {event_a}?",
            "What is the relationship between {event_a} and {event_b}?",
        ],
        4: [
            "What does {concept} mean?",
            "How would you explain {concept} to someone who had never heard of it?",
            "What is {concept} an example of?",
        ],
    }

    def generate(
        self,
        item,
        stage: int,
        recent_questions: list[str],
    ) -> str:
        # For image items with actual files, use simple image questions
        if item.item_type == "image" and item.image_path:
            for attempt in range(10):
                question = random.choice(self.IMAGE_TEMPLATES)
                if not self._too_similar(question, recent_questions):
                    return question
            return random.choice(self.IMAGE_TEMPLATES)

        templates = self.TEMPLATES.get(stage, self.TEMPLATES[4])
        for attempt in range(10):
            template = random.choice(templates)
            try:
                question = template.format(**item.template_slots)
            except KeyError:
                question = f"Tell me about {item.label or 'this'}."
            if not self._too_similar(question, recent_questions):
                return question
        return f"Tell me about {item.label or 'this'}."

    def _too_similar(
        self,
        question: str,
        recent: list[str],
        threshold: float = 0.8,
    ) -> bool:
        words = set(question.lower().split())
        for prev in recent[-10:]:
            prev_words = set(prev.lower().split())
            if not words or not prev_words:
                continue
            overlap = len(words & prev_words) / len(words | prev_words)
            if overlap > threshold:
                return True
        return False
