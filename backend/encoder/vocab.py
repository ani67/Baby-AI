import json


# 2048 most common English words (abridged seed list — expanded at runtime
# from teacher answers). This list covers the core vocabulary needed for
# early-stage communication.
_BASE_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "are", "was", "were", "been", "being",
    "has", "had", "does", "did", "will", "shall", "should", "may", "might",
    "must", "need", "here", "very", "much", "more", "many", "such", "long",
    "great", "little", "own", "old", "right", "big", "high", "small",
    "large", "next", "early", "young", "important", "few", "public", "bad",
    "same", "last", "able", "thing", "point", "still", "call", "hand",
    "keep", "eye", "never", "let", "thought", "city", "tree", "cross",
    "farm", "hard", "start", "might", "story", "far", "sea", "draw",
    "left", "late", "run", "while", "press", "close", "night", "real",
    "life", "open", "seem", "together", "always", "children", "begin",
    "got", "walk", "example", "ease", "paper", "group", "often", "play",
    "feel", "find", "tell", "ask", "men", "change", "went", "light",
    "kind", "off", "put", "try", "head", "help", "line", "turn", "move",
    "live", "found", "learn", "should", "world", "house", "water", "name",
    "school", "every", "home", "read", "place", "round", "animal", "food",
    "earth", "eye", "face", "watch", "dog", "cat", "bird", "fish", "horse",
    "man", "woman", "child", "boy", "girl", "baby", "mother", "father",
    "family", "friend", "body", "door", "room", "book", "word", "number",
    "part", "sound", "show", "side", "different", "end", "answer",
    "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "color", "red", "blue", "green", "white", "black", "yellow", "brown",
    "big", "small", "hot", "cold", "fast", "slow", "happy", "sad",
    "yes", "no", "true", "false", "top", "bottom", "left", "right",
    "front", "stop", "both", "between", "each", "under", "again",
    "same", "different", "before", "above", "below", "near", "inside",
    "outside", "around", "through", "during", "without", "along",
    "toward", "among", "across", "behind", "since", "until", "enough",
    "eat", "drink", "sleep", "sit", "stand", "fall", "cut", "hold",
    "carry", "bring", "buy", "sell", "sing", "write", "speak", "hear",
    "grow", "build", "drive", "fly", "swim", "jump", "throw", "catch",
    "push", "pull", "pick", "drop", "break", "wear", "win", "lose",
    "send", "receive", "believe", "remember", "forget", "decide",
    "choose", "follow", "lead", "reach", "touch", "include", "consider",
    "appear", "love", "hate", "wait", "serve", "die", "spend", "fill",
    "raise", "pass", "agree", "happen", "plan", "produce", "cause",
    "stay", "allow", "contain", "belong", "exist", "matter", "represent",
    "develop", "support", "cover", "suggest", "create", "continue",
    "set", "describe", "accept", "join", "add", "compare", "explain",
    "offer", "provide", "measure", "expect", "share", "mean",
    "car", "truck", "bus", "train", "plane", "boat", "ship", "road",
    "street", "bridge", "ball", "toy", "game", "table", "chair", "bed",
    "window", "wall", "floor", "picture", "music", "song", "dance",
    "rain", "snow", "wind", "sun", "moon", "star", "sky", "cloud",
    "mountain", "river", "lake", "ocean", "forest", "garden", "flower",
    "grass", "rock", "sand", "fire", "air", "dark", "bright", "soft",
    "hard", "smooth", "rough", "heavy", "round", "flat", "sharp",
    "sweet", "sour", "bitter", "loud", "quiet", "warm", "cool",
    "wet", "dry", "clean", "dirty", "empty", "full", "deep", "wide",
    "thin", "thick", "straight", "strong", "weak", "safe", "dangerous",
    "simple", "easy", "difficult", "possible", "certain", "clear",
    "sure", "ready", "free", "special", "usual", "beautiful", "nice",
    "pretty", "ugly", "strange", "funny", "serious", "angry", "afraid",
    "tired", "hungry", "sick", "alive", "dead", "alone", "together",
    "apart", "close", "far", "early", "late", "fast", "slow",
    "already", "often", "sometimes", "never", "always", "usually",
    "perhaps", "probably", "really", "quite", "almost", "enough",
    "too", "very", "only", "just", "even", "still", "already",
    "clothes", "shirt", "pants", "shoes", "hat", "dress", "coat",
    "minute", "hour", "week", "month", "season", "spring", "summer",
    "autumn", "winter", "morning", "afternoon", "evening",
    "today", "tomorrow", "yesterday", "ago", "soon", "later",
    "color", "shape", "size", "type", "form", "class", "sort",
    "idea", "reason", "problem", "question", "fact", "case", "area",
    "system", "program", "rule", "law", "power", "force", "energy",
    "age", "period", "history", "science", "nature", "art", "language",
    "north", "south", "east", "west", "center", "edge", "surface",
    "piece", "bit", "pair", "half", "double", "single", "whole",
    "general", "common", "basic", "main", "major", "important",
    "local", "natural", "physical", "human", "social", "political",
    "similar", "likely", "possible", "able", "available", "necessary",
    "known", "best", "better", "worse", "worst", "least", "less",
    "more", "most", "own", "second", "third", "million", "hundred",
    "thousand", "percent", "half", "quarter",
    "plant", "leaf", "root", "seed", "fruit", "vegetable",
    "meat", "bread", "milk", "egg", "sugar", "salt", "oil",
    "teeth", "nose", "ear", "mouth", "arm", "leg", "foot", "hand",
    "finger", "heart", "brain", "bone", "blood", "skin", "hair",
    "land", "island", "field", "hill", "valley", "beach", "farm",
    "town", "village", "country", "map", "path", "step",
    "king", "queen", "president", "doctor", "teacher", "student",
    "worker", "soldier", "police", "judge", "captain", "chief",
    "brother", "sister", "son", "daughter", "uncle", "aunt",
    "husband", "wife", "neighbor", "stranger", "enemy", "god",
    "box", "bag", "bottle", "cup", "plate", "bowl", "knife",
    "stick", "wheel", "machine", "engine", "tool", "gun", "bomb",
    "key", "lock", "ring", "chain", "wire", "rope", "string",
    "cloth", "cotton", "wood", "stone", "metal", "gold", "silver",
    "glass", "plastic", "rubber", "leather", "paper", "ink", "paint",
    "mark", "spot", "hole", "crack", "gap", "space", "corner",
    "smell", "taste", "sight", "noise", "voice", "cry", "laugh",
    "smile", "sign", "signal", "symbol", "letter", "note", "message",
    "news", "report", "record", "list", "test", "score",
    "price", "cost", "value", "deal", "trade", "market", "store",
    "shop", "bank", "office", "company", "business", "industry",
    "war", "peace", "fight", "battle", "attack", "defense", "army",
    "church", "god", "spirit", "soul", "mind", "memory", "dream",
    "fear", "hope", "wish", "desire", "pleasure", "pain", "joy",
    "anger", "surprise", "interest", "attention", "effort", "trouble",
    "success", "failure", "mistake", "accident", "chance", "luck",
    "purpose", "goal", "result", "effect", "cause", "source",
    "base", "model", "pattern", "design", "style", "method",
    "process", "action", "activity", "event", "situation", "condition",
    "position", "direction", "distance", "speed", "rate", "degree",
    "level", "amount", "total", "average", "range", "limit",
    "increase", "decrease", "rise", "fall", "growth", "reduction",
    "relation", "connection", "link", "bond", "tie", "contact",
    "difference", "contrast", "balance", "match", "fit",
    "image", "view", "scene", "figure", "shape", "circle", "square",
    "cross", "line", "point", "dot", "wave", "curve", "angle",
    "length", "width", "height", "depth", "weight", "mass",
    "temperature", "pressure", "current", "flow", "heat",
    "is", "has", "are", "does", "goes", "makes", "takes",
    "comes", "gives", "says", "looks", "gets", "knows", "thinks",
    "wants", "sees", "uses", "finds", "tells", "asks", "works",
    "seems", "tries", "leaves", "calls", "needs", "becomes", "keeps",
    "lets", "begins", "shows", "hears", "plays", "runs", "moves",
    "lives", "believes", "brings", "happens", "writes", "sits",
    "stands", "loses", "pays", "meets", "includes", "continues",
    "sets", "learns", "changes", "leads", "understands", "follows",
    "creates", "speaks", "reads", "allows", "adds", "grows",
    "opens", "walks", "wins", "offers", "remembers", "loves",
    "considers", "appears", "buys", "waits", "serves", "dies",
    "sends", "expects", "builds", "stays", "falls", "cuts",
    "reaches", "kills", "raises", "passes", "sells", "requires",
    "decides", "returns", "explains", "hopes", "develops", "carries",
    "breaks", "receives", "agrees", "supports", "holds", "produces",
    "eats", "covers", "catches", "draws", "chooses",
]


class Vocabulary:
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<START>": 1,
        "<END>": 2,
        "<UNK>": 3,
    }

    def __init__(self, max_size: int = 8192):
        self.max_size = max_size
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self.word_counts: dict[str, int] = {}
        self._load_base_vocab()

    def _load_base_vocab(self) -> None:
        # Start with special tokens
        for token, idx in self.SPECIAL_TOKENS.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        # Add base words, deduplicating
        next_id = len(self.SPECIAL_TOKENS)
        # Reserve slots for dynamic growth via add_word
        base_limit = max(self.max_size - max(self.max_size // 10, 4), len(self.SPECIAL_TOKENS))
        seen = set()
        for word in _BASE_WORDS:
            w = word.lower()
            if w in seen or w in self.word_to_id:
                continue
            if next_id >= base_limit:
                break
            seen.add(w)
            self.word_to_id[w] = next_id
            self.id_to_word[next_id] = w
            next_id += 1

    def add_word(self, word: str) -> None:
        """
        Increments count for word. If count >= 3 and word not in vocab
        and vocab not full: adds to vocab.
        """
        w = word.lower()
        self.word_counts[w] = self.word_counts.get(w, 0) + 1
        if (
            self.word_counts[w] >= 3
            and w not in self.word_to_id
            and len(self.word_to_id) < self.max_size
        ):
            next_id = len(self.word_to_id)
            self.word_to_id[w] = next_id
            self.id_to_word[next_id] = w

    def encode(self, text: str) -> list[int]:
        """Tokenizes text to list of ids. Unknown words -> <UNK>."""
        unk_id = self.SPECIAL_TOKENS["<UNK>"]
        tokens = text.strip().split()
        result = []
        for t in tokens:
            # Check exact match first (for special tokens like <START>)
            if t in self.word_to_id:
                result.append(self.word_to_id[t])
            else:
                result.append(self.word_to_id.get(t.lower(), unk_id))
        return result

    def decode(self, ids: list[int]) -> str:
        """Converts ids back to text string."""
        unk_token = "<UNK>"
        words = [self.id_to_word.get(i, unk_token) for i in ids]
        return " ".join(words)

    def save(self, path: str) -> None:
        """Serializes to JSON — included in model checkpoints."""
        data = {
            "max_size": self.max_size,
            "word_to_id": self.word_to_id,
            "word_counts": self.word_counts,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Restores from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.max_size = data["max_size"]
        self.word_to_id = data["word_to_id"]
        self.id_to_word = {int(v): k for k, v in self.word_to_id.items()}
        self.word_counts = data.get("word_counts", {})
