"""
Seed the training data with images + text concepts.
Run: python3 seed_data.py

Downloads ~420 images across 60+ categories from loremflickr.com,
and adds 300+ text concepts to concepts.txt.
"""

import os
import time
import urllib.request

DATA_DIR = "data/stage0"
IMAGES_PER_CATEGORY = 7

# ── Image categories (60+) with deliberate semantic diversity ──
IMAGE_CATEGORIES = {
    # Animals — Pets
    "dog": "dog",
    "cat": "cat",
    "hamster": "hamster",
    "rabbit": "rabbit",
    "goldfish": "goldfish",
    # Animals — Farm
    "horse": "horse",
    "cow": "cow",
    "pig": "pig",
    "sheep": "sheep",
    "chicken": "chicken",
    "goat": "goat",
    # Animals — Wild
    "lion": "lion",
    "elephant": "elephant",
    "tiger": "tiger",
    "bear": "bear",
    "wolf": "wolf",
    "deer": "deer",
    "monkey": "monkey",
    "giraffe": "giraffe",
    "zebra": "zebra",
    # Animals — Ocean
    "dolphin": "dolphin",
    "whale": "whale",
    "shark": "shark",
    "jellyfish": "jellyfish",
    "seahorse": "seahorse",
    # Animals — Insects
    "butterfly": "butterfly",
    "bee": "bee",
    "ladybug": "ladybug",
    "dragonfly": "dragonfly",
    # Animals — Birds
    "eagle": "eagle",
    "parrot": "parrot",
    "owl": "owl",
    "penguin": "penguin",
    "flamingo": "flamingo",
    # Vehicles
    "car": "car",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "train": "train",
    "airplane": "airplane",
    "boat": "boat",
    "helicopter": "helicopter",
    # Food — Fruits
    "apple": "apple",
    "banana": "banana",
    "strawberry": "strawberry",
    "orange fruit": "orange+fruit",
    "grapes": "grapes",
    # Food — Vegetables
    "broccoli": "broccoli",
    "carrot": "carrot",
    "tomato": "tomato",
    "corn": "corn",
    # Food — Meals & Bakery
    "pizza": "pizza",
    "burger": "burger",
    "bread": "bread",
    "cake": "cake",
    "sushi": "sushi",
    # Food — Drinks
    "coffee": "coffee+cup",
    "juice": "orange+juice",
    # Nature
    "mountain": "mountain+landscape",
    "forest": "forest",
    "beach": "beach+ocean",
    "river": "river+nature",
    "desert": "desert+sand",
    "sky": "sky+clouds",
    "waterfall": "waterfall",
    "volcano": "volcano",
    # Objects — Furniture
    "chair": "chair+furniture",
    "table": "table+furniture",
    "bed": "bed+bedroom",
    "sofa": "sofa+couch",
    # Objects — Tools
    "hammer": "hammer+tool",
    "scissors": "scissors",
    "wrench": "wrench+tool",
    # Objects — Clothing
    "shoe": "shoe+footwear",
    "hat": "hat+fashion",
    "dress": "dress+clothing",
    # Objects — Electronics
    "laptop": "laptop+computer",
    "phone": "smartphone",
    "camera": "camera+photography",
    "television": "television+screen",
    # Objects — Instruments
    "guitar": "guitar+music",
    "piano": "piano+keyboard",
    "violin": "violin+instrument",
    "drums": "drums+music",
    # People & Scenes
    "sports": "sports+athlete",
    "city street": "city+street",
    "playground": "playground+children",
    "classroom": "classroom+school",
    "garden": "garden+flowers",
    # Structures
    "house": "house+home",
    "bridge": "bridge+architecture",
    "castle": "castle+medieval",
    "lighthouse": "lighthouse+coast",
}

# ── Text concepts (300+) ──
TEXT_CONCEPTS = [
    # Colors (15)
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "brown", "gray", "gold", "silver", "turquoise", "magenta",
    # Shapes (10)
    "circle", "square", "triangle", "star", "heart",
    "diamond", "oval", "rectangle", "spiral", "cube",
    # Emotions (25)
    "happy", "sad", "angry", "scared", "surprised", "sleepy",
    "curious", "excited", "bored", "nervous", "proud", "shy",
    "jealous", "grateful", "lonely", "hopeful", "confused",
    "disgusted", "embarrassed", "amazed", "calm", "anxious",
    "cheerful", "grumpy", "content",
    # Opposites (40 — 20 pairs)
    "hot", "cold", "fast", "slow", "big", "small",
    "old", "new", "light", "dark", "loud", "quiet",
    "hard", "soft", "wet", "dry", "thick", "thin",
    "heavy", "light", "full", "empty", "open", "closed",
    "near", "far", "wide", "narrow", "deep", "shallow",
    "sharp", "dull", "smooth", "rough", "clean", "dirty",
    "strong", "weak", "rich", "poor",
    # Actions (40)
    "running", "jumping", "eating", "sleeping", "swimming", "flying",
    "walking", "sitting", "standing", "climbing", "falling", "throwing",
    "catching", "pushing", "pulling", "lifting", "carrying", "building",
    "breaking", "growing", "shrinking", "spinning", "dancing", "singing",
    "laughing", "crying", "whispering", "shouting", "reading", "writing",
    "drawing", "painting", "cooking", "digging", "planting", "watering",
    "hiding", "seeking", "hugging", "waving",
    # Abstract concepts (30)
    "time", "space", "justice", "beauty", "truth", "change",
    "freedom", "peace", "love", "hate", "fear", "hope",
    "memory", "dream", "silence", "chaos", "order", "balance",
    "gravity", "energy", "power", "knowledge", "wisdom", "imagination",
    "infinity", "nothing", "everything", "beginning", "ending", "pattern",
    # Sensory words (30)
    "rough", "smooth", "loud", "quiet", "bright", "bitter",
    "sweet", "sour", "salty", "spicy", "warm", "cool",
    "soft", "sharp", "fuzzy", "sticky", "slippery", "crunchy",
    "fragrant", "stinky", "glowing", "shimmering", "vibrating", "tingling",
    "echoing", "crackling", "bubbling", "creamy", "gritty", "silky",
    # Body parts (15)
    "hand", "foot", "eye", "nose", "mouth", "ear", "head",
    "arm", "leg", "finger", "knee", "shoulder", "back", "belly", "neck",
    # Nature words (20)
    "sun", "moon", "cloud", "rain", "snow", "wind",
    "mountain", "river", "ocean", "forest", "storm", "thunder",
    "lightning", "rainbow", "sunrise", "sunset", "fog", "ice",
    "fire", "earthquake",
    # Food words (15)
    "milk", "bread", "egg", "cheese", "water", "juice",
    "cookie", "cake", "rice", "pasta", "soup", "chocolate",
    "honey", "butter", "salt",
    # Family & social (12)
    "mama", "papa", "baby", "sister", "brother", "friend",
    "teacher", "doctor", "neighbor", "stranger", "crowd", "team",
    # Common objects (20)
    "book", "chair", "table", "door", "window", "bed",
    "cup", "spoon", "shoe", "hat", "phone", "clock",
    "key", "lamp", "mirror", "toy", "block", "puzzle",
    "umbrella", "candle",
    # Spatial & directional (15)
    "up", "down", "left", "right", "inside", "outside",
    "above", "below", "behind", "between", "through", "around",
    "across", "along", "toward",
    # Time words (12)
    "morning", "afternoon", "evening", "night", "yesterday", "tomorrow",
    "always", "never", "sometimes", "soon", "later", "now",
    # Weather (8)
    "sunny", "rainy", "cloudy", "windy", "snowy", "foggy",
    "stormy", "humid",
    # Materials (10)
    "wood", "metal", "glass", "plastic", "stone", "paper",
    "fabric", "rubber", "leather", "clay",
    # Musical terms (8)
    "rhythm", "melody", "harmony", "beat", "tempo", "chord",
    "note", "song",
]


def download_images():
    """Download images using Lorem Flickr (free, no API key needed)."""
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    categories = list(IMAGE_CATEGORIES.items())
    print(f"  {len(categories)} categories, {IMAGES_PER_CATEGORY} images each\n")

    for cat_idx, (category, search_term) in enumerate(categories, 1):
        cat_dir = os.path.join(DATA_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)

        existing = [f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.png'))]
        if len(existing) >= IMAGES_PER_CATEGORY:
            print(f"  [{cat_idx}/{len(categories)}] {category}: already has {len(existing)} images, skipping")
            total_skipped += len(existing)
            continue

        downloaded = 0
        # Use different sizes to get different images from loremflickr
        sizes = [(320, 240), (400, 300), (300, 400), (500, 350), (350, 500), (450, 300), (300, 450)]

        for i in range(IMAGES_PER_CATEGORY):
            img_name = f"{category.replace(' ', '_')}_{i}.jpg"
            img_path = os.path.join(cat_dir, img_name)

            if os.path.exists(img_path):
                downloaded += 1
                continue

            w, h = sizes[i % len(sizes)]
            url = f"https://loremflickr.com/{w}/{h}/{search_term}"

            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                })
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                    if len(data) > 1000:
                        with open(img_path, "wb") as f:
                            f.write(data)
                        downloaded += 1
                    else:
                        total_failed += 1
            except Exception as e:
                total_failed += 1
                print(f"    SKIP {category}/{img_name}: {e}")

            time.sleep(0.3)

        total_downloaded += downloaded
        print(f"  [{cat_idx}/{len(categories)}] {category}: {downloaded}/{IMAGES_PER_CATEGORY} images")

    return total_downloaded, total_skipped, total_failed


def write_concepts():
    """Write text concepts to concepts.txt, preserving any existing ones."""
    concepts_path = os.path.join(DATA_DIR, "concepts.txt")

    # Load existing concepts to avoid duplicates
    existing = set()
    if os.path.exists(concepts_path):
        with open(concepts_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    existing.add(stripped.lower())

    # Deduplicate new concepts (some words appear in multiple categories)
    seen = set()
    unique_concepts = []
    for concept in TEXT_CONCEPTS:
        key = concept.lower()
        if key not in seen:
            seen.add(key)
            unique_concepts.append(concept)

    # Merge: existing + new
    new_count = 0
    all_concepts = list(existing)
    for concept in unique_concepts:
        if concept.lower() not in existing:
            all_concepts.append(concept)
            new_count += 1

    with open(concepts_path, "w") as f:
        for concept in all_concepts:
            f.write(concept + "\n")

    return len(all_concepts), new_count


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=== Downloading images ===")
    img_count, img_skipped, img_failed = download_images()
    print(f"\n  Downloaded: {img_count}")
    print(f"  Skipped (already exist): {img_skipped}")
    print(f"  Failed: {img_failed}")

    print("\n=== Writing text concepts ===")
    total_concepts, new_concepts = write_concepts()
    print(f"  Total concepts: {total_concepts} ({new_concepts} new)")

    print(f"\n=== Summary ===")
    print(f"  Image categories: {len(IMAGE_CATEGORIES)}")
    print(f"  Images on disk:   {img_count + img_skipped}")
    print(f"  Text concepts:    {total_concepts}")
    print(f"  Total items:      {img_count + img_skipped + total_concepts}")
    print(f"\nRestart the backend to pick up new training data.")


if __name__ == "__main__":
    main()
