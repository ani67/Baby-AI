"""
Generate diverse training data for Baby AI.

Produces ~25K simple sentences, definitions, facts, and QA pairs
across 30+ categories. Output format matches TextCurriculum expectations.

Usage:
    cd backend && python -m scripts.download_data
"""

import json
import random
import itertools
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "text_diverse.json"

# ---------------------------------------------------------------------------
# Vocabulary banks
# ---------------------------------------------------------------------------

ANIMALS = [
    "dog", "cat", "bird", "fish", "cow", "horse", "pig", "sheep", "duck",
    "rabbit", "frog", "bear", "lion", "tiger", "elephant", "monkey", "snake",
    "whale", "dolphin", "eagle", "owl", "ant", "bee", "spider", "turtle",
    "penguin", "chicken", "goat", "deer", "wolf", "fox", "mouse", "rat",
    "bat", "shark", "crab", "octopus", "butterfly", "parrot", "crow",
    "zebra", "giraffe", "hippo", "rhino", "kangaroo", "koala", "panda",
    "seal", "otter", "hawk", "swan", "goose", "camel", "donkey", "squirrel",
]

COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "gold", "silver",
]

FOODS = [
    "apple", "banana", "bread", "rice", "milk", "egg", "cheese", "cake",
    "pizza", "pasta", "soup", "salad", "chicken", "fish", "meat", "corn",
    "potato", "carrot", "tomato", "orange", "grape", "strawberry", "cookie",
    "pie", "sandwich", "ice cream", "yogurt", "butter", "honey", "jam",
    "cereal", "noodle", "bean", "pea", "broccoli", "spinach", "onion",
    "pepper", "mushroom", "watermelon", "pear", "peach", "mango", "lemon",
    "coconut", "peanut", "walnut", "chocolate", "candy", "popcorn",
]

OBJECTS = [
    "ball", "book", "chair", "table", "cup", "box", "bag", "pen", "key",
    "lamp", "clock", "phone", "door", "window", "bed", "shoe", "hat",
    "car", "bus", "bike", "boat", "train", "plane", "truck", "wheel",
    "bell", "flag", "map", "coin", "ring", "rope", "net", "fan", "drum",
    "brush", "comb", "soap", "towel", "plate", "fork", "spoon", "knife",
    "bottle", "jar", "basket", "ladder", "hammer", "nail", "candle", "mirror",
    "pillow", "blanket", "umbrella", "kite", "balloon", "camera", "guitar",
]

BODY_PARTS = [
    "head", "eye", "ear", "nose", "mouth", "hand", "arm", "leg", "foot",
    "finger", "toe", "hair", "teeth", "tongue", "knee", "elbow", "neck",
    "back", "chest", "stomach", "shoulder", "chin", "lip", "cheek", "thumb",
]

PLACES = [
    "house", "school", "park", "store", "farm", "forest", "beach", "lake",
    "river", "mountain", "city", "town", "garden", "zoo", "library",
    "hospital", "kitchen", "bedroom", "bathroom", "yard", "road", "bridge",
    "island", "desert", "cave", "castle", "church", "market", "airport",
    "station", "museum", "playground", "pool", "field", "hill", "valley",
]

COUNTRIES = [
    "China", "India", "Brazil", "Japan", "France", "Egypt", "Canada",
    "Australia", "Mexico", "Russia", "Italy", "Germany", "Spain", "Kenya",
    "Peru", "Thailand", "Turkey", "Greece", "Sweden", "Norway",
    "Argentina", "Colombia", "Nigeria", "South Korea", "Indonesia",
]

WEATHER = [
    "sunny", "rainy", "cloudy", "snowy", "windy", "foggy", "stormy",
    "hot", "cold", "warm", "cool", "dry", "wet", "icy", "humid",
]

EMOTIONS = [
    "happy", "sad", "angry", "scared", "tired", "hungry", "thirsty",
    "excited", "bored", "surprised", "proud", "shy", "brave", "calm",
    "worried", "confused", "lonely", "grateful", "curious", "silly",
]

ACTIONS = [
    "run", "walk", "jump", "swim", "fly", "climb", "sing", "dance",
    "read", "write", "draw", "paint", "cook", "eat", "drink", "sleep",
    "play", "build", "dig", "push", "pull", "throw", "catch", "kick",
    "sit", "stand", "laugh", "cry", "shout", "whisper", "hide", "seek",
    "wash", "clean", "cut", "open", "close", "carry", "drop", "lift",
    "count", "clap", "wave", "point", "hug", "share", "help", "wait",
]

NAMES = [
    "Tom", "Sam", "Ben", "Mia", "Zoe", "Amy", "Leo", "Max", "Ava", "Eli",
    "Ivy", "Raj", "Kim", "Dan", "Eve", "Jay", "Kai", "Liv", "Nia", "Owen",
]

ADJECTIVES = [
    "big", "small", "tall", "short", "fast", "slow", "hot", "cold",
    "old", "new", "soft", "hard", "loud", "quiet", "clean", "dirty",
    "heavy", "light", "thick", "thin", "long", "round", "flat", "sharp",
    "sweet", "sour", "bright", "dark", "wet", "dry", "smooth", "rough",
    "strong", "weak", "deep", "wide", "narrow", "full", "empty", "rich",
]

MATERIALS = [
    "wood", "metal", "glass", "stone", "paper", "cloth", "plastic",
    "rubber", "leather", "cotton", "silk", "wool", "clay", "sand", "ice",
]

SEASONS = ["spring", "summer", "fall", "winter"]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

SHAPES = [
    "circle", "square", "triangle", "rectangle", "star", "heart",
    "diamond", "oval", "cube", "sphere", "cone", "cylinder",
]

INSTRUMENTS = [
    "piano", "guitar", "drum", "violin", "flute", "trumpet", "harp",
    "bell", "whistle",
]

JOBS = [
    "teacher", "doctor", "farmer", "pilot", "cook", "nurse", "driver",
    "singer", "painter", "builder", "baker", "firefighter", "police officer",
    "dentist", "vet", "librarian", "scientist", "astronaut", "judge",
]

PLANTS = [
    "tree", "flower", "grass", "bush", "vine", "fern", "cactus", "moss",
    "rose", "tulip", "daisy", "lily", "sunflower", "oak", "pine", "maple",
    "palm", "bamboo", "wheat", "corn", "rice",
]

PLANETS = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus",
    "Neptune",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

entries: list[dict] = []


def add(text: str, typ: str, level: int, category: str):
    entries.append({
        "text": text.lower().strip(),
        "type": typ,
        "level": level,
        "category": category,
    })


def pick(lst, n=1):
    if n == 1:
        return random.choice(lst)
    return random.sample(lst, min(n, len(lst)))


def pick2_diff(lst):
    a, b = random.sample(lst, 2)
    return a, b


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gen_animal_definitions():
    """Simple 'A ___ is ...' definitions for animals."""
    traits = {
        "dog": ["a pet that barks", "a loyal animal", "a furry pet with a tail"],
        "cat": ["a pet that meows", "a small furry animal", "a pet that purrs"],
        "bird": ["an animal that can fly", "an animal with feathers", "an animal with wings"],
        "fish": ["an animal that lives in water", "an animal that can swim", "an animal with fins"],
        "cow": ["a farm animal that gives milk", "a large animal that eats grass"],
        "horse": ["a large animal that people ride", "a fast animal with four legs"],
        "pig": ["a pink farm animal", "an animal that rolls in mud"],
        "sheep": ["an animal with wool", "a farm animal that says baa"],
        "duck": ["a bird that swims", "a bird that says quack"],
        "rabbit": ["a small animal with long ears", "a furry animal that hops"],
        "frog": ["a green animal that jumps", "an animal that lives near water and says ribbit"],
        "bear": ["a big strong animal", "a large animal that lives in the forest"],
        "lion": ["a big cat with a mane", "the king of the jungle"],
        "tiger": ["a big cat with stripes", "an orange cat with black stripes"],
        "elephant": ["the largest land animal", "a big gray animal with a trunk"],
        "monkey": ["an animal that climbs trees", "a smart animal with hands"],
        "snake": ["a long animal with no legs", "a reptile that slithers"],
        "whale": ["the largest animal in the sea", "a huge animal that swims"],
        "dolphin": ["a smart animal that lives in the ocean", "a friendly sea animal"],
        "eagle": ["a large bird that hunts", "a bird with sharp eyes"],
        "owl": ["a bird that comes out at night", "a wise bird with big eyes"],
        "ant": ["a tiny insect that is very strong", "a small bug that lives in colonies"],
        "bee": ["an insect that makes honey", "a flying bug that can sting"],
        "spider": ["a small creature with eight legs", "a bug that makes webs"],
        "turtle": ["a slow animal with a shell", "a reptile that carries its home"],
        "penguin": ["a bird that cannot fly but can swim", "a black and white bird that lives in the cold"],
        "chicken": ["a farm bird that lays eggs", "a bird that says cluck"],
        "goat": ["a farm animal that eats almost anything", "an animal with horns that climbs"],
        "deer": ["a fast animal with antlers", "a gentle animal that lives in the forest"],
        "wolf": ["a wild animal that howls", "a wild cousin of the dog"],
        "fox": ["a clever wild animal", "a red animal with a bushy tail"],
        "shark": ["a big fish with sharp teeth", "a fast hunter in the ocean"],
        "butterfly": ["an insect with colorful wings", "a pretty bug that was once a caterpillar"],
        "zebra": ["a horse-like animal with black and white stripes"],
        "giraffe": ["the tallest animal", "an animal with a very long neck"],
        "hippo": ["a big animal that lives in rivers", "a heavy animal that loves water"],
        "kangaroo": ["an animal that hops and has a pouch", "an Australian animal that jumps"],
        "koala": ["a cute animal that eats leaves", "an Australian animal that sleeps a lot"],
        "panda": ["a black and white bear that eats bamboo"],
        "seal": ["a sea animal with flippers", "an animal that swims and barks"],
        "otter": ["a playful animal that swims", "a furry animal that floats on its back"],
        "camel": ["an animal with a hump", "a desert animal that can go without water"],
        "squirrel": ["a small animal that collects nuts", "a furry animal with a bushy tail"],
        "octopus": ["a sea animal with eight arms", "a smart ocean creature"],
        "crab": ["a sea animal with claws", "an animal that walks sideways"],
        "parrot": ["a colorful bird that can talk", "a tropical bird with bright feathers"],
        "swan": ["a beautiful white bird", "a bird with a long curved neck"],
        "mouse": ["a tiny animal with a long tail", "a small rodent"],
        "bat": ["an animal that flies at night", "the only mammal that can fly"],
    }
    for animal, descs in traits.items():
        for desc in descs:
            add(f"a {animal} is {desc}", "definition", 1, "animals")


def gen_object_definitions():
    traits = {
        "ball": ["round and you can throw it", "used for playing games"],
        "book": ["full of pages with words", "something you read"],
        "chair": ["something you sit on", "a piece of furniture with legs"],
        "table": ["a flat surface with legs", "where you put things or eat"],
        "cup": ["something you drink from", "a small container for liquids"],
        "pen": ["a tool for writing", "used to write on paper"],
        "key": ["used to open a lock", "a small metal tool for doors"],
        "lamp": ["gives you light", "you turn it on when it is dark"],
        "clock": ["tells you the time", "has numbers and hands that move"],
        "phone": ["used to call people", "a device for talking to others"],
        "door": ["you open it to go in or out", "blocks the way into a room"],
        "window": ["lets light into a room", "made of glass in a wall"],
        "bed": ["where you sleep at night", "soft and comfortable for resting"],
        "shoe": ["you wear it on your foot", "protects your feet when you walk"],
        "hat": ["you wear it on your head", "keeps the sun off your face"],
        "car": ["a vehicle with four wheels", "takes you places on roads"],
        "boat": ["floats on water", "used to travel across water"],
        "train": ["travels on tracks", "a long vehicle that carries many people"],
        "plane": ["flies in the sky", "takes you to far away places fast"],
        "umbrella": ["keeps you dry in the rain", "you hold it over your head"],
        "mirror": ["shows your reflection", "made of glass so you can see yourself"],
        "candle": ["gives light when you burn it", "made of wax with a flame"],
        "bottle": ["holds water or other drinks", "a container with a narrow top"],
        "basket": ["used to carry things", "made of woven material"],
        "ladder": ["helps you climb up high", "has steps you can stand on"],
        "hammer": ["a tool for hitting nails", "used to build things"],
        "brush": ["used to paint or clean", "has bristles on one end"],
        "soap": ["used to clean your hands", "makes bubbles with water"],
        "pillow": ["soft and you rest your head on it", "used for sleeping"],
        "blanket": ["keeps you warm in bed", "a large soft cloth"],
        "kite": ["flies in the wind", "you hold the string while it flies"],
        "balloon": ["filled with air and floats", "round and colorful at parties"],
        "camera": ["takes pictures", "used to capture photos"],
        "guitar": ["a musical instrument with strings", "you strum it to make music"],
        "drum": ["you hit it to make sound", "a musical instrument you beat"],
    }
    for obj, descs in traits.items():
        for desc in descs:
            add(f"a {obj} is {desc}", "definition", 1, "objects")


def gen_body_facts():
    facts = [
        "you have two eyes to see",
        "you have two ears to hear",
        "you use your nose to smell",
        "you use your mouth to eat and talk",
        "you have ten fingers on your hands",
        "you have ten toes on your feet",
        "your heart pumps blood",
        "your brain helps you think",
        "your lungs help you breathe",
        "your bones hold you up",
        "your skin covers your body",
        "your teeth help you chew food",
        "your tongue helps you taste",
        "your legs help you walk and run",
        "your arms help you reach and carry",
        "you blink your eyes many times a day",
        "your hair grows on your head",
        "your stomach helps digest food",
        "you need sleep to rest your body",
        "your muscles help you move",
        "you have two hands and two feet",
        "your knee bends your leg",
        "your elbow bends your arm",
        "your neck holds up your head",
        "your back helps you stand straight",
    ]
    for f in facts:
        add(f, "fact", 1, "body")


def gen_science_facts():
    facts = [
        "the sun is a star",
        "the earth goes around the sun",
        "the moon goes around the earth",
        "water freezes into ice when it is very cold",
        "ice melts into water when it gets warm",
        "steam comes from very hot water",
        "plants need water and sunlight to grow",
        "rain falls from clouds",
        "snow is frozen water that falls from the sky",
        "a rainbow has seven colors",
        "the sky looks blue during the day",
        "stars come out at night",
        "the sun rises in the east",
        "the sun sets in the west",
        "gravity pulls things down to the ground",
        "light travels very fast",
        "sound travels through the air",
        "magnets can pull metal objects",
        "fire is hot and gives light",
        "air is all around us but we cannot see it",
        "seeds grow into plants",
        "trees make oxygen for us to breathe",
        "the earth spins once every day",
        "a year has twelve months",
        "there are eight planets in our solar system",
        "the ocean is full of salt water",
        "rivers flow to the sea",
        "mountains are very tall land",
        "volcanoes can erupt with hot lava",
        "dinosaurs lived millions of years ago",
        "fossils are very old bones in rocks",
        "the earth has land and water",
        "clouds are made of tiny water drops",
        "lightning is a flash of electricity in the sky",
        "thunder is the sound that lightning makes",
        "a shadow forms when something blocks light",
        "metal conducts electricity",
        "wood floats on water",
        "rocks sink in water",
        "oil floats on water",
        "hot air rises up",
        "cold air sinks down",
        "the desert is very dry and hot",
        "the north pole is very cold",
        "the equator is the middle of the earth",
        "earthquakes shake the ground",
        "wind is moving air",
        "a tornado is a spinning wind",
        "a hurricane is a very big storm",
        "coral reefs are in warm oceans",
    ]
    for f in facts:
        add(f, "fact", 2, "science")


def gen_geography_facts():
    country_facts = {
        "China": ["has the most people in the world", "has the Great Wall", "is in Asia"],
        "India": ["has many languages", "is in Asia", "has the Taj Mahal"],
        "Brazil": ["is in South America", "has the Amazon rainforest", "has big soccer teams"],
        "Japan": ["is a group of islands", "is in Asia", "has Mount Fuji"],
        "France": ["has the Eiffel Tower", "is in Europe", "is known for good food"],
        "Egypt": ["has the pyramids", "is in Africa", "has the Nile river"],
        "Canada": ["is very large", "is north of the United States", "has cold winters"],
        "Australia": ["is both a country and a continent", "has kangaroos", "is in the southern hemisphere"],
        "Mexico": ["is south of the United States", "has spicy food", "has many pyramids"],
        "Italy": ["has the shape of a boot", "is known for pizza and pasta", "has Rome"],
        "Germany": ["is in Europe", "makes many cars", "has cold winters"],
        "Spain": ["is in Europe", "has warm weather", "is known for flamenco"],
        "Kenya": ["is in Africa", "has many wild animals", "has big grasslands"],
        "Peru": ["is in South America", "has the Andes mountains", "has Machu Picchu"],
        "Thailand": ["is in Southeast Asia", "has beautiful temples", "has tropical weather"],
        "Greece": ["has many islands", "is very old", "had the first Olympics"],
        "Sweden": ["is in northern Europe", "has long winters", "has beautiful forests"],
        "Norway": ["has fjords", "is in northern Europe", "can see the northern lights"],
        "Argentina": ["is in South America", "has big grasslands", "loves soccer"],
        "South Korea": ["is in Asia", "has advanced technology", "has delicious food"],
    }
    for country, facts in country_facts.items():
        for fact in facts:
            add(f"{country} {fact}", "fact", 2, "geography")

    continent_facts = [
        "there are seven continents on earth",
        "Africa is a very large continent",
        "Asia is the biggest continent",
        "Europe has many countries close together",
        "North America has the United States Canada and Mexico",
        "South America has the Amazon river",
        "Antarctica is the coldest continent",
        "Australia is the smallest continent",
        "the Pacific Ocean is the biggest ocean",
        "the Atlantic Ocean is between America and Europe",
    ]
    for f in continent_facts:
        add(f, "fact", 2, "geography")


def gen_weather_sentences():
    templates = [
        "today it is {w}",
        "it is {w} outside",
        "the weather is {w} today",
        "when it is {w} you should {advice}",
    ]
    advice_map = {
        "sunny": "wear a hat",
        "rainy": "take an umbrella",
        "cloudy": "bring a jacket",
        "snowy": "wear a warm coat",
        "windy": "hold your hat",
        "foggy": "be careful on the road",
        "stormy": "stay inside",
        "hot": "drink lots of water",
        "cold": "wear warm clothes",
        "warm": "go outside and play",
    }
    for w in WEATHER:
        for t in templates[:3]:
            add(t.format(w=w), "simple_sentence", 1, "weather")
        if w in advice_map:
            add(templates[3].format(w=w, advice=advice_map[w]), "fact", 2, "weather")


def gen_color_sentences():
    for _ in range(800):
        obj = pick(OBJECTS + ANIMALS + FOODS)
        color = pick(COLORS)
        template = pick([
            f"the {obj} is {color}",
            f"i see a {color} {obj}",
            f"there is a {color} {obj}",
            f"that {obj} looks {color}",
            f"look at the {color} {obj}",
            f"{color} is the color of the {obj}",
        ])
        add(template, "simple_sentence", 1, "colors")


def gen_size_comparison():
    pairs = [
        ("elephant", "mouse"), ("whale", "fish"), ("lion", "cat"),
        ("horse", "dog"), ("mountain", "hill"), ("tree", "flower"),
        ("bus", "car"), ("ocean", "lake"), ("sun", "moon"),
        ("planet", "star"), ("house", "box"), ("river", "stream"),
    ]
    for big, small in pairs:
        add(f"a {big} is bigger than a {small}", "comparison", 2, "sizes")
        add(f"a {small} is smaller than a {big}", "comparison", 2, "sizes")

    for _ in range(500):
        a1, a2 = pick(ADJECTIVES, 2)
        obj = pick(OBJECTS)
        add(f"this {obj} is {a1} but that {obj} is {a2}", "comparison", 2, "sizes")


def gen_spatial_sentences():
    preps = ["on", "under", "next to", "behind", "in front of", "inside", "above", "below", "near", "between",
             "to the left of", "to the right of", "on top of", "at the bottom of", "beside", "across from"]
    for _ in range(800):
        o1, o2 = pick(OBJECTS, 2)
        prep = pick(preps)
        template = pick([
            f"the {o1} is {prep} the {o2}",
            f"put the {o1} {prep} the {o2}",
            f"i moved the {o1} {prep} the {o2}",
        ])
        add(template, "spatial", 2, "spatial")


def gen_action_sentences():
    time_phrases = ["today", "every day", "in the morning", "after school", "on weekends",
                    "at the park", "at home", "with friends", "by the lake", "in the garden"]
    for _ in range(1500):
        name = pick(NAMES)
        action = pick(ACTIONS)
        template = pick([
            f"{name} likes to {action}",
            f"{name} can {action}",
            f"{name} wants to {action}",
            f"i like to {action}",
            f"we can {action} together",
            f"let us {action}",
            f"{name} is going to {action} {pick(time_phrases)}",
            f"{name} and {pick(NAMES)} like to {action}",
            f"everyone can learn to {action}",
            f"it is fun to {action}",
        ])
        add(template, "simple_sentence", 1, "actions")


def gen_possession():
    for _ in range(600):
        name = pick(NAMES)
        obj = pick(OBJECTS + ANIMALS)
        color = pick(COLORS)
        template = pick([
            f"{name} has a {obj}",
            f"{name} has a {color} {obj}",
            f"the {obj} belongs to {name}",
            f"this is {name}'s {obj}",
        ])
        add(template, "simple_sentence", 1, "possession")


def gen_emotion_sentences():
    causes = {
        "happy": ["got a gift", "ate cake", "played with friends", "saw a rainbow", "won a game"],
        "sad": ["lost a toy", "said goodbye", "fell down", "missed a friend"],
        "angry": ["lost the game", "broke a toy", "was not fair"],
        "scared": ["heard a loud noise", "saw a spider", "it was very dark"],
        "tired": ["ran a lot", "played all day", "did not sleep"],
        "hungry": ["did not eat lunch", "smelled food cooking", "played outside for hours"],
        "excited": ["is going to the park", "got a new toy", "has a birthday soon"],
        "surprised": ["got a gift", "saw a magic trick", "found a secret"],
        "proud": ["finished a puzzle", "helped a friend", "learned something new"],
        "brave": ["tried something new", "was not afraid of the dark", "helped someone"],
        "curious": ["saw something new", "asked many questions", "found a bug"],
        "shy": ["met someone new", "was in front of many people"],
        "calm": ["took a deep breath", "sat in a quiet room", "listened to soft music"],
        "bored": ["had nothing to do", "waited for a long time"],
    }
    for emotion, cause_list in causes.items():
        for cause in cause_list:
            name = pick(NAMES)
            add(f"{name} feels {emotion} because {name} {cause}", "simple_sentence", 2, "emotions")
            add(f"when you feel {emotion} you can take a deep breath", "fact", 2, "emotions")


def gen_food_facts():
    """Nutritional and origin facts about food."""
    origins = {
        "apple": "grows on trees", "banana": "grows in warm places", "rice": "grows in wet fields",
        "wheat": "grows in fields and makes flour", "corn": "grows on tall stalks",
        "potato": "grows underground", "carrot": "grows underground and is orange",
        "tomato": "grows on a vine", "grape": "grows on a vine", "peanut": "grows underground",
        "coconut": "grows on palm trees", "orange": "grows on trees in warm places",
        "strawberry": "grows close to the ground", "watermelon": "grows on the ground and is very big",
        "mushroom": "grows in dark damp places", "chocolate": "is made from cacao beans",
        "bread": "is made from flour and water", "cheese": "is made from milk",
        "butter": "is made from cream", "yogurt": "is made from milk",
        "pasta": "is made from flour and eggs", "ice cream": "is made from cream and sugar",
        "jam": "is made from fruit and sugar", "honey": "is made by bees",
        "popcorn": "is made from corn kernels", "noodle": "is made from flour and water",
    }
    groups = {
        "fruit": ["apple", "banana", "orange", "grape", "strawberry", "watermelon", "pear", "peach", "mango", "lemon"],
        "vegetable": ["carrot", "potato", "tomato", "broccoli", "spinach", "onion", "pepper", "pea", "corn", "bean"],
        "grain": ["rice", "bread", "pasta", "cereal", "noodle"],
        "dairy": ["milk", "cheese", "butter", "yogurt", "ice cream"],
        "meat": ["chicken", "fish", "meat"],
    }
    for food, origin in origins.items():
        add(f"{food} {origin}", "fact", 2, "food")
    for group, foods in groups.items():
        for food in foods:
            add(f"{food} is a {group}", "fact", 1, "food")

    meals = {
        "breakfast": ["cereal", "egg", "bread", "milk", "banana", "yogurt"],
        "lunch": ["sandwich", "soup", "salad", "rice", "pasta"],
        "dinner": ["chicken", "fish", "rice", "potato", "soup", "pasta"],
        "snack": ["apple", "cookie", "popcorn", "cheese", "candy"],
    }
    for meal, foods in meals.items():
        for food in foods:
            add(f"you can eat {food} for {meal}", "fact", 1, "food")

def gen_food_sentences():
    tastes = {
        "apple": "sweet", "banana": "sweet", "lemon": "sour", "chocolate": "sweet",
        "candy": "sweet", "honey": "sweet", "cheese": "salty", "bread": "soft",
        "ice cream": "cold and sweet", "soup": "warm", "pepper": "spicy",
        "cookie": "crunchy and sweet", "watermelon": "juicy",
    }
    for food, taste in tastes.items():
        add(f"a {food} tastes {taste}", "fact", 1, "food")

    for _ in range(600):
        name = pick(NAMES)
        food = pick(FOODS)
        template = pick([
            f"{name} likes to eat {food}",
            f"{name} is eating {food}",
            f"we had {food} for lunch",
            f"i want some {food}",
            f"{food} is good to eat",
            f"can i have some {food}",
            f"{name} cooked {food} for dinner",
            f"do you like {food}",
            f"my favorite food is {food}",
            f"{name} shared some {food} with {pick(NAMES)}",
        ])
        add(template, "simple_sentence", 1, "food")


def gen_number_facts():
    facts = [
        "one plus one is two",
        "two plus two is four",
        "three plus three is six",
        "five plus five is ten",
        "ten minus five is five",
        "two times two is four",
        "three times three is nine",
        "a triangle has three sides",
        "a square has four sides",
        "a week has seven days",
        "a year has twelve months",
        "a day has twenty four hours",
        "an hour has sixty minutes",
        "a minute has sixty seconds",
        "a dozen means twelve",
        "zero means nothing",
        "one hundred has three digits",
        "ten plus ten is twenty",
        "half of ten is five",
        "double five is ten",
    ]
    for f in facts:
        add(f, "fact", 2, "numbers")

    # Counting
    for n in range(1, 21):
        add(f"the number {n} comes after {n - 1}", "fact", 1, "numbers")

    # Simple arithmetic
    for a in range(1, 11):
        for b in range(1, 11):
            if random.random() < 0.3:
                add(f"{a} plus {b} is {a + b}", "fact", 2, "numbers")
            if a >= b and random.random() < 0.2:
                add(f"{a} minus {b} is {a - b}", "fact", 2, "numbers")


def gen_time_sentences():
    for day in DAYS:
        add(f"today is {day}", "simple_sentence", 1, "time")
        activity = pick(["go to school", "play outside", "read a book", "rest at home",
                         "visit friends", "go to the park", "help at home"])
        add(f"on {day} we {activity}", "simple_sentence", 1, "time")

    for month in MONTHS:
        add(f"{month} is a month of the year", "fact", 1, "time")

    for season in SEASONS:
        traits = {
            "spring": "flowers bloom and birds sing",
            "summer": "it is warm and days are long",
            "fall": "leaves change color and fall down",
            "winter": "it is cold and it may snow",
        }
        add(f"in {season} {traits[season]}", "fact", 2, "time")

    time_facts = [
        "morning is when the sun comes up",
        "noon is the middle of the day",
        "afternoon comes after lunch",
        "evening is when the sun goes down",
        "night is when it is dark outside",
        "we eat breakfast in the morning",
        "we eat lunch in the afternoon",
        "we eat dinner in the evening",
        "we sleep at night",
        "yesterday was before today",
        "tomorrow comes after today",
        "a clock tells us what time it is",
        "the sun rises every morning",
        "the moon comes out at night",
    ]
    for f in time_facts:
        add(f, "fact", 1, "time")


def gen_family_sentences():
    relations = [
        ("mother", "mom"), ("father", "dad"), ("sister", "sis"),
        ("brother", "bro"), ("grandmother", "grandma"), ("grandfather", "grandpa"),
    ]
    facts = [
        "a family lives together",
        "a baby is very young",
        "children grow up to be adults",
        "parents take care of children",
        "grandparents are the parents of your parents",
        "a brother is a boy in your family",
        "a sister is a girl in your family",
        "an uncle is your parent's brother",
        "an aunt is your parent's sister",
        "a cousin is your uncle's or aunt's child",
        "families can be big or small",
        "families love each other",
        "a twin is born at the same time as you",
        "older children help younger children",
    ]
    for f in facts:
        add(f, "fact", 1, "family")
    for rel, short in relations:
        add(f"your {rel} loves you", "simple_sentence", 1, "family")
        add(f"you can call your {rel} {short}", "fact", 1, "family")


def gen_shape_sentences():
    props = {
        "circle": "round with no corners",
        "square": "has four equal sides",
        "triangle": "has three sides",
        "rectangle": "has four sides and two are longer",
        "star": "has points sticking out",
        "heart": "shaped like love",
        "diamond": "like a square turned sideways",
        "oval": "like a stretched circle",
        "cube": "a square in three dimensions",
        "sphere": "a circle in three dimensions",
        "cone": "has a point at the top and a circle at the bottom",
        "cylinder": "like a can or a tube",
    }
    for shape, desc in props.items():
        add(f"a {shape} is {desc}", "definition", 1, "shapes")
    for _ in range(100):
        obj = pick(OBJECTS)
        shape = pick(SHAPES)
        add(f"the {obj} is shaped like a {shape}", "simple_sentence", 1, "shapes")


def gen_material_sentences():
    examples = {
        "wood": ["table", "chair", "door", "ladder"],
        "metal": ["key", "coin", "nail", "bell"],
        "glass": ["window", "bottle", "mirror", "cup"],
        "paper": ["book", "map", "bag", "kite"],
        "cloth": ["hat", "bag", "blanket", "flag"],
        "plastic": ["bottle", "cup", "bag", "ball"],
    }
    for mat, objs in examples.items():
        for obj in objs:
            add(f"a {obj} can be made of {mat}", "fact", 2, "materials")
            add(f"this {obj} is made of {mat}", "simple_sentence", 1, "materials")

    for mat in MATERIALS:
        adj = pick(ADJECTIVES)
        add(f"{mat} is {adj}", "fact", 1, "materials")


def gen_plant_facts():
    facts = [
        "plants need sunlight to grow",
        "plants need water to live",
        "roots hold a plant in the ground",
        "leaves catch sunlight for the plant",
        "flowers can be many colors",
        "trees are the tallest plants",
        "grass covers the ground",
        "fruit grows on trees and bushes",
        "vegetables grow in gardens",
        "seeds become plants when they grow",
        "a sunflower faces the sun",
        "a cactus lives in the desert",
        "bamboo grows very fast",
        "roses often have thorns",
        "oak trees can be very old",
        "pine trees stay green all year",
        "maple trees have colorful leaves in fall",
        "wheat is used to make bread",
        "corn grows on tall stalks",
        "rice grows in wet fields",
    ]
    for f in facts:
        add(f, "fact", 2, "plants")

    for _ in range(100):
        plant = pick(PLANTS)
        color = pick(COLORS)
        adj = pick(["tall", "small", "beautiful", "green", "colorful", "old", "young"])
        template = pick([
            f"the {plant} is {adj}",
            f"i see a {adj} {plant}",
            f"this {plant} has {color} leaves",
        ])
        add(template, "simple_sentence", 1, "plants")


def gen_planet_facts():
    facts = {
        "Mercury": "is the closest planet to the sun",
        "Venus": "is the hottest planet",
        "Earth": "is where we live",
        "Mars": "is called the red planet",
        "Jupiter": "is the biggest planet",
        "Saturn": "has beautiful rings",
        "Uranus": "spins on its side",
        "Neptune": "is very far from the sun",
    }
    for planet, fact in facts.items():
        add(f"{planet} {fact}", "fact", 3, "space")

    space_facts = [
        "the sun is the closest star to earth",
        "the moon has no air",
        "astronauts go to space in rockets",
        "stars look tiny because they are very far away",
        "space is very big and mostly empty",
        "the milky way is our galaxy",
        "a day on earth is twenty four hours",
        "the earth tilts which gives us seasons",
        "the moon has craters from rocks that hit it",
        "a comet has a long bright tail",
    ]
    for f in space_facts:
        add(f, "fact", 3, "space")


def gen_job_sentences():
    duties = {
        "teacher": "helps children learn",
        "doctor": "helps sick people get better",
        "farmer": "grows food for people to eat",
        "pilot": "flies airplanes",
        "cook": "makes food in a kitchen",
        "nurse": "takes care of sick people",
        "driver": "drives a bus or a truck",
        "singer": "sings songs for people",
        "painter": "makes art with paint",
        "builder": "builds houses and buildings",
        "baker": "bakes bread and cakes",
        "firefighter": "puts out fires and helps people",
        "police officer": "keeps people safe",
        "dentist": "takes care of your teeth",
        "vet": "takes care of animals",
        "librarian": "helps people find books",
        "scientist": "studies how the world works",
        "astronaut": "travels to space",
    }
    for job, duty in duties.items():
        add(f"a {job} {duty}", "definition", 2, "jobs")
        name = pick(NAMES)
        add(f"{name} wants to be a {job}", "simple_sentence", 1, "jobs")


def gen_manners_and_social():
    facts = [
        "say please when you ask for something",
        "say thank you when someone helps you",
        "say sorry when you make a mistake",
        "be kind to others",
        "share your toys with friends",
        "take turns when you play",
        "listen when someone is talking",
        "do not shout indoors",
        "wash your hands before you eat",
        "brush your teeth every day",
        "cover your mouth when you cough",
        "be nice to animals",
        "help people who need it",
        "tell the truth",
        "clean up after yourself",
        "be patient and wait your turn",
        "say hello when you meet someone",
        "say goodbye when you leave",
        "look people in the eye when you talk",
        "smile at others to be friendly",
    ]
    for f in facts:
        add(f, "fact", 1, "manners")


def gen_music_sentences():
    for inst in INSTRUMENTS:
        add(f"you can play the {inst}", "simple_sentence", 1, "music")
        add(f"the {inst} makes a nice sound", "simple_sentence", 1, "music")

    facts = [
        "music is made of sounds",
        "a song has a melody",
        "you can clap to keep the beat",
        "singing is using your voice to make music",
        "dancing is moving your body to music",
        "a band plays music together",
        "a lullaby is a song to help you sleep",
        "music can make you feel happy or sad",
        "a concert is when people play music for others",
        "you can hum a tune without words",
    ]
    for f in facts:
        add(f, "fact", 1, "music")


def gen_opposites():
    pairs = [
        ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("tall", "short"),
        ("loud", "quiet"), ("happy", "sad"), ("light", "dark"), ("up", "down"),
        ("open", "close"), ("push", "pull"), ("wet", "dry"), ("hard", "soft"),
        ("full", "empty"), ("old", "new"), ("clean", "dirty"), ("long", "short"),
        ("thick", "thin"), ("heavy", "light"), ("near", "far"), ("wide", "narrow"),
        ("smooth", "rough"), ("sweet", "sour"), ("in", "out"), ("on", "off"),
        ("yes", "no"), ("day", "night"), ("left", "right"), ("front", "back"),
        ("start", "stop"), ("come", "go"),
    ]
    for a, b in pairs:
        add(f"the opposite of {a} is {b}", "fact", 1, "opposites")
        add(f"{a} and {b} are opposites", "fact", 1, "opposites")


def gen_cause_effect():
    pairs = [
        ("you water a plant", "it grows"),
        ("you drop a glass", "it breaks"),
        ("you turn off the light", "it gets dark"),
        ("you put ice in the sun", "it melts"),
        ("you mix red and blue", "you get purple"),
        ("you mix red and yellow", "you get orange"),
        ("you mix blue and yellow", "you get green"),
        ("you press a button", "the machine starts"),
        ("it rains a lot", "there are puddles"),
        ("you study hard", "you learn more"),
        ("you eat too much", "your stomach hurts"),
        ("you exercise", "you get stronger"),
        ("you practice", "you get better"),
        ("you are kind", "people like you"),
        ("you stay up late", "you feel tired"),
        ("it gets cold", "water turns to ice"),
        ("the sun comes out", "it gets warm"),
        ("you plant a seed", "a plant grows"),
        ("you push the swing", "it moves"),
        ("wind blows the leaves", "they fall down"),
        ("you blow out a candle", "the flame goes out"),
        ("you open the window", "fresh air comes in"),
        ("you put on a coat", "you stay warm"),
        ("you cut paper with scissors", "it splits in two"),
        ("you throw a ball up", "it comes back down"),
    ]
    for cause, effect in pairs:
        add(f"if {cause} then {effect}", "cause_effect", 2, "cause_effect")
        add(f"when {cause} {effect}", "cause_effect", 2, "cause_effect")


def gen_sequence_sentences():
    sequences = [
        "first you wake up then you brush your teeth then you eat breakfast",
        "first you put on socks then you put on shoes",
        "first you wash your hands then you eat",
        "first you mix the batter then you bake the cake",
        "first it rains then you see a rainbow",
        "first the seed grows roots then it grows a stem then it grows leaves",
        "first you open the book then you read the words",
        "first you learn to crawl then you learn to walk then you learn to run",
        "first you pick up the crayon then you draw a picture",
        "first you fill the cup then you drink the water",
        "first morning then afternoon then evening then night",
        "first spring then summer then fall then winter",
        "a caterpillar turns into a butterfly",
        "an egg hatches into a chick",
        "a tadpole turns into a frog",
    ]
    for s in sequences:
        add(s, "sequence", 2, "sequences")


def gen_context_qa():
    """Generate diverse context + question pairs."""
    for _ in range(2000):
        kind = random.randint(0, 5)
        if kind == 0:
            # Who has what
            n1, n2 = pick(NAMES, 2)
            o1, o2 = pick(OBJECTS, 2)
            text = f"{n1} has a {o1}. {n2} has a {o2}. what does {n1} have"
            add(text, "context_qa", 3, "memory")
        elif kind == 1:
            # Color memory
            o1, o2 = pick(OBJECTS, 2)
            c1, c2 = pick(COLORS, 2)
            text = f"the {o1} is {c1}. the {o2} is {c2}. what color is the {o1}"
            add(text, "context_qa", 3, "memory")
        elif kind == 2:
            # Who did what
            n1, n2 = pick(NAMES, 2)
            a1, a2 = pick(ACTIONS, 2)
            text = f"{n1} likes to {a1}. {n2} likes to {a2}. what does {n2} like to do"
            add(text, "context_qa", 3, "memory")
        elif kind == 3:
            # Where is it
            o1 = pick(OBJECTS)
            place = pick(["on the table", "under the bed", "in the box", "next to the door",
                          "behind the chair", "near the window", "on the shelf", "in the bag"])
            text = f"the {o1} is {place}. where is the {o1}"
            add(text, "context_qa", 3, "memory")
        elif kind == 4:
            # Food preference
            n1, n2 = pick(NAMES, 2)
            f1, f2 = pick(FOODS, 2)
            text = f"{n1} ate {f1}. {n2} ate {f2}. what did {n2} eat"
            add(text, "context_qa", 3, "memory")
        elif kind == 5:
            # Multi-fact
            n1 = pick(NAMES)
            animal = pick(ANIMALS)
            color = pick(COLORS)
            food = pick(FOODS)
            text = (f"{n1} has a {animal}. the {animal} is {color}. "
                    f"{n1} likes {food}. what color is the {animal}")
            add(text, "context_qa", 4, "multi_fact")


def gen_yes_no_qa():
    for _ in range(500):
        kind = random.randint(0, 3)
        if kind == 0:
            animal = pick(ANIMALS)
            trait = pick(["fly", "swim", "run fast", "climb trees", "live in water"])
            answer = pick(["yes", "no"])
            add(f"can a {animal} {trait}", "yn_qa", 2, "yn_qa")
        elif kind == 1:
            obj = pick(OBJECTS)
            prop = pick(["alive", "heavy", "soft", "round", "made of wood", "made of metal"])
            add(f"is a {obj} {prop}", "yn_qa", 2, "yn_qa")
        elif kind == 2:
            food = pick(FOODS)
            taste = pick(["sweet", "sour", "salty", "hot", "cold"])
            add(f"is {food} {taste}", "yn_qa", 2, "yn_qa")
        elif kind == 3:
            place = pick(PLACES)
            prop = pick(["big", "small", "inside", "outside", "cold", "hot"])
            add(f"is a {place} {prop}", "yn_qa", 2, "yn_qa")


def gen_category_qa():
    categories = {
        "animal": ANIMALS[:20],
        "food": FOODS[:20],
        "color": COLORS,
        "shape": SHAPES,
        "body part": BODY_PARTS[:15],
        "place": PLACES[:15],
    }
    for cat, members in categories.items():
        for member in members:
            add(f"a {member} is a type of {cat}", "fact", 2, "categories")
            if random.random() < 0.5:
                add(f"is a {member} a {cat}", "yn_qa", 2, "categories")


def gen_animal_habitat():
    habitats = {
        "forest": ["bear", "deer", "wolf", "fox", "owl", "squirrel"],
        "ocean": ["whale", "dolphin", "shark", "octopus", "crab", "seal"],
        "farm": ["cow", "horse", "pig", "sheep", "chicken", "goat", "duck"],
        "jungle": ["monkey", "tiger", "parrot", "snake", "frog"],
        "desert": ["camel", "snake", "spider"],
        "arctic": ["penguin", "seal", "polar bear"],
        "pond": ["frog", "duck", "turtle", "fish"],
        "sky": ["eagle", "hawk", "bird", "bat", "butterfly"],
        "garden": ["bee", "ant", "spider", "butterfly"],
        "house": ["dog", "cat", "fish", "rabbit", "mouse"],
    }
    for habitat, animals in habitats.items():
        for animal in animals:
            add(f"a {animal} lives in the {habitat}", "fact", 2, "habitats")


def gen_animal_sounds():
    sounds = {
        "dog": "bark", "cat": "meow", "cow": "moo", "pig": "oink",
        "sheep": "baa", "duck": "quack", "chicken": "cluck", "frog": "ribbit",
        "lion": "roar", "owl": "hoot", "bee": "buzz", "snake": "hiss",
        "horse": "neigh", "crow": "caw", "mouse": "squeak", "wolf": "howl",
        "goat": "bleat", "bird": "chirp", "elephant": "trumpet", "monkey": "chatter",
    }
    for animal, sound in sounds.items():
        add(f"a {animal} goes {sound}", "fact", 1, "animal_sounds")
        add(f"a {animal} can {sound}", "fact", 1, "animal_sounds")


def gen_animal_features():
    features = {
        "elephant": ["a long trunk", "big ears", "thick gray skin"],
        "giraffe": ["a very long neck", "long legs", "brown spots"],
        "zebra": ["black and white stripes"],
        "rabbit": ["long ears", "a fluffy tail", "soft fur"],
        "bird": ["feathers", "a beak", "wings"],
        "fish": ["fins", "scales", "gills"],
        "turtle": ["a hard shell", "short legs"],
        "spider": ["eight legs", "many eyes"],
        "octopus": ["eight arms", "no bones"],
        "shark": ["sharp teeth", "a big fin"],
        "lion": ["a big mane", "sharp claws"],
        "eagle": ["sharp eyes", "large wings", "a curved beak"],
        "kangaroo": ["a pouch", "strong back legs"],
        "penguin": ["a black and white body", "small wings"],
    }
    for animal, feats in features.items():
        for feat in feats:
            add(f"a {animal} has {feat}", "fact", 1, "animal_features")


def gen_daily_routines():
    routines = [
        "i wake up in the morning",
        "i brush my teeth after waking up",
        "i eat breakfast before school",
        "i go to school to learn",
        "i play with friends at school",
        "i eat lunch in the middle of the day",
        "i come home after school",
        "i do my homework in the afternoon",
        "i play outside when the weather is nice",
        "i eat dinner with my family",
        "i take a bath before bed",
        "i read a book before sleep",
        "i go to bed at night",
        "i sleep and dream at night",
        "i get dressed in the morning",
        "i put on my shoes before going outside",
        "i wash my hands before eating",
        "i drink water when i am thirsty",
        "i say goodnight before bed",
        "i hug my family before sleep",
    ]
    for r in routines:
        add(r, "simple_sentence", 1, "daily_life")


def gen_transport():
    vehicles = {
        "car": ["has four wheels", "drives on roads", "needs gas"],
        "bus": ["carries many people", "stops at bus stops", "is very big"],
        "train": ["runs on tracks", "is very long", "stops at stations"],
        "plane": ["flies in the sky", "has wings", "goes very fast"],
        "boat": ["floats on water", "has a sail or motor"],
        "bike": ["has two wheels", "you pedal to move it"],
        "truck": ["carries heavy things", "is bigger than a car"],
        "helicopter": ["can fly and hover", "has spinning blades on top"],
        "subway": ["goes underground", "carries people in cities"],
        "rocket": ["goes to space", "is very powerful"],
    }
    for v, facts in vehicles.items():
        for f in facts:
            add(f"a {v} {f}", "fact", 1, "transport")


def gen_senses():
    facts = [
        "you see with your eyes",
        "you hear with your ears",
        "you smell with your nose",
        "you taste with your tongue",
        "you touch with your hands",
        "eyes help you see colors and shapes",
        "ears help you hear music and voices",
        "your nose can smell flowers",
        "your tongue can taste sweet and sour",
        "your skin can feel hot and cold",
        "a loud sound is easy to hear",
        "a bright light is easy to see",
        "a strong smell is easy to notice",
        "a soft touch feels gentle",
        "we have five senses",
    ]
    for f in facts:
        add(f, "fact", 1, "senses")


def gen_safety():
    facts = [
        "look both ways before you cross the road",
        "do not touch hot things",
        "wear a helmet when you ride a bike",
        "do not talk to strangers",
        "tell an adult if something is wrong",
        "do not play with fire",
        "do not run near the pool",
        "always wear your seatbelt in a car",
        "be careful with sharp things like scissors",
        "do not put small things in your mouth",
        "hold an adult's hand when crossing the street",
        "do not lean out of windows",
        "stay away from deep water if you cannot swim",
        "do not play with electricity",
        "call for help if you are in danger",
    ]
    for f in facts:
        add(f, "fact", 2, "safety")


def gen_descriptive_scenes():
    """Short scene descriptions combining multiple concepts."""
    for _ in range(1500):
        kind = random.randint(0, 4)
        if kind == 0:
            color = pick(COLORS)
            animal = pick(ANIMALS)
            place = pick(PLACES)
            add(f"a {color} {animal} is in the {place}", "simple_sentence", 2, "scenes")
        elif kind == 1:
            name = pick(NAMES)
            action = pick(ACTIONS)
            place = pick(PLACES)
            add(f"{name} likes to {action} at the {place}", "simple_sentence", 2, "scenes")
        elif kind == 2:
            adj = pick(ADJECTIVES)
            obj = pick(OBJECTS)
            place = pick(["on the table", "in the room", "by the window", "on the floor",
                          "in the garden", "near the tree", "on the shelf"])
            add(f"there is a {adj} {obj} {place}", "simple_sentence", 1, "scenes")
        elif kind == 3:
            weather = pick(WEATHER)
            name = pick(NAMES)
            action = pick(["playing outside", "reading a book", "eating lunch",
                           "looking out the window", "walking in the park"])
            add(f"it is {weather} and {name} is {action}", "simple_sentence", 2, "scenes")
        elif kind == 4:
            animal = pick(ANIMALS)
            adj = pick(ADJECTIVES)
            action = pick(["running", "sleeping", "eating", "playing", "swimming", "sitting"])
            add(f"the {adj} {animal} is {action}", "simple_sentence", 1, "scenes")


def gen_reasoning():
    """Simple reasoning and logic patterns."""
    # Categorization reasoning
    for _ in range(100):
        a1, a2 = pick(ANIMALS, 2)
        add(f"a {a1} is an animal and a {a2} is also an animal", "fact", 2, "reasoning")

    for _ in range(100):
        f1, f2 = pick(FOODS, 2)
        add(f"both {f1} and {f2} are things you can eat", "fact", 2, "reasoning")

    # If-then patterns
    patterns = [
        "if it is raining you should bring an umbrella",
        "if you are hungry you should eat something",
        "if you are tired you should rest",
        "if you want to be strong you should exercise",
        "if you want to learn you should read books",
        "if it is dark you need a light",
        "if the cup is empty you need to fill it",
        "if the door is locked you need a key",
        "if you are cold you should wear a jacket",
        "if you are lost you should ask for help",
        "all dogs are animals but not all animals are dogs",
        "all apples are fruit but not all fruit are apples",
        "all squares are shapes but not all shapes are squares",
        "if something is alive it needs food and water",
        "if something is broken it does not work",
    ]
    for p in patterns:
        add(p, "reasoning", 3, "reasoning")


def gen_polysemy():
    """Words with multiple meanings to build richer representations."""
    examples = [
        ("a bat can be an animal that flies", "bat is an animal"),
        ("a bat can be used to hit a ball", "bat is a tool"),
        ("a ring can be jewelry you wear", "ring is jewelry"),
        ("a ring can be the sound a bell makes", "ring is a sound"),
        ("a bank is where you keep money", "bank for money"),
        ("a bank is the side of a river", "bank of a river"),
        ("a light can mean not heavy", "light means weight"),
        ("a light can mean something that shines", "light means bright"),
        ("to fly means to move through the air", "fly in the air"),
        ("a fly is a small insect", "fly the insect"),
        ("a wave is water moving in the ocean", "ocean wave"),
        ("to wave means to move your hand to say hello", "wave hello"),
    ]
    for text, _ in examples:
        add(text, "fact", 3, "word_meanings")


def gen_comparatives():
    """More/less, -er/-est patterns."""
    for _ in range(600):
        adj = pick(["big", "small", "fast", "slow", "tall", "short", "old", "young",
                     "heavy", "light", "long", "loud", "quiet", "strong", "weak"])
        a1, a2 = pick(ANIMALS + OBJECTS, 2)
        template = pick([
            f"the {a1} is {adj}er than the {a2}",
            f"the {a1} is more {adj} than the {a2}",
            f"which is {adj}er the {a1} or the {a2}",
        ])
        add(template, "comparison", 2, "comparisons")


def gen_negation():
    """Sentences with negation to learn 'not'."""
    for _ in range(500):
        kind = random.randint(0, 3)
        if kind == 0:
            animal = pick(ANIMALS)
            trait = pick(["a plant", "a color", "a food", "a number", "a shape"])
            add(f"a {animal} is not {trait}", "negation", 2, "negation")
        elif kind == 1:
            obj = pick(OBJECTS)
            prop = pick(["alive", "an animal", "food", "a person"])
            add(f"a {obj} is not {prop}", "negation", 2, "negation")
        elif kind == 2:
            animal = pick(ANIMALS)
            action = pick(["fly", "swim", "talk", "read", "drive a car", "cook"])
            add(f"a {animal} cannot {action}", "negation", 2, "negation")
        elif kind == 3:
            color1, color2 = pick(COLORS, 2)
            add(f"{color1} is not the same as {color2}", "negation", 1, "negation")


def gen_quantity():
    """Counting and quantity concepts."""
    for _ in range(400):
        n = random.randint(1, 10)
        obj = pick(OBJECTS + ANIMALS)
        s = "s" if n > 1 else ""
        add(f"there are {n} {obj}{s}", "simple_sentence", 1, "quantity")

    quantity_words = [
        "many", "few", "some", "all", "none", "most", "several",
    ]
    for q in quantity_words:
        obj = pick(OBJECTS + ANIMALS)
        add(f"{q} {obj}s are here", "simple_sentence", 2, "quantity")
        add(f"there are {q} {obj}s", "simple_sentence", 2, "quantity")


def gen_adj_noun_sentences():
    """Large-scale adjective+noun combinations for vocabulary breadth."""
    templates = [
        "the {a} {n} is here",
        "i found a {a} {n}",
        "look at the {a} {n}",
        "that is a very {a} {n}",
        "have you seen the {a} {n}",
        "the {a} {n} is nice",
        "i want the {a} {n}",
        "where is the {a} {n}",
    ]
    for _ in range(3000):
        adj = pick(ADJECTIVES)
        noun = pick(OBJECTS + ANIMALS + FOODS + PLANTS)
        t = pick(templates)
        add(t.format(a=adj, n=noun), "simple_sentence", 1, "descriptions")


def gen_two_clause():
    """Two-clause sentences with 'and', 'but', 'so', 'because'."""
    conjunctions = ["and", "but", "so", "because"]
    clause_templates = [
        "{n} likes to {a}",
        "the {o} is {adj}",
        "it is {w} outside",
        "{n} has a {o}",
        "{n} is {e}",
        "{n} ate {f}",
        "{n} went to the {p}",
        "the {o} is {c}",
    ]
    for _ in range(3000):
        conj = pick(conjunctions)
        n1, n2 = pick(NAMES, 2)
        o1, o2 = pick(OBJECTS, 2)
        a1, a2 = pick(ACTIONS, 2)
        adj1, adj2 = pick(ADJECTIVES, 2)
        e1, e2 = pick(EMOTIONS, 2)
        w1 = pick(WEATHER)
        f1 = pick(FOODS)

        p1, p2 = pick(PLACES, 2)
        c1_color, c2_color = pick(COLORS, 2)
        c1 = pick(clause_templates).format(n=n1, a=a1, o=o1, adj=adj1, w=w1, e=e1, f=f1, p=p1, c=c1_color)
        c2 = pick(clause_templates).format(n=n2, a=a2, o=o2, adj=adj2, w=w1, e=e2, f=f1, p=p2, c=c2_color)
        add(f"{c1} {conj} {c2}", "simple_sentence", 2, "compound")


def gen_where_what_who_qa():
    """Question-word variety."""
    for _ in range(1000):
        kind = random.randint(0, 4)
        if kind == 0:
            animal = pick(ANIMALS)
            add(f"what sound does a {animal} make", "qa", 2, "knowledge_qa")
        elif kind == 1:
            obj = pick(OBJECTS)
            add(f"what is a {obj} used for", "qa", 2, "knowledge_qa")
        elif kind == 2:
            name = pick(NAMES)
            place = pick(PLACES)
            add(f"where is {name} going. {name} is going to the {place}", "qa", 2, "knowledge_qa")
        elif kind == 3:
            job = pick(JOBS)
            add(f"who helps you when you are sick. a {job} helps you", "qa", 2, "knowledge_qa")
        elif kind == 4:
            food = pick(FOODS)
            adj = pick(["sweet", "sour", "salty", "crunchy", "soft", "cold", "warm"])
            add(f"how does {food} taste. {food} is {adj}", "qa", 2, "knowledge_qa")


def gen_story_snippets():
    """Tiny 2-3 sentence stories."""
    for _ in range(1600):
        name = pick(NAMES)
        kind = random.randint(0, 5)
        if kind == 0:
            animal = pick(ANIMALS)
            color = pick(COLORS)
            action = pick(["played with", "fed", "saw", "found", "petted"])
            add(f"{name} {action} a {color} {animal}. {name} was happy",
                "simple_sentence", 2, "stories")
        elif kind == 1:
            place = pick(PLACES)
            weather = pick(WEATHER)
            add(f"{name} went to the {place}. it was {weather}. {name} had fun",
                "simple_sentence", 2, "stories")
        elif kind == 2:
            food = pick(FOODS)
            emotion = pick(EMOTIONS)
            add(f"{name} ate some {food}. it was very good. {name} felt {emotion}",
                "simple_sentence", 2, "stories")
        elif kind == 3:
            obj = pick(OBJECTS)
            color = pick(COLORS)
            add(f"{name} lost a {color} {obj}. {name} looked everywhere. {name} found it under the bed",
                "simple_sentence", 3, "stories")
        elif kind == 4:
            friend = pick(NAMES)
            action = pick(ACTIONS)
            add(f"{name} and {friend} like to {action}. they are good friends",
                "simple_sentence", 2, "stories")
        elif kind == 5:
            animal = pick(ANIMALS)
            adj = pick(ADJECTIVES)
            add(f"once there was a {adj} {animal}. it lived in the {pick(PLACES)}. it was very {pick(EMOTIONS)}",
                "simple_sentence", 2, "stories")


def gen_analogy():
    """Simple analogies: A is to B as C is to D."""
    pairs = [
        ("hot", "cold", "big", "small"),
        ("day", "night", "light", "dark"),
        ("cat", "kitten", "dog", "puppy"),
        ("hand", "finger", "foot", "toe"),
        ("bird", "fly", "fish", "swim"),
        ("eye", "see", "ear", "hear"),
        ("teacher", "school", "doctor", "hospital"),
        ("rain", "wet", "sun", "dry"),
        ("book", "read", "song", "sing"),
        ("key", "lock", "lid", "jar"),
        ("pen", "write", "brush", "paint"),
        ("bee", "honey", "cow", "milk"),
    ]
    for a, b, c, d in pairs:
        add(f"{a} is to {b} as {c} is to {d}", "reasoning", 3, "analogies")
        add(f"just like {a} goes with {b} so {c} goes with {d}", "reasoning", 3, "analogies")


def gen_plurals_and_articles():
    """Practice with a/an/the and plural forms."""
    vowel_start = ["apple", "egg", "elephant", "orange", "owl", "ant", "octopus",
                   "otter", "umbrella", "onion", "ear", "eye", "ice cream"]
    for word in vowel_start:
        add(f"you say an {word} not a {word}", "fact", 2, "grammar")

    for _ in range(300):
        obj = pick(OBJECTS + ANIMALS)
        n = random.randint(2, 10)
        add(f"one {obj} but {n} {obj}s", "fact", 1, "grammar")


def gen_color_mixing():
    mixes = [
        ("red", "blue", "purple"),
        ("red", "yellow", "orange"),
        ("blue", "yellow", "green"),
        ("red", "white", "pink"),
        ("black", "white", "gray"),
        ("blue", "white", "light blue"),
        ("red", "green", "brown"),
    ]
    for a, b, result in mixes:
        add(f"if you mix {a} and {b} you get {result}", "fact", 2, "colors")
        add(f"{a} and {b} make {result}", "fact", 2, "colors")


def gen_animal_diet():
    """What animals eat."""
    diets = {
        "cow": ["grass", "hay"], "horse": ["grass", "hay", "apple"],
        "rabbit": ["carrot", "grass", "lettuce"], "cat": ["fish", "meat"],
        "dog": ["meat", "bones"], "bird": ["seeds", "worms"],
        "monkey": ["banana", "fruit"], "bear": ["fish", "honey", "berries"],
        "elephant": ["grass", "fruit", "leaves"], "panda": ["bamboo"],
        "koala": ["leaves"], "penguin": ["fish"], "frog": ["bugs", "flies"],
        "spider": ["bugs", "flies"], "owl": ["mice", "small animals"],
        "shark": ["fish"], "whale": ["small fish", "tiny sea creatures"],
        "chicken": ["seeds", "corn", "bugs"], "pig": ["almost anything"],
        "goat": ["grass", "leaves", "almost anything"],
        "squirrel": ["nuts", "seeds", "acorns"],
    }
    for animal, foods in diets.items():
        for food in foods:
            add(f"a {animal} eats {food}", "fact", 1, "animal_diet")
        add(f"a {animal} is hungry and wants to eat", "simple_sentence", 1, "animal_diet")


def gen_place_descriptions():
    """What you find and do at various places."""
    place_info = {
        "school": ["you learn at school", "there are teachers at school", "children go to school"],
        "park": ["you can play at the park", "there are trees at the park", "you can run at the park"],
        "beach": ["there is sand at the beach", "you can swim at the beach", "waves come to the beach"],
        "farm": ["animals live on the farm", "crops grow on the farm", "the farmer works on the farm"],
        "forest": ["many trees grow in the forest", "wild animals live in the forest", "the forest is green"],
        "library": ["books are in the library", "you read at the library", "the library is quiet"],
        "zoo": ["you can see animals at the zoo", "the zoo has many animals", "the zoo is fun to visit"],
        "kitchen": ["you cook in the kitchen", "food is in the kitchen", "the kitchen has a stove"],
        "bedroom": ["you sleep in the bedroom", "the bed is in the bedroom", "the bedroom is for resting"],
        "garden": ["flowers grow in the garden", "you water plants in the garden", "the garden is colorful"],
        "hospital": ["doctors work at the hospital", "sick people go to the hospital"],
        "store": ["you buy things at the store", "the store has many things to sell"],
        "mountain": ["mountains are very tall", "you can climb a mountain", "snow is on top of mountains"],
        "lake": ["you can fish at the lake", "the lake has clean water", "ducks swim on the lake"],
        "river": ["a river flows with water", "fish live in the river", "rivers go to the sea"],
        "desert": ["the desert is hot and dry", "sand covers the desert", "cactus grows in the desert"],
    }
    for place, facts in place_info.items():
        for f in facts:
            add(f, "fact", 1, "places")


def gen_multi_step_qa():
    """QA requiring two facts to answer."""
    for _ in range(800):
        kind = random.randint(0, 3)
        if kind == 0:
            n1, n2, n3 = pick(NAMES, 3)
            o1, o2, o3 = pick(OBJECTS, 3)
            c1, c2, c3 = pick(COLORS, 3)
            text = (f"{n1} has a {c1} {o1}. {n2} has a {c2} {o2}. "
                    f"{n3} has a {c3} {o3}. what color is {n2}'s {o2}")
            add(text, "context_qa", 4, "multi_fact")
        elif kind == 1:
            n1, n2 = pick(NAMES, 2)
            a1, a2 = pick(ANIMALS, 2)
            f1, f2 = pick(FOODS, 2)
            text = (f"{n1} has a {a1} and likes {f1}. "
                    f"{n2} has a {a2} and likes {f2}. "
                    f"who has a {a1}")
            add(text, "context_qa", 4, "multi_fact")
        elif kind == 2:
            n1, n2 = pick(NAMES, 2)
            place1, place2 = pick(PLACES, 2)
            action1, action2 = pick(ACTIONS, 2)
            text = (f"{n1} went to the {place1} to {action1}. "
                    f"{n2} went to the {place2} to {action2}. "
                    f"where did {n1} go")
            add(text, "context_qa", 4, "multi_fact")
        elif kind == 3:
            n1, n2 = pick(NAMES, 2)
            e1, e2 = pick(EMOTIONS, 2)
            cause1 = pick(["got a gift", "ate cake", "played with friends",
                           "lost a toy", "fell down", "won a game"])
            cause2 = pick(["heard a noise", "saw a rainbow", "found a coin",
                           "missed lunch", "broke a cup", "learned to swim"])
            text = (f"{n1} is {e1} because {n1} {cause1}. "
                    f"{n2} is {e2} because {n2} {cause2}. "
                    f"how does {n1} feel")
            add(text, "context_qa", 4, "multi_fact")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def deduplicate(data: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for item in data:
        key = item["text"]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def main():
    generators = [
        ("animal definitions", gen_animal_definitions),
        ("object definitions", gen_object_definitions),
        ("body facts", gen_body_facts),
        ("science facts", gen_science_facts),
        ("geography facts", gen_geography_facts),
        ("weather sentences", gen_weather_sentences),
        ("color sentences", gen_color_sentences),
        ("size comparisons", gen_size_comparison),
        ("spatial sentences", gen_spatial_sentences),
        ("action sentences", gen_action_sentences),
        ("possession sentences", gen_possession),
        ("emotion sentences", gen_emotion_sentences),
        ("food facts", gen_food_facts),
        ("food sentences", gen_food_sentences),
        ("number facts", gen_number_facts),
        ("time sentences", gen_time_sentences),
        ("family sentences", gen_family_sentences),
        ("shape sentences", gen_shape_sentences),
        ("material sentences", gen_material_sentences),
        ("plant facts", gen_plant_facts),
        ("planet facts", gen_planet_facts),
        ("job sentences", gen_job_sentences),
        ("manners", gen_manners_and_social),
        ("music sentences", gen_music_sentences),
        ("opposites", gen_opposites),
        ("cause and effect", gen_cause_effect),
        ("sequences", gen_sequence_sentences),
        ("context QA", gen_context_qa),
        ("yes/no QA", gen_yes_no_qa),
        ("category QA", gen_category_qa),
        ("animal habitats", gen_animal_habitat),
        ("animal sounds", gen_animal_sounds),
        ("animal features", gen_animal_features),
        ("daily routines", gen_daily_routines),
        ("transport", gen_transport),
        ("senses", gen_senses),
        ("safety", gen_safety),
        ("descriptive scenes", gen_descriptive_scenes),
        ("reasoning", gen_reasoning),
        ("polysemy", gen_polysemy),
        ("comparatives", gen_comparatives),
        ("negation", gen_negation),
        ("quantity", gen_quantity),
        ("adj+noun descriptions", gen_adj_noun_sentences),
        ("two-clause sentences", gen_two_clause),
        ("where/what/who QA", gen_where_what_who_qa),
        ("story snippets", gen_story_snippets),
        ("analogies", gen_analogy),
        ("plurals and articles", gen_plurals_and_articles),
        ("color mixing", gen_color_mixing),
        ("animal diet", gen_animal_diet),
        ("place descriptions", gen_place_descriptions),
        ("multi-step QA", gen_multi_step_qa),
    ]

    for name, gen_fn in generators:
        before = len(entries)
        gen_fn()
        after = len(entries)
        print(f"  {name}: +{after - before} entries")

    # Deduplicate
    unique = deduplicate(entries)
    print(f"\nTotal before dedup: {len(entries)}")
    print(f"Total after dedup:  {len(unique)}")

    # Remove overlap with existing curriculum
    curriculum_path = OUTPUT_PATH.parent / "text_curriculum.json"
    if curriculum_path.exists():
        existing = json.load(open(curriculum_path))
        existing_texts = {item["text"] for item in existing}
        before_filter = len(unique)
        unique = [item for item in unique if item["text"] not in existing_texts]
        removed = before_filter - len(unique)
        if removed:
            print(f"Removed {removed} items overlapping with text_curriculum.json")

    # Shuffle
    random.shuffle(unique)

    # Category stats
    cats: dict[str, int] = {}
    types: dict[str, int] = {}
    levels: dict[int, int] = {}
    for item in unique:
        cats[item["category"]] = cats.get(item["category"], 0) + 1
        types[item["type"]] = types.get(item["type"], 0) + 1
        levels[item["level"]] = levels.get(item["level"], 0) + 1

    print(f"\nCategories ({len(cats)}):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nTypes ({len(types)}):")
    for typ, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {typ}: {count}")

    print(f"\nLevels:")
    for lvl in sorted(levels):
        print(f"  level {lvl}: {levels[lvl]}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(unique, f, indent=1)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(unique)} entries to {OUTPUT_PATH}")
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
