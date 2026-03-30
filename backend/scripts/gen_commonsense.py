"""
Generate ~10,000 common-sense / world-knowledge training entries.

Usage:  python3 backend/scripts/gen_commonsense.py
Output: backend/data/text_commonsense.json

Deterministic (random.seed(43)), stdlib only.
"""

import json
import random
import os

random.seed(43)

entries: list[dict] = []


def add(text: str, typ: str, level: int, category: str,
        answer: str | None = None):
    e: dict = {"text": text, "type": typ, "level": level, "category": category}
    if answer is not None:
        e["answer"] = answer
    entries.append(e)


def add_statement_and_qa(statement: str, question: str, answer: str,
                         typ: str, level: int, category: str):
    """Add both a plain statement and a QA pair."""
    add(statement, typ, level, category)
    add(f"{question}", "qa", level + 1 if level < 4 else 4, category,
        answer=answer)


# ── helpers ───────────────────────────────────────────────────────────
def pick(lst, n=1):
    return random.sample(lst, min(n, len(lst)))


def a_an(word: str) -> str:
    return "an " + word if word[0] in "aeiou" else "a " + word


# =====================================================================
# 1. PHYSICAL WORLD  (~2,000)
# =====================================================================

# --- gravity / weight ---
heavy_things = ["rock", "brick", "anvil", "boulder", "iron ball", "bag of sand",
                "heavy box", "big stone", "metal block", "lead weight"]
light_things = ["feather", "leaf", "balloon", "bubble", "cotton ball",
                "paper plane", "dandelion seed", "dust", "snowflake", "soap bubble"]
fall_objects = ["ball", "apple", "cup", "book", "toy", "spoon", "plate",
                "shoe", "hat", "coin", "pencil", "block", "phone", "key", "rock"]

for obj in fall_objects:
    add(f"if you drop {a_an(obj)} it falls down", "physics", 1, "commonsense")
    add(f"what happens if you drop {a_an(obj)}", "qa", 2, "commonsense",
        answer="it falls down")

for obj in heavy_things:
    add(f"{a_an(obj)} is heavy", "physics", 1, "commonsense")
    add(f"{a_an(obj)} sinks in water", "physics", 2, "commonsense")

for obj in light_things:
    add(f"{a_an(obj)} is very light", "physics", 1, "commonsense")
    add(f"{a_an(obj)} floats in the air", "physics", 2, "commonsense")

add("things fall down because of gravity", "physics", 2, "commonsense")
add("heavy things fall faster through water", "physics", 2, "commonsense")
add("a stone sinks but a stick floats", "physics", 2, "commonsense")
add("if you throw a ball up it comes back down", "physics", 1, "commonsense")
add("water flows downhill", "physics", 1, "commonsense")
add("rivers flow from mountains to the sea", "physics", 2, "commonsense")
add("rain falls from the sky to the ground", "physics", 1, "commonsense")
add("snow falls gently from the clouds", "physics", 1, "commonsense")
add("a ball rolls down a hill", "physics", 1, "commonsense")
add("heavy things are hard to lift", "physics", 1, "commonsense")
add("light things are easy to carry", "physics", 1, "commonsense")
add("what makes things fall down", "qa", 2, "commonsense", answer="gravity")
add("what happens when you throw a ball up", "qa", 2, "commonsense",
    answer="it comes back down")
add("where does rain come from", "qa", 2, "commonsense", answer="the clouds")
add("which way does water flow", "qa", 2, "commonsense", answer="downhill")

# --- temperature / fire / ice ---
hot_things = ["fire", "the sun", "boiling water", "a stove", "hot soup",
              "a campfire", "lava", "a hot pan", "a heater", "a candle flame"]
cold_things = ["ice", "snow", "a freezer", "cold water", "winter wind",
               "a snowball", "an ice cube", "a cold drink", "frost", "hail"]

for h in hot_things:
    add(f"{h} is very hot", "physics", 1, "commonsense")
    add(f"do not touch {h} or you will get burned", "cause_effect", 2, "commonsense")

for c in cold_things:
    add(f"{c} is very cold", "physics", 1, "commonsense")

add("if you heat ice it melts into water", "cause_effect", 2, "science")
add("if you heat water it turns into steam", "cause_effect", 2, "science")
add("if you cool steam it turns back into water", "cause_effect", 2, "science")
add("if you cool water it turns into ice", "cause_effect", 2, "science")
add("fire needs air to keep burning", "physics", 2, "science")
add("water puts out fire", "physics", 1, "commonsense")
add("the sun keeps the earth warm", "physics", 2, "science")
add("in summer it is warm and in winter it is cold", "physics", 1, "commonsense")
add("hot air rises up", "physics", 2, "science")
add("cold things warm up in a hot room", "physics", 2, "commonsense")
add("hot things cool down over time", "physics", 2, "commonsense")
add("what happens when you heat ice", "qa", 2, "science", answer="it melts into water")
add("what happens when you heat water a lot", "qa", 2, "science", answer="it turns into steam")
add("what puts out fire", "qa", 1, "commonsense", answer="water")
add("what are the three forms of water", "qa", 3, "science",
    answer="ice and water and steam")
add("why does ice melt", "qa", 2, "science", answer="because it gets warm")

# --- states of matter ---
solids = ["rock", "wood", "metal", "glass", "ice", "brick", "bone", "diamond"]
liquids = ["water", "milk", "juice", "oil", "soup", "rain"]
gases = ["air", "steam", "smoke", "wind"]

for s in solids:
    add(f"{s} is a solid", "physics", 2, "science")
for l in liquids:
    add(f"{l} is a liquid", "physics", 2, "science")
for g in gases:
    add(f"{g} is a gas", "physics", 2, "science")

add("a solid has a fixed shape", "physics", 2, "science")
add("a liquid takes the shape of its container", "physics", 2, "science")
add("a gas fills the whole room", "physics", 2, "science")
add("you can pour a liquid but not a solid", "physics", 2, "science")
add("you can not see most gases", "physics", 2, "science")
add("ice is solid water", "physics", 2, "science")
add("steam is water as a gas", "physics", 2, "science")

# --- light and dark ---
add("the sun gives us light during the day", "physics", 1, "commonsense")
add("the moon shines at night", "physics", 1, "commonsense")
add("stars come out at night", "physics", 1, "commonsense")
add("shadows appear when something blocks light", "physics", 2, "commonsense")
add("a shadow is dark", "physics", 1, "commonsense")
add("when the sun sets it gets dark", "physics", 1, "commonsense")
add("when the sun rises it gets light", "physics", 1, "commonsense")
add("light helps us see", "physics", 1, "commonsense")
add("we cannot see in the dark", "physics", 1, "commonsense")
add("a flashlight makes light", "physics", 1, "commonsense")
add("a lamp lights up a room", "physics", 1, "commonsense")
add("colors come from light", "physics", 2, "science")
add("a rainbow has many colors", "physics", 1, "commonsense")
add("a rainbow appears after rain when the sun shines", "physics", 2, "commonsense")
add("the sky is blue during the day", "physics", 1, "commonsense")
add("the sky gets orange and pink at sunset", "physics", 1, "commonsense")
add("when is it dark outside", "qa", 1, "commonsense", answer="at night")
add("what makes shadows", "qa", 2, "commonsense",
    answer="something blocking the light")
add("when do stars come out", "qa", 1, "commonsense", answer="at night")
add("what color is the sky during the day", "qa", 1, "commonsense", answer="blue")

# --- sound ---
loud_things = ["thunder", "a drum", "a siren", "a shout", "a horn",
               "fireworks", "a lion roar", "a crash", "an alarm", "a jet"]
quiet_things = ["a whisper", "a mouse", "falling snow", "a feather landing",
                "a soft breeze", "a cat walking", "a yawn", "a sigh"]

for s in loud_things:
    add(f"{s} is very loud", "physics", 1, "commonsense")
for s in quiet_things:
    add(f"{s} is very quiet", "physics", 1, "commonsense")

add("sound travels through the air", "physics", 2, "science")
add("an echo is a sound that bounces back", "physics", 2, "commonsense")
add("thunder comes after lightning", "physics", 2, "commonsense")
add("if you shout in a cave you hear an echo", "cause_effect", 2, "commonsense")
add("covering your ears makes sounds quieter", "physics", 1, "commonsense")
add("music is made of sounds", "physics", 1, "commonsense")
add("what comes after lightning", "qa", 2, "commonsense", answer="thunder")
add("what is an echo", "qa", 2, "commonsense",
    answer="a sound that bounces back")

# --- material properties ---
materials = [
    ("glass", "see through", "breaks easily"),
    ("wood", "hard", "floats on water"),
    ("metal", "strong", "feels cold"),
    ("rubber", "stretchy", "bounces"),
    ("paper", "thin", "tears easily"),
    ("cloth", "soft", "can be folded"),
    ("plastic", "light", "can be many shapes"),
    ("stone", "very hard", "heavy"),
    ("cotton", "soft and fluffy", "keeps you warm"),
    ("leather", "tough", "comes from animal skin"),
    ("clay", "soft when wet", "gets hard when baked"),
    ("sand", "made of tiny grains", "found at the beach"),
]

for mat, prop1, prop2 in materials:
    add(f"{mat} is {prop1}", "physics", 1, "commonsense")
    add(f"{mat} {prop2}", "physics", 2, "commonsense")

# --- motion and force ---
for verb, result in [
    ("push", "moves away"), ("pull", "comes toward you"),
    ("kick", "flies forward"), ("throw", "goes through the air"),
    ("spin", "turns around"), ("squeeze", "gets smaller"),
    ("stretch", "gets longer"), ("bend", "changes shape"),
    ("shake", "moves back and forth"), ("roll", "turns over and over"),
]:
    add(f"when you {verb} something it {result}", "physics", 1, "commonsense")

add("wheels make things easier to move", "physics", 2, "commonsense")
add("smooth floors are easier to slide on", "physics", 2, "commonsense")
add("rough surfaces slow things down", "physics", 2, "commonsense")
add("a magnet pulls metal toward it", "physics", 2, "science")
add("things with wheels roll", "physics", 1, "commonsense")
add("a seesaw goes up on one side and down on the other", "physics", 1, "commonsense")

# --- weather ---
weather_types = [
    ("sunny", "the sun is shining", "warm"),
    ("rainy", "water falls from the clouds", "wet"),
    ("snowy", "white snow covers the ground", "cold"),
    ("windy", "the air moves fast", "breezy"),
    ("cloudy", "clouds cover the sky", "gray"),
    ("foggy", "you cannot see far", "damp"),
    ("stormy", "there is thunder and lightning", "scary"),
]

for w, desc, feel in weather_types:
    add(f"when it is {w} {desc}", "physics", 1, "commonsense")
    add(f"a {w} day feels {feel}", "physics", 1, "commonsense")

add("clouds are made of tiny water drops", "physics", 2, "science")
add("rain comes from clouds", "physics", 1, "commonsense")
add("snow is frozen water", "physics", 1, "commonsense")
add("wind is moving air", "physics", 2, "science")
add("puddles form when it rains", "physics", 1, "commonsense")
add("puddles dry up when the sun comes out", "cause_effect", 2, "commonsense")
add("what are clouds made of", "qa", 2, "science",
    answer="tiny water drops")
add("what happens to puddles in the sun", "qa", 2, "commonsense",
    answer="they dry up")

# =====================================================================
# 2. LIVING THINGS  (~2,000)
# =====================================================================

# --- animals ---
animals_info = [
    ("dog", "bark", "meat and kibble", "puppy", "a house or yard", "pack"),
    ("cat", "meow", "fish and meat", "kitten", "a house", "clowder"),
    ("cow", "moo", "grass", "calf", "a farm", "herd"),
    ("horse", "neigh", "hay and grass", "foal", "a stable", "herd"),
    ("pig", "oink", "scraps and grain", "piglet", "a farm", "drove"),
    ("sheep", "baa", "grass", "lamb", "a farm", "flock"),
    ("chicken", "cluck", "seeds and bugs", "chick", "a coop", "flock"),
    ("duck", "quack", "plants and bugs", "duckling", "a pond", "flock"),
    ("lion", "roar", "meat", "cub", "the savanna", "pride"),
    ("elephant", "trumpet", "plants and fruit", "calf", "the grasslands", "herd"),
    ("monkey", "chatter", "fruit and bugs", "baby monkey", "the jungle", "troop"),
    ("fish", "splash", "smaller fish and plants", "fry", "water", "school"),
    ("bird", "chirp", "seeds and worms", "chick", "a nest in a tree", "flock"),
    ("frog", "croak", "bugs", "tadpole", "a pond", "group"),
    ("bear", "growl", "fish and berries", "cub", "the forest", "sleuth"),
    ("rabbit", "squeak", "carrots and grass", "bunny", "a burrow", "colony"),
    ("snake", "hiss", "mice and eggs", "snakelet", "the ground", "nest"),
    ("bee", "buzz", "nectar", "larva", "a hive", "swarm"),
    ("ant", "click", "crumbs and leaves", "larva", "an anthill", "colony"),
    ("owl", "hoot", "mice", "owlet", "a tree", "parliament"),
    ("whale", "sing", "tiny sea creatures", "calf", "the ocean", "pod"),
    ("dolphin", "click and whistle", "fish", "calf", "the ocean", "pod"),
    ("tiger", "growl", "meat", "cub", "the jungle", "streak"),
    ("penguin", "squawk", "fish", "chick", "the ice", "colony"),
    ("giraffe", "hum", "leaves from tall trees", "calf", "the savanna", "tower"),
    ("zebra", "bark", "grass", "foal", "the savanna", "herd"),
    ("turtle", "grunt", "plants and bugs", "hatchling", "a pond", "bale"),
    ("butterfly", "flutter", "nectar", "caterpillar", "a garden", "swarm"),
    ("spider", "hiss", "bugs", "spiderling", "a web", "cluster"),
    ("mouse", "squeak", "cheese and seeds", "pup", "a hole", "nest"),
]

for name, sound, food, baby, home, group in animals_info:
    a = a_an(name)
    add(f"{a} says {sound}", "fact", 1, "animals")
    add(f"{a} eats {food}", "fact", 1, "animals")
    add(f"a baby {name} is called {a_an(baby)}", "fact", 2, "animals")
    add(f"{a} lives in {home}", "fact", 1, "animals")
    add(f"a group of {name}s is called {a_an(group)}", "fact", 3, "animals")
    add(f"what sound does {a} make", "qa", 1, "animals", answer=sound)
    add(f"what does {a} eat", "qa", 2, "animals", answer=food)
    add(f"where does {a} live", "qa", 2, "animals", answer=home)

# animal features
features = [
    ("fish", "fins and scales", "swim"),
    ("bird", "wings and feathers", "fly"),
    ("dog", "four legs and a tail", "run and fetch"),
    ("cat", "sharp claws and soft fur", "climb and pounce"),
    ("frog", "long legs and smooth skin", "jump and swim"),
    ("snake", "no legs and scales", "slither"),
    ("elephant", "a long trunk and big ears", "carry water with its trunk"),
    ("rabbit", "long ears and a fluffy tail", "hop"),
    ("monkey", "long arms and a tail", "climb trees"),
    ("turtle", "a hard shell", "hide inside its shell"),
    ("spider", "eight legs", "spin webs"),
    ("kangaroo", "strong back legs and a pouch", "jump very far"),
    ("eagle", "sharp eyes and big wings", "soar high in the sky"),
    ("cheetah", "long legs and spots", "run very fast"),
    ("octopus", "eight arms", "squeeze into small spaces"),
]

for animal, feat, can_do in features:
    a = a_an(animal)
    add(f"{a} has {feat}", "fact", 1, "animals")
    add(f"{a} can {can_do}", "fact", 1, "animals")

# --- plants ---
plants = ["flower", "tree", "grass", "bush", "vine", "fern", "cactus", "moss"]
fruits = ["apple", "banana", "orange", "grape", "strawberry", "cherry",
          "peach", "pear", "mango", "watermelon", "lemon", "blueberry"]
vegetables = ["carrot", "potato", "tomato", "broccoli", "pea", "corn",
              "lettuce", "onion", "pepper", "bean", "cucumber", "spinach"]

add("plants grow from seeds", "fact", 1, "biology")
add("plants need water and sunlight to grow", "fact", 1, "biology")
add("plants have roots under the ground", "fact", 2, "biology")
add("roots take in water from the soil", "fact", 2, "biology")
add("leaves use sunlight to make food", "fact", 2, "biology")
add("flowers make seeds", "fact", 2, "biology")
add("some plants grow fruit", "fact", 1, "biology")
add("trees are the biggest plants", "fact", 1, "biology")
add("grass is a very common plant", "fact", 1, "biology")
add("a seed needs water and warmth to sprout", "fact", 2, "biology")
add("bees help flowers by spreading pollen", "fact", 2, "biology")
add("what do plants need to grow", "qa", 1, "biology",
    answer="water and sunlight")
add("where are the roots of a plant", "qa", 2, "biology",
    answer="under the ground")
add("what do leaves use to make food", "qa", 2, "biology",
    answer="sunlight")
add("what grows from a seed", "qa", 1, "biology", answer="a plant")

for f in fruits:
    a = a_an(f)
    add(f"{a} is a fruit", "fact", 1, "biology")
    add(f"{a} grows on a plant", "fact", 1, "biology")

for v in vegetables:
    a = a_an(v)
    add(f"{a} is a vegetable", "fact", 1, "biology")
    add(f"people eat {v}s", "fact", 1, "biology")

# tree types
trees = ["oak", "pine", "maple", "birch", "palm", "willow", "cherry",
         "apple", "cedar", "elm"]
for t in trees:
    add(f"{a_an(t)} tree is a kind of tree", "fact", 2, "biology")

# --- body ---
body_parts = [
    ("eyes", "see"), ("ears", "hear"), ("nose", "smell"),
    ("tongue", "taste"), ("skin", "feel"),
    ("hands", "grab things"), ("feet", "walk"),
    ("legs", "run and jump"), ("arms", "carry things"),
    ("heart", "pump blood"), ("lungs", "breathe air"),
    ("brain", "think"), ("stomach", "digest food"),
    ("teeth", "chew food"), ("bones", "hold up your body"),
    ("muscles", "move your body"),
]

for part, func in body_parts:
    add(f"your {part} help you {func}", "fact", 1, "biology")
    add(f"what do your {part} do", "qa", 2, "biology", answer=func)

senses = [("see", "eyes"), ("hear", "ears"), ("smell", "nose"),
           ("taste", "tongue"), ("touch", "skin")]
for sense, organ in senses:
    add(f"you {sense} with your {organ}", "fact", 1, "biology")
    add(f"what do you use to {sense}", "qa", 1, "biology", answer=f"your {organ}")

add("your body needs food and water to stay healthy", "fact", 1, "biology")
add("sleep helps your body rest and grow", "fact", 1, "biology")
add("exercise makes your body strong", "fact", 1, "biology")
add("blood carries food and air to all parts of your body", "fact", 2, "biology")
add("your skin protects the inside of your body", "fact", 2, "biology")

# --- life cycle ---
life_stages = [
    ("person", ["baby", "child", "teenager", "adult", "old person"]),
    ("butterfly", ["egg", "caterpillar", "cocoon", "butterfly"]),
    ("frog", ["egg", "tadpole", "froglet", "frog"]),
    ("chicken", ["egg", "chick", "hen or rooster"]),
    ("tree", ["seed", "sprout", "sapling", "tree"]),
    ("flower", ["seed", "sprout", "bud", "flower"]),
]

for creature, stages in life_stages:
    chain = " then ".join(stages)
    add(f"a {creature} goes through stages: {chain}", "fact", 2, "biology")
    add(f"what are the stages of a {creature}", "qa", 3, "biology",
        answer=chain)

add("all living things are born and grow and die", "fact", 2, "biology")
add("babies grow into children", "fact", 1, "biology")
add("children grow into adults", "fact", 1, "biology")
add("old people have lived a long time", "fact", 1, "biology")

# --- food chain ---
chains = [
    ("grass", "rabbit", "fox"),
    ("seeds", "mouse", "owl"),
    ("plants", "deer", "wolf"),
    ("algae", "small fish", "big fish"),
    ("leaves", "caterpillar", "bird"),
    ("flowers", "bee", "bear"),
    ("grass", "zebra", "lion"),
    ("plankton", "shrimp", "whale"),
    ("fruit", "monkey", "eagle"),
    ("grass", "cow", "human"),
]

for plant, herb, pred in chains:
    add(f"{herb}s eat {plant}", "fact", 2, "biology")
    add(f"{pred}s eat {herb}s", "fact", 2, "biology")
    add(f"the food chain goes {plant} then {herb} then {pred}", "fact", 3, "biology")
    add(f"what does {a_an(herb)} eat", "qa", 2, "biology", answer=plant)
    add(f"what eats {herb}s", "qa", 2, "biology", answer=f"{pred}s")

add("plants get energy from the sun", "fact", 2, "biology")
add("animals get energy from food", "fact", 2, "biology")
add("herbivores eat only plants", "fact", 2, "biology")
add("carnivores eat only meat", "fact", 2, "biology")
add("omnivores eat both plants and meat", "fact", 2, "biology")
add("what is an herbivore", "qa", 3, "biology",
    answer="an animal that eats only plants")
add("what is a carnivore", "qa", 3, "biology",
    answer="an animal that eats only meat")

# =====================================================================
# 3. SOCIAL / BEHAVIORAL  (~1,500)
# =====================================================================

# --- emotions ---
emotions_actions = [
    ("happy", "smile and laugh"),
    ("sad", "cry"),
    ("angry", "shout or frown"),
    ("scared", "run away or hide"),
    ("surprised", "gasp"),
    ("excited", "jump up and down"),
    ("tired", "yawn and want to sleep"),
    ("bored", "sigh and look around"),
    ("proud", "stand tall and smile"),
    ("shy", "hide behind someone"),
    ("lonely", "feel sad and want a friend"),
    ("nervous", "feel butterflies in your tummy"),
    ("jealous", "feel upset about what others have"),
    ("grateful", "say thank you"),
    ("embarrassed", "blush and look away"),
    ("confused", "scratch your head"),
    ("curious", "ask lots of questions"),
    ("brave", "face scary things"),
    ("calm", "breathe slowly"),
    ("frustrated", "stomp your feet"),
]

for emotion, action in emotions_actions:
    add(f"when you feel {emotion} you {action}", "social", 1, "social")
    add(f"what do you do when you feel {emotion}", "qa", 2, "social",
        answer=action)

emotion_causes = [
    ("happy", "something good happens"),
    ("sad", "you lose something you love"),
    ("angry", "something is not fair"),
    ("scared", "something is dangerous"),
    ("surprised", "something unexpected happens"),
    ("excited", "something fun is about to happen"),
    ("tired", "you have been awake a long time"),
    ("proud", "you do something well"),
    ("lonely", "you are all alone"),
    ("nervous", "you try something new"),
    ("grateful", "someone helps you"),
    ("frustrated", "you can not do something"),
]

for emotion, cause in emotion_causes:
    add(f"you feel {emotion} when {cause}", "social", 2, "social")

# --- manners ---
manners = [
    "say please when you ask for something",
    "say thank you when someone helps you",
    "say sorry when you make a mistake",
    "share your toys with friends",
    "take turns when playing a game",
    "listen when someone is talking",
    "do not interrupt when others speak",
    "cover your mouth when you cough",
    "wash your hands before eating",
    "say excuse me when you bump into someone",
    "be kind to everyone",
    "do not hit or push others",
    "help people who need help",
    "wait in line patiently",
    "use a quiet voice inside",
    "knock before entering a room",
    "hold the door for the next person",
    "look people in the eyes when talking",
    "clean up after yourself",
    "respect other people's things",
]

for m in manners:
    add(f"it is good to {m}", "social", 1, "social")

add("what do you say when someone gives you something", "qa", 1, "social",
    answer="thank you")
add("what do you say when you want something", "qa", 1, "social",
    answer="please")
add("what do you say when you make a mistake", "qa", 1, "social",
    answer="sorry")
add("why should you share", "qa", 2, "social", answer="because it is kind")
add("why should you take turns", "qa", 2, "social",
    answer="so everyone gets a chance")

# --- family ---
family_members = [
    ("mother", "your mom who takes care of you"),
    ("father", "your dad who takes care of you"),
    ("sister", "a girl in your family"),
    ("brother", "a boy in your family"),
    ("grandmother", "your parent's mother"),
    ("grandfather", "your parent's father"),
    ("aunt", "your parent's sister"),
    ("uncle", "your parent's brother"),
    ("cousin", "your aunt or uncle's child"),
    ("baby", "the youngest in a family"),
]

for member, desc in family_members:
    a = a_an(member)
    add(f"{a} is {desc}", "social", 1, "social")
    add(f"what is {a}", "qa", 2, "social", answer=desc)

add("a family lives together and loves each other", "social", 1, "social")
add("parents take care of their children", "social", 1, "social")
add("grandparents are older and very wise", "social", 1, "social")
add("siblings are brothers and sisters", "social", 1, "social")
add("families can be big or small", "social", 1, "social")
add("some families have pets too", "social", 1, "social")

# --- jobs ---
jobs = [
    ("doctor", "heals sick people"),
    ("teacher", "teaches children"),
    ("farmer", "grows food"),
    ("firefighter", "puts out fires"),
    ("police officer", "keeps people safe"),
    ("chef", "cooks food"),
    ("pilot", "flies planes"),
    ("driver", "drives a bus or truck"),
    ("nurse", "helps doctors and cares for patients"),
    ("dentist", "takes care of teeth"),
    ("vet", "takes care of animals"),
    ("builder", "builds houses"),
    ("artist", "makes art and paintings"),
    ("musician", "plays music"),
    ("baker", "bakes bread and cakes"),
    ("scientist", "learns about the world"),
    ("astronaut", "goes to space"),
    ("librarian", "helps people find books"),
    ("gardener", "takes care of plants"),
    ("mail carrier", "brings letters to your house"),
    ("mechanic", "fixes cars"),
    ("plumber", "fixes pipes and water"),
    ("electrician", "fixes lights and wires"),
    ("writer", "writes stories and books"),
]

for job, desc in jobs:
    a = a_an(job)
    add(f"{a} {desc}", "social", 1, "social")
    add(f"what does {a} do", "qa", 2, "social", answer=desc)

# --- rules and safety ---
rules = [
    ("stop at a red light", "so you do not get hit by a car"),
    ("look both ways before crossing the street", "so you stay safe"),
    ("wear a seatbelt in a car", "to protect you in a crash"),
    ("do not talk to strangers", "to stay safe"),
    ("do not play with fire", "because fire can burn you"),
    ("do not run with scissors", "because you might get hurt"),
    ("hold an adult's hand near the road", "to stay safe"),
    ("do not eat things from the floor", "because they might be dirty"),
    ("wear a helmet when riding a bike", "to protect your head"),
    ("tell an adult if you feel unsafe", "so they can help you"),
    ("stay close to your parents in public", "so you do not get lost"),
    ("do not touch hot things on the stove", "because they burn"),
    ("always wash your hands after the bathroom", "to get rid of germs"),
    ("drink water every day", "to stay healthy"),
    ("eat fruits and vegetables", "because they are good for you"),
]

for rule, reason in rules:
    add(f"you should {rule}", "social", 1, "social")
    add(f"you should {rule} {reason}", "social", 2, "social")
    add(f"why should you {rule}", "qa", 2, "social", answer=reason)

# --- feelings in context ---
feeling_situations = [
    ("you get a birthday present", "happy"),
    ("your pet is lost", "sad"),
    ("someone takes your toy", "angry"),
    ("you hear a loud noise at night", "scared"),
    ("you win a race", "proud"),
    ("you start at a new school", "nervous"),
    ("a friend shares candy with you", "grateful"),
    ("you cannot find your shoe", "frustrated"),
    ("you see a magic trick", "surprised"),
    ("you are about to go to the park", "excited"),
    ("your friend says something nice about you", "happy"),
    ("you fall and scrape your knee", "hurt and sad"),
    ("someone laughs at you", "embarrassed"),
    ("you are home alone", "lonely"),
    ("you finish a hard puzzle", "proud"),
]

for situation, feeling in feeling_situations:
    add(f"when {situation} you feel {feeling}", "social", 2, "social")
    add(f"how do you feel when {situation}", "qa", 2, "social",
        answer=feeling)

# --- friendship ---
friendship_facts = [
    "friends play together and share",
    "a good friend listens to you",
    "friends help each other",
    "it is okay to disagree with friends",
    "you can make friends by being kind",
    "friends make you feel happy",
    "good friends are honest",
    "you can have many friends or just a few",
    "it is important to be a good friend",
    "friends take turns choosing games",
    "if a friend is sad you can comfort them",
    "saying nice things makes friends happy",
]

for f in friendship_facts:
    add(f, "social", 1, "social")

# =====================================================================
# 4. TIME AND SEQUENCE  (~1,500)
# =====================================================================

# --- day ---
add("the day starts in the morning", "time", 1, "time")
add("morning comes before afternoon", "time", 1, "time")
add("afternoon comes before evening", "time", 1, "time")
add("evening comes before night", "time", 1, "time")
add("we sleep at night", "time", 1, "time")
add("we wake up in the morning", "time", 1, "time")
add("the sun rises in the morning", "time", 1, "time")
add("the sun sets in the evening", "time", 1, "time")
add("noon is the middle of the day", "time", 1, "time")
add("midnight is the middle of the night", "time", 2, "time")
add("what comes after morning", "qa", 1, "time", answer="afternoon")
add("what comes after afternoon", "qa", 1, "time", answer="evening")
add("what comes after evening", "qa", 1, "time", answer="night")
add("when does the sun rise", "qa", 1, "time", answer="in the morning")
add("when do we sleep", "qa", 1, "time", answer="at night")

# morning/afternoon/evening/night activities
morning_acts = ["wake up", "brush your teeth", "eat breakfast", "get dressed",
                "go to school"]
afternoon_acts = ["eat lunch", "play outside", "do homework", "have a snack",
                  "read a book"]
evening_acts = ["eat dinner", "take a bath", "watch a show", "talk to family",
                "brush your teeth"]
night_acts = ["put on pajamas", "read a story", "say goodnight", "go to sleep",
              "dream"]

for act in morning_acts:
    add(f"in the morning you {act}", "time", 1, "time")
for act in afternoon_acts:
    add(f"in the afternoon you {act}", "time", 1, "time")
for act in evening_acts:
    add(f"in the evening you {act}", "time", 1, "time")
for act in night_acts:
    add(f"at night you {act}", "time", 1, "time")

# --- seasons ---
seasons_info = [
    ("spring", "flowers bloom and baby animals are born", "warm"),
    ("summer", "it is hot and the days are long", "hot"),
    ("fall", "leaves change color and fall from trees", "cool"),
    ("winter", "it is cold and sometimes it snows", "cold"),
]

for season, desc, temp in seasons_info:
    add(f"in {season} {desc}", "time", 1, "time")
    add(f"{season} is {temp}", "time", 1, "time")
    add(f"what happens in {season}", "qa", 2, "time", answer=desc)

add("spring comes after winter", "time", 1, "time")
add("summer comes after spring", "time", 1, "time")
add("fall comes after summer", "time", 1, "time")
add("winter comes after fall", "time", 1, "time")
add("after winter it is spring again", "time", 1, "time")
add("there are four seasons in a year", "time", 1, "time")
add("what season comes after winter", "qa", 1, "time", answer="spring")
add("what season comes after spring", "qa", 1, "time", answer="summer")
add("what season comes after summer", "qa", 1, "time", answer="fall")
add("what season comes after fall", "qa", 1, "time", answer="winter")
add("how many seasons are there", "qa", 1, "time", answer="four")

# season activities
season_activities = [
    ("spring", ["plant flowers", "fly kites", "watch birds", "splash in puddles",
                "see baby animals"]),
    ("summer", ["swim in the pool", "eat ice cream", "go to the beach",
                "play in the sun", "catch fireflies"]),
    ("fall", ["jump in leaf piles", "pick apples", "wear a sweater",
              "carve pumpkins", "go back to school"]),
    ("winter", ["build a snowman", "drink hot cocoa", "wear a warm coat",
                "go sledding", "sit by the fire"]),
]

for season, acts in season_activities:
    for act in acts:
        add(f"in {season} you can {act}", "time", 1, "time")

# --- growth ---
growth_sequences = [
    ("a person", ["baby", "toddler", "child", "teenager", "adult"]),
    ("a dog", ["puppy", "young dog", "adult dog", "old dog"]),
    ("a cat", ["kitten", "young cat", "adult cat", "old cat"]),
    ("a plant", ["seed", "sprout", "stem with leaves", "tall plant"]),
    ("a tree", ["seed", "small sprout", "sapling", "big tree"]),
]

for thing, stages in growth_sequences:
    for i in range(len(stages) - 1):
        add(f"{thing} grows from {a_an(stages[i])} to {a_an(stages[i+1])}",
            "time", 2, "time")
    add(f"what does {thing} start as", "qa", 2, "time", answer=a_an(stages[0]))

# --- meals ---
meals = [
    ("breakfast", "morning", ["cereal", "toast", "eggs", "pancakes", "fruit",
                               "milk", "oatmeal", "yogurt"]),
    ("lunch", "noon", ["sandwich", "soup", "salad", "pizza", "pasta",
                        "rice", "chicken", "fruit"]),
    ("dinner", "evening", ["rice", "chicken", "fish", "vegetables", "pasta",
                            "steak", "bread", "soup"]),
    ("snack", "between meals", ["apple", "crackers", "cheese", "nuts",
                                 "banana", "cookies", "carrot sticks"]),
]

for meal, when, foods in meals:
    add(f"we eat {meal} in the {when}", "time", 1, "time")
    for food in foods:
        add(f"{food} is good for {meal}", "fact", 1, "time")
    add(f"when do we eat {meal}", "qa", 1, "time", answer=f"in the {when}")

# --- days of the week ---
days = ["monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday"]
for i in range(len(days)):
    nxt = days[(i + 1) % len(days)]
    add(f"after {days[i]} comes {nxt}", "time", 1, "time")
    add(f"what day comes after {days[i]}", "qa", 1, "time", answer=nxt)

add("there are seven days in a week", "time", 1, "time")
add("monday through friday are weekdays", "time", 2, "time")
add("saturday and sunday are the weekend", "time", 2, "time")
add("many kids go to school on weekdays", "time", 1, "time")
add("many families rest on the weekend", "time", 1, "time")
add("how many days are in a week", "qa", 1, "time", answer="seven")

# --- months ---
months = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]
for i in range(len(months)):
    nxt = months[(i + 1) % len(months)]
    add(f"after {months[i]} comes {nxt}", "time", 2, "time")
    add(f"what month comes after {months[i]}", "qa", 2, "time", answer=nxt)

add("there are twelve months in a year", "time", 1, "time")
add("how many months are in a year", "qa", 1, "time", answer="twelve")

month_seasons = [
    ("december", "winter"), ("january", "winter"), ("february", "winter"),
    ("march", "spring"), ("april", "spring"), ("may", "spring"),
    ("june", "summer"), ("july", "summer"), ("august", "summer"),
    ("september", "fall"), ("october", "fall"), ("november", "fall"),
]
for m, s in month_seasons:
    add(f"{m} is in {s}", "time", 2, "time")

# --- time concepts ---
time_concepts = [
    "yesterday is the day before today",
    "tomorrow is the day after today",
    "a minute is shorter than an hour",
    "an hour is shorter than a day",
    "a day is shorter than a week",
    "a week is shorter than a month",
    "a month is shorter than a year",
    "one year has three hundred and sixty five days",
    "a clock tells you what time it is",
    "the short hand on a clock shows the hour",
    "the long hand on a clock shows the minutes",
]

for c in time_concepts:
    add(c, "time", 2, "time")

# --- sequences ---
sequences = [
    ("making a sandwich", ["get bread", "put filling on bread",
                            "put another bread on top", "eat it"]),
    ("brushing your teeth", ["put toothpaste on the brush", "brush your teeth",
                              "spit out the paste", "rinse your mouth"]),
    ("planting a seed", ["dig a hole", "put the seed in", "cover it with soil",
                          "water it"]),
    ("getting dressed", ["put on underwear", "put on a shirt",
                          "put on pants", "put on socks and shoes"]),
    ("washing your hands", ["turn on the water", "use soap",
                             "rub your hands together", "rinse and dry"]),
    ("baking a cake", ["mix the ingredients", "pour into a pan",
                        "put in the oven", "let it cool and eat"]),
    ("sending a letter", ["write the letter", "put it in an envelope",
                           "put a stamp on it", "put it in the mailbox"]),
    ("going to bed", ["brush your teeth", "put on pajamas",
                       "get in bed", "close your eyes"]),
]

for task, steps in sequences:
    chain = " then ".join(steps)
    add(f"to do {task}: {chain}", "sequence", 2, "time")
    add(f"what is the first step of {task}", "qa", 2, "time",
        answer=steps[0])
    add(f"what is the last step of {task}", "qa", 2, "time",
        answer=steps[-1])

# =====================================================================
# 5. CAUSE AND EFFECT  (~2,000)
# =====================================================================

cause_effects = [
    ("you drop a glass", "it breaks"),
    ("you eat too much", "you feel sick"),
    ("it rains", "the ground gets wet"),
    ("you study", "you learn"),
    ("you exercise", "you get strong"),
    ("you do not sleep", "you feel tired"),
    ("you water a plant", "it grows"),
    ("you do not water a plant", "it dies"),
    ("you mix red and blue paint", "you get purple"),
    ("you mix red and yellow paint", "you get orange"),
    ("you mix blue and yellow paint", "you get green"),
    ("you mix black and white paint", "you get gray"),
    ("you turn off the light", "the room gets dark"),
    ("you open the window", "fresh air comes in"),
    ("you close the door", "the noise gets quieter"),
    ("you add sugar to water", "the water tastes sweet"),
    ("you add salt to water", "the water tastes salty"),
    ("you put ice in a warm drink", "the drink gets cold"),
    ("you leave food out too long", "it goes bad"),
    ("you put food in the fridge", "it stays fresh"),
    ("you throw a ball hard", "it goes far"),
    ("you throw a ball softly", "it stays close"),
    ("you push a swing", "it moves forward"),
    ("you pull a wagon", "it comes toward you"),
    ("you blow on hot soup", "it cools down"),
    ("you sit in the sun", "you get warm"),
    ("you stand in the rain", "you get wet"),
    ("you run fast", "you breathe hard"),
    ("you eat breakfast", "you have energy"),
    ("you skip breakfast", "you feel hungry"),
    ("you touch a hot stove", "you burn your hand"),
    ("you touch ice", "your hand gets cold"),
    ("you cut paper with scissors", "the paper splits"),
    ("you blow up a balloon", "it gets bigger"),
    ("you pop a balloon", "it makes a loud bang"),
    ("you shake a bottle of soda", "it fizzes when opened"),
    ("a tire has a hole", "the air leaks out"),
    ("you leave the door open in winter", "the house gets cold"),
    ("you wear a coat in winter", "you stay warm"),
    ("you do not wear a coat in winter", "you get cold"),
    ("you practice a lot", "you get better"),
    ("you are kind to others", "they are kind to you"),
    ("you are mean to others", "they feel sad"),
    ("you smile at someone", "they usually smile back"),
    ("you say something funny", "people laugh"),
    ("you tell the truth", "people trust you"),
    ("you tell a lie", "people stop trusting you"),
    ("it is very cold outside", "water freezes into ice"),
    ("it is very hot outside", "ice melts into water"),
    ("the wind blows hard", "trees sway back and forth"),
    ("a seed gets water and sun", "it starts to grow"),
    ("you plug in a lamp", "it can make light"),
    ("you flip a switch", "the light turns on or off"),
    ("you press the gas pedal", "the car goes faster"),
    ("you press the brake", "the car slows down"),
    ("you turn the steering wheel", "the car turns"),
    ("a dog is hungry", "it looks for food"),
    ("a cat is scared", "it hisses and runs"),
    ("a bird builds a nest", "it can lay eggs"),
    ("you leave crumbs on the floor", "ants come"),
    ("you clap your hands", "it makes a sound"),
    ("you stomp your feet", "it makes a thud"),
    ("you whistle", "it makes a high sound"),
    ("you yell into a canyon", "you hear an echo"),
    ("you put on sunscreen", "you do not get sunburned"),
    ("you do not put on sunscreen", "you might get sunburned"),
    ("you read books", "you learn new words"),
    ("you eat vegetables", "you stay healthy"),
    ("you eat too much candy", "your teeth can get cavities"),
    ("you brush your teeth", "they stay clean"),
    ("you wash your hands", "germs go away"),
    ("you drink water", "you are not thirsty"),
    ("a balloon is filled with helium", "it floats up"),
    ("you let go of a helium balloon", "it flies into the sky"),
    ("the alarm clock rings", "you wake up"),
    ("the phone rings", "someone is calling"),
    ("the doorbell rings", "someone is at the door"),
    ("you step on a twig", "it snaps"),
    ("you bend a stick too far", "it breaks"),
    ("you rub your hands together fast", "they get warm"),
    ("you put a magnet near metal", "it sticks"),
    ("you shake salt on food", "it tastes saltier"),
    ("you add lemon to water", "it tastes sour"),
    ("you mix flour and water", "you get dough"),
    ("you heat dough in the oven", "it becomes bread"),
    ("you add yeast to dough", "it rises and puffs up"),
    ("you forget to set the alarm", "you might oversleep"),
    ("the clouds block the sun", "it gets cooler"),
    ("the sun comes out", "it gets warmer"),
    ("you paint on paper", "the paper becomes colorful"),
    ("you erase a pencil mark", "the mark goes away"),
    ("you sharpen a pencil", "it writes better"),
    ("you charge a battery", "it gets full of power"),
    ("the battery runs out", "the device stops working"),
    ("you plant a seed in good soil", "it grows into a plant"),
    ("a river flows fast", "it can move rocks"),
    ("the wind blows sand", "it piles up into dunes"),
    ("you put clothes in the dryer", "they get dry"),
    ("you hang wet clothes in the sun", "they dry"),
    ("you spill water on the floor", "the floor gets slippery"),
    ("you oil a squeaky door", "it stops squeaking"),
    ("you leave metal in the rain", "it rusts"),
    ("you cover a candle", "the flame goes out"),
    ("you blow out a candle", "the flame goes out"),
    ("you open an umbrella", "the rain does not hit you"),
]

for cause, effect in cause_effects:
    # Statement form
    add(f"if {cause} {effect}", "cause_effect", 2, "commonsense")
    # QA form
    add(f"what happens if {cause}", "qa", 2, "commonsense", answer=effect)

# Additional phrased as "X makes Y happen" / "because" forms
because_pairs = [
    ("we wear coats", "it is cold outside"),
    ("we use umbrellas", "it is raining"),
    ("we eat food", "we are hungry"),
    ("we drink water", "we are thirsty"),
    ("we sleep", "our body needs rest"),
    ("we go to school", "we want to learn"),
    ("flowers bloom in spring", "the weather gets warm"),
    ("leaves fall in autumn", "the days get shorter and colder"),
    ("birds fly south in winter", "it is too cold"),
    ("bears hibernate", "there is no food in winter"),
    ("we sweat", "our body is hot"),
    ("we shiver", "our body is cold"),
    ("we blink", "our eyes need to stay moist"),
    ("we yawn", "we are tired"),
    ("we sneeze", "something tickles our nose"),
    ("we cough", "something is in our throat"),
    ("dogs pant", "they are hot"),
    ("cats purr", "they are happy"),
    ("babies cry", "they need something"),
    ("the sun is bright", "it makes a lot of light"),
]

for effect, cause in because_pairs:
    add(f"{effect} because {cause}", "cause_effect", 2, "commonsense")
    add(f"why do {effect}", "qa", 2, "commonsense", answer=f"because {cause}")

# =====================================================================
# 6. SPATIAL / RELATIONAL  (~1,000)
# =====================================================================

# --- position ---
spatial_facts = [
    ("the sky", "above", "the ground"),
    ("the floor", "below", "the ceiling"),
    ("roots", "under", "the tree"),
    ("a bird", "above", "the ground"),
    ("a fish", "under", "the water"),
    ("the sun", "above", "the clouds"),
    ("the basement", "below", "the house"),
    ("the roof", "on top of", "the house"),
    ("your feet", "below", "your head"),
    ("your hat", "on top of", "your head"),
    ("a rug", "on", "the floor"),
    ("a picture", "on", "the wall"),
    ("stars", "above", "us at night"),
    ("the moon", "above", "the earth"),
    ("submarines", "under", "the ocean"),
    ("worms", "under", "the ground"),
    ("clouds", "above", "the mountains"),
    ("snow", "on top of", "the mountain"),
    ("a nest", "in", "a tree"),
    ("a bed", "in", "a bedroom"),
    ("a stove", "in", "a kitchen"),
    ("a bathtub", "in", "a bathroom"),
    ("books", "on", "a shelf"),
    ("clothes", "in", "a closet"),
    ("food", "in", "the fridge"),
    ("dishes", "in", "a cupboard"),
    ("a car", "in", "a garage"),
    ("a boat", "on", "the water"),
    ("a plane", "in", "the sky"),
    ("a train", "on", "the tracks"),
]

for thing, prep, place in spatial_facts:
    add(f"{thing} is {prep} {place}", "spatial", 1, "spatial")
    add(f"where is {thing}", "qa", 2, "spatial", answer=f"{prep} {place}")

# --- inside/outside ---
inside_things = [
    ("a house", ["rooms", "furniture", "people"]),
    ("a school", ["classrooms", "teachers", "students"]),
    ("a car", ["seats", "a steering wheel", "mirrors"]),
    ("a body", ["bones", "blood", "organs"]),
    ("an egg", ["a yolk", "egg white"]),
    ("a fruit", ["seeds", "juice"]),
    ("a box", ["things you put inside"]),
    ("a backpack", ["books", "pencils", "a lunchbox"]),
]

for container, contents in inside_things:
    for c in contents:
        add(f"inside {container} there are {c}", "spatial", 1, "spatial")

# --- relative / comparative ---
comparisons = [
    ("an elephant", "a mouse", "bigger"),
    ("a mountain", "a hill", "taller"),
    ("the ocean", "a lake", "larger"),
    ("the sun", "the moon", "brighter"),
    ("a car", "a person", "faster"),
    ("a cheetah", "a turtle", "faster"),
    ("a whale", "a goldfish", "bigger"),
    ("a tree", "a flower", "taller"),
    ("a river", "a puddle", "longer"),
    ("a truck", "a bike", "heavier"),
    ("a whisper", "a shout", "quieter"),
    ("night", "day", "darker"),
    ("summer", "winter", "hotter"),
    ("a feather", "a rock", "lighter"),
    ("honey", "water", "thicker"),
    ("a snail", "a rabbit", "slower"),
    ("ice", "fire", "colder"),
    ("a skyscraper", "a house", "taller"),
    ("the earth", "the moon", "bigger"),
    ("an ant", "a dog", "smaller"),
]

for a, b, comp in comparisons:
    add(f"{a} is {comp} than {b}", "spatial", 1, "spatial")
    add(f"which is {comp} {a} or {b}", "qa", 2, "spatial", answer=a)
    # reverse
    opposite = {
        "bigger": "smaller", "taller": "shorter", "larger": "smaller",
        "brighter": "dimmer", "faster": "slower", "lighter": "heavier",
        "quieter": "louder", "darker": "lighter", "hotter": "colder",
        "thicker": "thinner", "slower": "faster", "colder": "hotter",
        "heavier": "lighter",
    }
    if comp in opposite:
        add(f"{b} is {opposite[comp]} than {a}", "spatial", 2, "spatial")

# --- direction ---
directions = [
    ("the sun rises in the east", "east"),
    ("the sun sets in the west", "west"),
    ("north is at the top of a map", "north"),
    ("south is at the bottom of a map", "south"),
    ("a compass shows you which way is north", "north"),
    ("up is the opposite of down", "up"),
    ("left is the opposite of right", "left"),
    ("front is the opposite of back", "front"),
    ("near is the opposite of far", "near"),
]

for fact, _ in directions:
    add(fact, "spatial", 2, "spatial")

# --- rooms in a house ---
rooms = [
    ("kitchen", "cook food"),
    ("bedroom", "sleep"),
    ("bathroom", "take a bath or shower"),
    ("living room", "watch TV and relax"),
    ("dining room", "eat meals"),
    ("garage", "park the car"),
    ("garden", "grow plants outside"),
    ("attic", "store old things"),
    ("basement", "store things underground"),
]

for room, purpose in rooms:
    add(f"the {room} is where you {purpose}", "spatial", 1, "spatial")
    add(f"what do you do in the {room}", "qa", 1, "spatial", answer=purpose)

# --- places in a town ---
places = [
    ("school", "learn things"),
    ("hospital", "get better when you are sick"),
    ("library", "borrow and read books"),
    ("park", "play outside"),
    ("store", "buy things"),
    ("restaurant", "eat food someone else cooked"),
    ("fire station", "where firefighters work"),
    ("police station", "where police officers work"),
    ("zoo", "see many different animals"),
    ("farm", "where animals and crops are raised"),
    ("beach", "play in the sand and swim"),
    ("airport", "where planes take off and land"),
    ("bank", "where people keep money"),
    ("post office", "where you send letters"),
    ("church", "where some people pray"),
    ("bakery", "where bread and cakes are made"),
]

for place, purpose in places:
    a = a_an(place)
    add(f"{a} is where you {purpose}", "spatial", 1, "spatial")
    add(f"what do you do at {a}", "qa", 2, "spatial", answer=purpose)

# =====================================================================
# BONUS: More varied entries to reach ~10,000
# =====================================================================

# --- colors ---
colors = ["red", "blue", "green", "yellow", "orange", "purple",
          "pink", "brown", "black", "white", "gray"]
color_things = [
    ("red", ["apple", "fire truck", "strawberry", "tomato", "rose"]),
    ("blue", ["sky", "ocean", "blueberry", "jeans"]),
    ("green", ["grass", "frog", "leaf", "pea", "turtle"]),
    ("yellow", ["banana", "sun", "lemon", "duckling", "sunflower"]),
    ("orange", ["orange", "carrot", "pumpkin", "goldfish"]),
    ("purple", ["grape", "plum", "eggplant"]),
    ("pink", ["flamingo", "pig", "bubblegum"]),
    ("brown", ["chocolate", "bear", "mud", "wood", "coconut"]),
    ("black", ["night sky", "crow", "bat"]),
    ("white", ["snow", "cloud", "milk", "polar bear", "egg"]),
    ("gray", ["elephant", "mouse", "rain cloud", "rock"]),
]

for color, things in color_things:
    for thing in things:
        a = a_an(thing)
        add(f"{a} is {color}", "fact", 1, "commonsense")
    add(f"name something {color}", "qa", 1, "commonsense",
        answer=random.choice(things))

# --- numbers and counting ---
for n in range(1, 21):
    word = ["one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen", "twenty"][n - 1]
    add(f"the number {n} is called {word}", "fact", 1, "commonsense")
    if n < 20:
        nxt = ["one", "two", "three", "four", "five", "six", "seven",
               "eight", "nine", "ten", "eleven", "twelve", "thirteen",
               "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
               "nineteen", "twenty"][n]
        add(f"after {word} comes {nxt}", "fact", 1, "commonsense")
        add(f"what comes after {word}", "qa", 1, "commonsense", answer=nxt)

# basic math
for a_num in range(1, 11):
    for b_num in range(1, 11):
        if a_num + b_num <= 20:
            add(f"{a_num} plus {b_num} is {a_num + b_num}", "fact", 2, "commonsense")
        if a_num > b_num:
            add(f"{a_num} minus {b_num} is {a_num - b_num}", "fact", 2, "commonsense")

# --- shapes ---
shapes = [
    ("circle", "round with no corners"),
    ("square", "has four equal sides and four corners"),
    ("triangle", "has three sides and three corners"),
    ("rectangle", "has four sides and four corners"),
    ("star", "has points sticking out"),
    ("heart", "shaped like love"),
    ("oval", "like a stretched circle"),
    ("diamond", "like a turned square"),
]

for shape, desc in shapes:
    a = a_an(shape)
    add(f"{a} is {desc}", "fact", 1, "commonsense")
    add(f"what does {a} look like", "qa", 1, "commonsense", answer=desc)

shape_things = [
    ("circle", ["wheel", "ball", "coin", "clock", "plate"]),
    ("square", ["dice", "tile", "box", "window"]),
    ("triangle", ["slice of pizza", "roof", "mountain peak"]),
    ("rectangle", ["door", "book", "phone", "TV screen"]),
    ("star", ["starfish"]),
]

for shape, things in shape_things:
    for thing in things:
        a = a_an(thing)
        add(f"{a} is shaped like {a_an(shape)}", "fact", 1, "commonsense")

# --- opposites ---
opposites = [
    ("big", "small"), ("hot", "cold"), ("fast", "slow"),
    ("tall", "short"), ("long", "short"), ("heavy", "light"),
    ("hard", "soft"), ("loud", "quiet"), ("dark", "light"),
    ("up", "down"), ("in", "out"), ("open", "closed"),
    ("wet", "dry"), ("happy", "sad"), ("full", "empty"),
    ("old", "young"), ("near", "far"), ("clean", "dirty"),
    ("smooth", "rough"), ("thick", "thin"), ("wide", "narrow"),
    ("strong", "weak"), ("rich", "poor"), ("good", "bad"),
    ("right", "wrong"), ("awake", "asleep"), ("day", "night"),
    ("summer", "winter"), ("start", "stop"), ("push", "pull"),
]

for w1, w2 in opposites:
    add(f"the opposite of {w1} is {w2}", "fact", 1, "commonsense")
    add(f"what is the opposite of {w1}", "qa", 1, "commonsense", answer=w2)
    add(f"what is the opposite of {w2}", "qa", 1, "commonsense", answer=w1)

# --- categories / grouping ---
category_groups = [
    ("animal", ["dog", "cat", "fish", "bird", "horse", "cow", "lion"]),
    ("fruit", ["apple", "banana", "orange", "grape", "mango", "cherry"]),
    ("vegetable", ["carrot", "potato", "broccoli", "pea", "corn"]),
    ("color", ["red", "blue", "green", "yellow", "purple"]),
    ("shape", ["circle", "square", "triangle", "rectangle"]),
    ("vehicle", ["car", "bus", "bike", "train", "plane", "boat"]),
    ("furniture", ["chair", "table", "bed", "sofa", "desk", "shelf"]),
    ("clothing", ["shirt", "pants", "shoes", "hat", "socks", "coat"]),
    ("tool", ["hammer", "saw", "screwdriver", "wrench", "drill"]),
    ("instrument", ["piano", "guitar", "drum", "violin", "flute"]),
    ("sport", ["soccer", "basketball", "swimming", "running", "tennis"]),
    ("drink", ["water", "milk", "juice", "tea"]),
    ("body part", ["arm", "leg", "head", "hand", "foot", "ear", "eye"]),
    ("weather", ["rain", "snow", "wind", "sunshine", "fog"]),
    ("room", ["kitchen", "bedroom", "bathroom", "living room"]),
]

for cat, items in category_groups:
    for item in items:
        a = a_an(item)
        add(f"{a} is {a_an(cat)}", "fact", 1, "commonsense")
    add(f"name some {cat}s", "qa", 2, "commonsense",
        answer=" and ".join(items[:3]))
    # odd-one-out
    odd_item = random.choice(
        [x for group_cat, group_items in category_groups
         for x in group_items if group_cat != cat][:10])
    trio = random.sample(items, min(2, len(items)))
    add(f"which one is not {a_an(cat)}: {trio[0]} or {odd_item}",
        "qa", 3, "commonsense", answer=odd_item)

# --- simple analogies ---
analogies = [
    ("a bird", "fly", "a fish", "swim"),
    ("a dog", "bark", "a cat", "meow"),
    ("the sun", "day", "the moon", "night"),
    ("a hand", "grab", "a foot", "kick"),
    ("eyes", "see", "ears", "hear"),
    ("a book", "read", "a song", "listen"),
    ("a knife", "cut", "a pen", "write"),
    ("a chair", "sit", "a bed", "sleep"),
    ("hot", "summer", "cold", "winter"),
    ("up", "sky", "down", "ground"),
    ("fast", "cheetah", "slow", "turtle"),
    ("big", "elephant", "small", "mouse"),
    ("sweet", "sugar", "sour", "lemon"),
]

for a_thing, a_prop, b_thing, b_prop in analogies:
    add(f"{a_thing} is to {a_prop} as {b_thing} is to {b_prop}",
        "analogy", 3, "commonsense")
    add(f"if {a_thing} can {a_prop} then {b_thing} can {b_prop}",
        "analogy", 2, "commonsense")

# --- everyday objects and uses ---
objects_uses = [
    ("a spoon", "eat soup"),
    ("a fork", "pick up food"),
    ("a knife", "cut things"),
    ("a cup", "drink from"),
    ("a plate", "put food on"),
    ("a key", "open a lock"),
    ("a pillow", "rest your head on"),
    ("a blanket", "keep you warm"),
    ("a brush", "brush your hair"),
    ("a towel", "dry yourself"),
    ("a broom", "sweep the floor"),
    ("scissors", "cut paper"),
    ("a pencil", "write and draw"),
    ("an eraser", "remove pencil marks"),
    ("a ruler", "measure things"),
    ("a clock", "tell the time"),
    ("a phone", "call and talk to people"),
    ("a book", "read stories"),
    ("a mirror", "see yourself"),
    ("a hammer", "hit nails"),
    ("a ladder", "climb up high"),
    ("an umbrella", "keep rain off you"),
    ("a map", "find your way"),
    ("a camera", "take pictures"),
    ("glasses", "help you see better"),
    ("a thermometer", "check the temperature"),
    ("a wheel", "make things roll"),
    ("a zipper", "close clothes and bags"),
    ("a button", "hold clothes together"),
    ("a bell", "make a ringing sound"),
]

for obj, use in objects_uses:
    add(f"{obj} is used to {use}", "fact", 1, "commonsense")
    add(f"what is {obj} used for", "qa", 1, "commonsense", answer=f"to {use}")

# --- vehicles and transport ---
vehicles = [
    ("car", "on roads", "drives"),
    ("bus", "on roads", "carries many people"),
    ("truck", "on roads", "carries heavy things"),
    ("bike", "on roads and paths", "you pedal"),
    ("train", "on tracks", "carries people and cargo"),
    ("plane", "in the sky", "flies fast"),
    ("boat", "on water", "floats"),
    ("ship", "on the ocean", "is very big"),
    ("helicopter", "in the sky", "hovers"),
    ("rocket", "into space", "goes very fast"),
    ("submarine", "under water", "dives deep"),
    ("ambulance", "on roads", "takes sick people to the hospital"),
    ("fire truck", "on roads", "carries firefighters"),
]

for v, where, what in vehicles:
    a = a_an(v)
    add(f"{a} goes {where}", "fact", 1, "commonsense")
    add(f"{a} {what}", "fact", 1, "commonsense")

# --- more cause-effect combos from templates ---
actions_on_objects = [
    ("water", "a flower", "it grows"),
    ("freeze", "water", "it becomes ice"),
    ("boil", "water", "it becomes steam"),
    ("paint", "a wall", "it changes color"),
    ("wash", "dirty clothes", "they become clean"),
    ("iron", "wrinkled clothes", "they become smooth"),
    ("sew", "torn cloth", "it gets fixed"),
    ("glue", "broken pieces", "they stick together"),
    ("sand", "rough wood", "it becomes smooth"),
    ("polish", "shoes", "they become shiny"),
]

for action, obj, result in actions_on_objects:
    add(f"if you {action} {obj} {result}", "cause_effect", 2, "commonsense")
    add(f"what happens when you {action} {obj}", "qa", 2, "commonsense",
        answer=result)

# --- "which sense" questions ---
sense_examples = [
    ("a rainbow", "see", "eyes"),
    ("music", "hear", "ears"),
    ("a flower's smell", "smell", "nose"),
    ("ice cream's flavor", "taste", "tongue"),
    ("a soft blanket", "feel", "skin"),
    ("a painting", "see", "eyes"),
    ("a bird singing", "hear", "ears"),
    ("cookies baking", "smell", "nose"),
    ("salty chips", "taste", "tongue"),
    ("hot sand", "feel", "feet"),
]

for thing, sense, organ in sense_examples:
    add(f"you {sense} {thing} with your {organ}", "fact", 1, "biology")
    add(f"which sense do you use for {thing}", "qa", 2, "biology",
        answer=sense)

# --- animal classification ---
animal_classes = [
    ("mammal", ["dog", "cat", "cow", "whale", "bat", "elephant", "human"]),
    ("bird", ["eagle", "penguin", "parrot", "owl", "chicken"]),
    ("fish", ["goldfish", "shark", "salmon", "clownfish"]),
    ("reptile", ["snake", "lizard", "crocodile", "turtle"]),
    ("insect", ["ant", "bee", "butterfly", "beetle", "fly"]),
    ("amphibian", ["frog", "toad", "salamander", "newt"]),
]

for cls, animals in animal_classes:
    for animal in animals:
        a = a_an(animal)
        add(f"{a} is {a_an(cls)}", "fact", 2, "biology")
    add(f"name some {cls}s", "qa", 2, "biology",
        answer=" and ".join(animals[:3]))

mammal_facts = [
    "mammals have warm blood",
    "mammals have fur or hair",
    "baby mammals drink milk from their mother",
    "birds have feathers and lay eggs",
    "fish have scales and breathe through gills",
    "reptiles have scales and are cold blooded",
    "insects have six legs",
    "amphibians can live in water and on land",
]
for f in mammal_facts:
    add(f, "fact", 2, "biology")

# --- earth and space (simple) ---
space_facts = [
    "the earth goes around the sun",
    "the moon goes around the earth",
    "the sun is a star",
    "the sun is very very far away",
    "the earth is round like a ball",
    "there are eight planets in our solar system",
    "stars are very far away suns",
    "the moon has no air",
    "astronauts go to space in rockets",
    "the earth has land and water",
    "most of the earth is covered by water",
    "the earth spins around once every day",
    "one trip around the sun takes one year",
    "the moon looks different each night",
    "a full moon is round and bright",
    "mars is called the red planet",
    "jupiter is the biggest planet",
    "saturn has rings around it",
]

for f in space_facts:
    add(f, "fact", 2, "science")

add("what shape is the earth", "qa", 2, "science", answer="round like a ball")
add("what does the earth go around", "qa", 2, "science", answer="the sun")
add("what goes around the earth", "qa", 2, "science", answer="the moon")
add("how many planets are there", "qa", 2, "science", answer="eight")
add("which planet has rings", "qa", 2, "science", answer="saturn")
add("which planet is the biggest", "qa", 2, "science", answer="jupiter")

# --- geography basics ---
geo_facts = [
    "the ocean is a huge body of salt water",
    "a river is water that flows to the sea",
    "a lake is water surrounded by land",
    "a mountain is very high land",
    "a valley is low land between mountains",
    "a desert is very dry and sandy",
    "a forest is full of trees",
    "an island is land surrounded by water",
    "a volcano can erupt with hot lava",
    "a waterfall is water falling down a cliff",
    "a cave is a hole inside a mountain or hill",
    "the north pole is very cold",
    "the south pole is very cold too",
    "the equator is the hottest part of the earth",
    "continents are very large pieces of land",
]

for f in geo_facts:
    add(f, "fact", 2, "commonsense")

add("what is a desert", "qa", 2, "commonsense",
    answer="a very dry and sandy place")
add("what is a forest", "qa", 2, "commonsense",
    answer="a place full of trees")
add("what is an island", "qa", 2, "commonsense",
    answer="land surrounded by water")
add("what is a volcano", "qa", 2, "commonsense",
    answer="a mountain that can erupt with hot lava")

# --- safety in nature ---
nature_safety = [
    "do not touch wild animals",
    "do not eat unknown berries or mushrooms",
    "stay away from deep water if you can not swim",
    "lightning is dangerous so go inside during a storm",
    "do not stand under a tree in a lightning storm",
    "wear sunscreen in the hot sun",
    "drink lots of water when it is hot",
    "stay on the trail when you hike",
]

for s in nature_safety:
    add(s, "social", 2, "social")

# --- simple proverbs / wisdom ---
proverbs = [
    "practice makes perfect",
    "slow and steady wins the race",
    "be kind and others will be kind to you",
    "never give up even when it is hard",
    "everyone makes mistakes and that is okay",
    "sharing makes everyone happier",
    "honesty is very important",
    "treat others how you want to be treated",
    "a little help can make a big difference",
    "it is okay to ask for help",
    "you can learn from your mistakes",
    "try your best every day",
    "good things take time",
    "be brave and try new things",
    "a smile can brighten someone's day",
    "being different is what makes you special",
    "teamwork makes things easier",
    "listen more than you talk",
]

for p in proverbs:
    add(p, "social", 2, "social")

# =====================================================================
# 7. EXPANDED GENERATION TO REACH ~10,000
# =====================================================================

# --- "X is a type of Y" taxonomy (bulk) ---
taxonomy = {
    "animal": ["dog", "cat", "fish", "bird", "horse", "cow", "lion", "bear",
               "frog", "snake", "whale", "dolphin", "tiger", "eagle", "penguin",
               "shark", "rabbit", "deer", "fox", "wolf", "goat", "pig",
               "chicken", "duck", "owl", "bat", "seal", "crab", "ant", "bee"],
    "fruit": ["apple", "banana", "orange", "grape", "strawberry", "cherry",
              "peach", "pear", "mango", "watermelon", "lemon", "blueberry",
              "raspberry", "plum", "kiwi", "coconut", "pineapple", "fig",
              "lime", "melon"],
    "vegetable": ["carrot", "potato", "tomato", "broccoli", "pea", "corn",
                  "lettuce", "onion", "pepper", "bean", "cucumber", "spinach",
                  "celery", "garlic", "cabbage", "turnip", "beet", "radish",
                  "zucchini", "sweet potato"],
    "vehicle": ["car", "bus", "truck", "bike", "train", "plane", "boat",
                "ship", "helicopter", "rocket", "van", "taxi", "scooter",
                "tractor", "ambulance", "canoe", "sled", "skateboard"],
    "tool": ["hammer", "saw", "screwdriver", "wrench", "drill", "pliers",
             "shovel", "rake", "ax", "tape measure", "level", "chisel"],
    "instrument": ["piano", "guitar", "drum", "violin", "flute", "trumpet",
                   "harp", "cello", "tuba", "banjo", "harmonica", "xylophone"],
    "clothing": ["shirt", "pants", "shoes", "hat", "socks", "coat", "scarf",
                 "gloves", "boots", "dress", "skirt", "jacket", "belt",
                 "sweater", "vest", "sandals", "tie", "apron"],
    "furniture": ["chair", "table", "bed", "sofa", "desk", "shelf", "lamp",
                  "rug", "dresser", "stool", "bench", "crib", "bookcase"],
    "flower": ["rose", "daisy", "tulip", "sunflower", "lily", "orchid",
               "violet", "daffodil", "poppy", "carnation", "iris", "peony"],
}

for category, items in taxonomy.items():
    for item in items:
        add(f"{item} is a kind of {category}", "fact", 1, "commonsense")

# --- "X can Y" abilities (bulk) ---
can_do = [
    ("bird", "fly"), ("bird", "sing"), ("bird", "build nests"),
    ("fish", "swim"), ("fish", "breathe underwater"),
    ("dog", "bark"), ("dog", "fetch"), ("dog", "dig"), ("dog", "sniff"),
    ("cat", "purr"), ("cat", "climb"), ("cat", "hunt mice"),
    ("frog", "jump"), ("frog", "swim"), ("frog", "catch flies"),
    ("horse", "gallop"), ("horse", "jump fences"), ("horse", "pull carts"),
    ("monkey", "climb trees"), ("monkey", "swing on branches"),
    ("elephant", "spray water"), ("elephant", "remember things"),
    ("rabbit", "hop"), ("rabbit", "dig burrows"),
    ("snake", "slither"), ("snake", "shed its skin"),
    ("bear", "climb trees"), ("bear", "catch fish"),
    ("bee", "make honey"), ("bee", "sting"), ("bee", "fly from flower to flower"),
    ("ant", "carry heavy things"), ("ant", "build tunnels"),
    ("spider", "spin webs"), ("spider", "catch bugs"),
    ("dolphin", "jump out of water"), ("dolphin", "swim fast"),
    ("penguin", "swim"), ("penguin", "slide on ice"),
    ("cheetah", "run very fast"),
    ("owl", "see in the dark"), ("owl", "turn its head far"),
    ("beaver", "build dams"), ("beaver", "chew wood"),
    ("parrot", "talk"), ("parrot", "copy sounds"),
    ("kangaroo", "hop"), ("kangaroo", "carry babies in a pouch"),
    ("chameleon", "change color"), ("octopus", "change color"),
    ("bat", "fly at night"), ("bat", "find things with sound"),
]

for animal, ability in can_do:
    a = a_an(animal)
    add(f"{a} can {ability}", "fact", 1, "animals")
    add(f"what can {a} do", "qa", 2, "animals", answer=ability)

# --- "X is made of Y" ---
made_of = [
    ("a table", "wood"), ("a window", "glass"), ("a coin", "metal"),
    ("a tire", "rubber"), ("a shirt", "cloth"), ("paper", "wood pulp"),
    ("a brick", "clay"), ("a nail", "metal"), ("a rope", "fibers"),
    ("a candle", "wax"), ("a snowman", "snow"), ("a sandcastle", "sand"),
    ("a chain", "metal links"), ("a basket", "woven strips"),
    ("bread", "flour and water"), ("butter", "cream"),
    ("cheese", "milk"), ("ice cream", "cream and sugar"),
    ("a pearl", "layers inside an oyster"), ("honey", "nectar from flowers"),
    ("glass", "melted sand"), ("pottery", "clay"),
    ("chocolate", "cocoa beans"), ("sugar", "sugarcane"),
    ("wool", "sheep fur"), ("silk", "silkworm thread"),
    ("leather", "animal skin"), ("linen", "flax plant"),
    ("concrete", "cement and sand and water"), ("a pencil", "wood and graphite"),
]

for thing, material in made_of:
    add(f"{thing} is made of {material}", "fact", 2, "commonsense")
    add(f"what is {thing} made of", "qa", 2, "commonsense", answer=material)

# --- "where do you find X" ---
find_where = [
    ("sand", "at the beach"), ("books", "in a library"),
    ("fish", "in the water"), ("clouds", "in the sky"),
    ("trees", "in a forest"), ("flowers", "in a garden"),
    ("stars", "in the night sky"), ("shells", "at the beach"),
    ("rocks", "on the ground"), ("snow", "on mountains in winter"),
    ("birds", "in the sky or in trees"), ("cows", "on a farm"),
    ("lions", "in the wild or at a zoo"), ("penguins", "in cold places"),
    ("cactus", "in the desert"), ("seaweed", "in the ocean"),
    ("mushrooms", "in the forest"), ("worms", "in the soil"),
    ("bats", "in caves"), ("coral", "in the ocean"),
    ("a doctor", "at a hospital"), ("a teacher", "at a school"),
    ("a pilot", "in a plane"), ("a farmer", "on a farm"),
    ("food", "in the kitchen"), ("toys", "in a toy box"),
    ("money", "in a bank or wallet"), ("letters", "in a mailbox"),
    ("medicine", "at a pharmacy"), ("gasoline", "at a gas station"),
]

for thing, location in find_where:
    add(f"you find {thing} {location}", "fact", 1, "commonsense")
    add(f"where do you find {thing}", "qa", 1, "commonsense", answer=location)

# --- "X tastes Y" ---
tastes = [
    ("sugar", "sweet"), ("salt", "salty"), ("lemon", "sour"),
    ("coffee", "bitter"), ("pepper", "spicy"), ("honey", "sweet"),
    ("chocolate", "sweet"), ("vinegar", "sour"), ("ginger", "spicy"),
    ("candy", "sweet"), ("grapefruit", "sour and bitter"),
    ("ice cream", "sweet and cold"), ("pizza", "savory"),
    ("bread", "mild"), ("cheese", "salty or creamy"),
    ("apple", "sweet or tart"), ("banana", "sweet and soft"),
    ("orange", "sweet and tangy"), ("milk", "creamy"),
    ("water", "like nothing"),
]

for food, taste in tastes:
    add(f"{food} tastes {taste}", "fact", 1, "commonsense")
    add(f"what does {food} taste like", "qa", 1, "commonsense", answer=taste)

# --- "X feels Y" (texture/touch) ---
feels = [
    ("ice", "cold and hard"), ("sand", "gritty and warm"),
    ("cotton", "soft and fluffy"), ("rock", "hard and rough"),
    ("water", "wet and cool"), ("fur", "soft and warm"),
    ("metal", "cold and smooth"), ("wood", "hard and smooth"),
    ("grass", "tickly and soft"), ("mud", "squishy and wet"),
    ("sandpaper", "very rough"), ("silk", "very smooth"),
    ("a cactus", "prickly"), ("a sponge", "squishy"),
    ("a feather", "light and tickly"), ("snow", "cold and soft"),
    ("a balloon", "smooth and stretchy"), ("jelly", "wobbly and soft"),
    ("a pine cone", "bumpy and hard"), ("velvet", "very soft"),
]

for thing, texture in feels:
    add(f"{thing} feels {texture}", "fact", 1, "commonsense")
    add(f"how does {thing} feel", "qa", 1, "commonsense", answer=texture)

# --- "X smells like Y" ---
smells = [
    ("a rose", "sweet"), ("cookies baking", "warm and sweet"),
    ("a skunk", "very bad"), ("rain", "fresh and clean"),
    ("the ocean", "salty"), ("a fire", "smoky"),
    ("fresh bread", "warm and yummy"), ("garbage", "terrible"),
    ("a lemon", "fresh and sour"), ("pine trees", "fresh and green"),
    ("a cake in the oven", "sweet"), ("popcorn", "buttery"),
    ("soap", "clean"), ("perfume", "flowery"),
    ("grass after mowing", "fresh and green"),
]

for thing, smell in smells:
    add(f"{thing} smells {smell}", "fact", 1, "commonsense")
    add(f"how does {thing} smell", "qa", 1, "commonsense", answer=smell)

# --- "X sounds like Y" ---
sounds_like = [
    ("rain", "pitter patter"), ("thunder", "a loud boom"),
    ("a clock", "tick tock"), ("a bell", "ding dong"),
    ("wind", "whoosh"), ("a cat", "meow"),
    ("a cow", "moo"), ("a rooster", "cock a doodle doo"),
    ("a train", "choo choo"), ("a car horn", "beep beep"),
    ("a drum", "boom boom"), ("popcorn", "pop pop pop"),
    ("water dripping", "drip drip"), ("a whistle", "wheee"),
    ("thunder", "rumble and crack"), ("a snake", "hiss"),
    ("footsteps", "tap tap"), ("a fire", "crackle"),
    ("waves", "splash and crash"), ("birds in the morning", "tweet tweet"),
]

for thing, sound in sounds_like:
    add(f"{thing} sounds like {sound}", "fact", 1, "commonsense")
    add(f"what sound does {thing} make", "qa", 1, "commonsense", answer=sound)

# --- combinatorial: adjective + noun sentences ---
adjectives = ["big", "small", "red", "blue", "green", "yellow", "soft",
              "hard", "old", "new", "fast", "slow", "hot", "cold",
              "happy", "sad", "tall", "short", "round", "flat"]
nouns = ["ball", "box", "cat", "dog", "tree", "house", "car", "bird",
         "fish", "hat", "cup", "book", "shoe", "boat", "star"]
locations = ["on the table", "under the bed", "in the garden",
             "next to the door", "behind the tree", "near the river",
             "inside the box", "on the shelf", "by the window",
             "in the park"]

random.seed(43)  # re-seed for deterministic combos
for _ in range(600):
    adj = random.choice(adjectives)
    noun = random.choice(nouns)
    loc = random.choice(locations)
    add(f"the {adj} {noun} is {loc}", "spatial", 1, "spatial")

# --- combinatorial: "who/what" context QA ---
names = ["ben", "sam", "mia", "zoe", "leo", "ava", "max", "liv", "tom", "ivy"]
actions_list = [
    ("ate", ["an apple", "a cookie", "some bread", "a banana", "some rice",
             "soup", "a sandwich", "some grapes", "corn", "a peach"]),
    ("saw", ["a bird", "a dog", "a cat", "a rainbow", "a butterfly",
             "a rabbit", "a fish", "a plane", "a flower", "a frog"]),
    ("has", ["a red ball", "a blue hat", "a toy car", "a big book",
             "a green cup", "a small dog", "a pink bag", "a yellow pen",
             "a soft bear", "a new bike"]),
    ("likes", ["to swim", "to read", "to paint", "to run", "to sing",
               "to cook", "to dance", "to climb", "to draw", "to play"]),
    ("went to", ["the park", "the store", "the beach", "the school",
                 "the farm", "the zoo", "the library", "the lake",
                 "the garden", "the hill"]),
]

for _ in range(800):
    name1 = random.choice(names)
    name2 = random.choice([n for n in names if n != name1])
    verb, objs = random.choice(actions_list)
    obj1 = random.choice(objs)
    obj2 = random.choice([o for o in objs if o != obj1])
    add(f"{name1} {verb} {obj1}. {name2} {verb} {obj2}. what did {name1} {verb.split()[0]}",
        "qa", 3, "commonsense", answer=obj1)

# --- "true or false" style ---
true_false = [
    ("the sun is hot", True), ("ice is warm", False),
    ("fish can fly", False), ("birds can fly", True),
    ("water is wet", True), ("fire is cold", False),
    ("dogs can bark", True), ("cats can bark", False),
    ("trees have leaves", True), ("rocks are alive", False),
    ("the moon is a star", False), ("the earth is round", True),
    ("snow is white", True), ("grass is blue", False),
    ("elephants are small", False), ("mice are small", True),
    ("the ocean is salty", True), ("rivers flow uphill", False),
    ("humans need air to breathe", True), ("plants need sunlight", True),
    ("whales are fish", False), ("bats are birds", False),
    ("penguins can fly", False), ("dolphins are mammals", True),
    ("the sun rises in the west", False), ("winter is cold", True),
    ("babies are born small", True), ("adults are younger than children", False),
    ("a square has three sides", False), ("a triangle has three sides", True),
    ("you see with your ears", False), ("you hear with your ears", True),
    ("metal is attracted to magnets", True), ("wood is attracted to magnets", False),
    ("rain falls up", False), ("apples grow on trees", True),
    ("chickens lay eggs", True), ("cows lay eggs", False),
    ("the moon shines by itself", False), ("spiders have six legs", False),
    ("spiders have eight legs", True), ("insects have six legs", True),
]

for statement, is_true in true_false:
    answer = "true" if is_true else "false"
    add(f"true or false: {statement}", "qa", 2, "commonsense", answer=answer)

# --- "what color is X" questions ---
color_qa = [
    ("the sky on a clear day", "blue"), ("grass", "green"),
    ("a banana", "yellow"), ("a strawberry", "red"),
    ("milk", "white"), ("chocolate", "brown"),
    ("the sun", "yellow"), ("a carrot", "orange"),
    ("a grape", "purple"), ("a lime", "green"),
    ("snow", "white"), ("coal", "black"),
    ("a flamingo", "pink"), ("the ocean", "blue"),
    ("a lemon", "yellow"), ("a cherry", "red"),
    ("a pumpkin", "orange"), ("an eggplant", "purple"),
    ("a cloud", "white"), ("dirt", "brown"),
    ("a pea", "green"), ("a polar bear", "white"),
    ("a crow", "black"), ("sand", "tan or yellow"),
    ("a ruby", "red"), ("a sapphire", "blue"),
    ("gold", "gold or yellow"), ("silver", "silver or gray"),
]

for thing, color in color_qa:
    add(f"what color is {thing}", "qa", 1, "commonsense", answer=color)

# --- "how many X does Y have" ---
how_many = [
    ("legs", "a dog", "four"), ("legs", "a spider", "eight"),
    ("legs", "an insect", "six"), ("legs", "a bird", "two"),
    ("legs", "a person", "two"), ("eyes", "a person", "two"),
    ("ears", "a person", "two"), ("arms", "a person", "two"),
    ("fingers", "a hand", "five"), ("toes", "a foot", "five"),
    ("wings", "a bird", "two"), ("wheels", "a car", "four"),
    ("wheels", "a bike", "two"), ("wheels", "a tricycle", "three"),
    ("sides", "a triangle", "three"), ("sides", "a square", "four"),
    ("corners", "a rectangle", "four"), ("points", "a star", "five"),
    ("days", "a week", "seven"), ("months", "a year", "twelve"),
    ("seasons", "a year", "four"), ("colors", "a rainbow", "seven"),
    ("strings", "a guitar", "six"), ("keys", "a piano", "eighty eight"),
    ("humps", "a camel", "one or two"), ("stripes", "a zebra", "many"),
    ("legs", "a snake", "zero"), ("legs", "a fish", "zero"),
    ("tentacles", "an octopus", "eight"), ("horns", "a unicorn", "one"),
]

for what, thing, count in how_many:
    add(f"how many {what} does {thing} have", "qa", 2, "commonsense",
        answer=count)
    add(f"{thing} has {count} {what}", "fact", 1, "commonsense")

# --- "which is bigger/smaller/faster/etc" ---
which_questions = [
    ("bigger", "an elephant", "a cat", "an elephant"),
    ("bigger", "the sun", "the moon", "the sun"),
    ("bigger", "a house", "a dog", "a house"),
    ("faster", "a car", "a person walking", "a car"),
    ("faster", "a rocket", "a car", "a rocket"),
    ("faster", "a turtle", "a rabbit", "a rabbit"),
    ("taller", "a giraffe", "a dog", "a giraffe"),
    ("taller", "a building", "a tree", "a building"),
    ("heavier", "a truck", "a cat", "a truck"),
    ("heavier", "a bowling ball", "a tennis ball", "a bowling ball"),
    ("louder", "a whisper", "a shout", "a shout"),
    ("louder", "thunder", "a whisper", "thunder"),
    ("hotter", "the sun", "the moon", "the sun"),
    ("hotter", "fire", "ice", "fire"),
    ("colder", "ice cream", "soup", "ice cream"),
    ("older", "a grandfather", "a baby", "a grandfather"),
    ("smaller", "an ant", "a horse", "an ant"),
    ("deeper", "the ocean", "a puddle", "the ocean"),
    ("longer", "a year", "a day", "a year"),
    ("brighter", "the sun", "a candle", "the sun"),
]

for comp, a_item, b_item, answer in which_questions:
    add(f"which is {comp}: {a_item} or {b_item}", "qa", 2, "commonsense",
        answer=answer)

# --- daily routine sentences (templated) ---
routine_people = ["a child", "a teacher", "a farmer", "a doctor", "a baker"]
routine_morning = ["wakes up early", "eats breakfast", "gets ready for the day",
                   "brushes teeth", "puts on clothes"]
routine_day = ["goes to work", "helps people", "does tasks", "eats lunch",
               "takes a short break"]
routine_evening = ["goes home", "eats dinner", "rests", "spends time with family",
                   "reads or watches something"]

for person in routine_people:
    for act in routine_morning:
        add(f"in the morning {person} {act}", "time", 1, "time")
    for act in routine_day:
        add(f"during the day {person} {act}", "time", 1, "time")
    for act in routine_evening:
        add(f"in the evening {person} {act}", "time", 1, "time")

# --- "X needs Y to work" ---
needs_to_work = [
    ("a car", "gas or electricity"),
    ("a lamp", "electricity"),
    ("a phone", "a battery and signal"),
    ("a plant", "water and sunlight"),
    ("a person", "food and water and air"),
    ("a fire", "fuel and air"),
    ("a boat", "water to float on"),
    ("a clock", "a battery or winding"),
    ("a TV", "electricity"),
    ("a computer", "electricity"),
    ("a bicycle", "someone to pedal"),
    ("an oven", "electricity or gas"),
    ("a flashlight", "batteries"),
    ("a kite", "wind"),
    ("a windmill", "wind"),
    ("a waterwheel", "flowing water"),
    ("a guitar", "someone to play it"),
    ("a sailboat", "wind"),
]

for thing, needs in needs_to_work:
    add(f"{thing} needs {needs} to work", "fact", 2, "commonsense")
    add(f"what does {thing} need to work", "qa", 2, "commonsense", answer=needs)

# --- part-whole relationships ---
part_whole = [
    ("a tree", ["trunk", "branches", "leaves", "roots", "bark"]),
    ("a car", ["wheels", "engine", "doors", "seats", "windows"]),
    ("a house", ["walls", "roof", "floor", "door", "windows"]),
    ("a flower", ["petals", "stem", "leaves", "roots"]),
    ("a face", ["eyes", "nose", "mouth", "ears", "chin"]),
    ("a hand", ["fingers", "thumb", "palm", "nails"]),
    ("a book", ["pages", "cover", "spine", "words"]),
    ("a bicycle", ["wheels", "pedals", "handlebars", "seat", "chain"]),
    ("a shoe", ["sole", "heel", "tongue", "laces"]),
    ("a clock", ["hands", "face", "numbers"]),
    ("a fish", ["fins", "scales", "tail", "gills"]),
    ("a bird", ["wings", "beak", "feathers", "claws", "tail"]),
    ("an airplane", ["wings", "engine", "tail", "cockpit", "wheels"]),
    ("a kitchen", ["stove", "sink", "fridge", "counter", "cabinets"]),
]

for whole, parts in part_whole:
    for part in parts:
        add(f"{whole} has {a_an(part) if part[0] in 'aeiou' else part}",
            "fact", 1, "commonsense")
    add(f"name parts of {whole}", "qa", 2, "commonsense",
        answer=" and ".join(parts[:3]))

# --- "what do you wear when" ---
wear_when = [
    ("it is cold", "a coat and a scarf"),
    ("it is raining", "a raincoat and boots"),
    ("it is sunny", "a hat and sunglasses"),
    ("you go swimming", "a swimsuit"),
    ("you ride a bike", "a helmet"),
    ("you play in the snow", "snow boots and gloves"),
    ("you go to bed", "pajamas"),
    ("it is very hot", "shorts and a light shirt"),
    ("you play sports", "sneakers and a jersey"),
    ("you go to a party", "nice clothes"),
]

for when, what in wear_when:
    add(f"when {when} you wear {what}", "social", 1, "social")
    add(f"what do you wear when {when}", "qa", 1, "social", answer=what)

# --- "what do you use X for" (more) ---
use_for = [
    ("soap", "washing and cleaning"),
    ("a toothbrush", "brushing your teeth"),
    ("a comb", "combing your hair"),
    ("a pan", "cooking food"),
    ("a pot", "boiling water or making soup"),
    ("a bowl", "eating cereal or soup"),
    ("a straw", "drinking from a cup"),
    ("a napkin", "wiping your mouth"),
    ("a seatbelt", "staying safe in a car"),
    ("a backpack", "carrying your things"),
    ("a crayon", "coloring pictures"),
    ("glue", "sticking things together"),
    ("tape", "holding things together"),
    ("a saw", "cutting wood"),
    ("a needle", "sewing cloth"),
    ("a bucket", "carrying water"),
    ("a net", "catching fish or butterflies"),
    ("a tent", "sleeping outside"),
    ("a sleeping bag", "staying warm at camp"),
    ("binoculars", "seeing far away things"),
]

for tool, purpose in use_for:
    add(f"you use {tool} for {purpose}", "fact", 1, "commonsense")
    add(f"what do you use {tool} for", "qa", 1, "commonsense", answer=purpose)

# --- more math: multiplication and simple word problems ---
for a_n in range(1, 6):
    for b_n in range(1, 6):
        add(f"{a_n} times {b_n} is {a_n * b_n}", "fact", 3, "commonsense")

# word problems
word_problems = [
    ("you have 3 apples and get 2 more. how many do you have", "5"),
    ("you have 5 cookies and eat 2. how many are left", "3"),
    ("there are 4 birds on a tree. 1 flies away. how many are left", "3"),
    ("you have 2 bags with 3 toys in each. how many toys do you have", "6"),
    ("a dog has 4 legs. how many legs do 2 dogs have", "8"),
    ("you have 10 grapes and share half. how many do you keep", "5"),
    ("there are 6 eggs. 2 crack. how many are not cracked", "4"),
    ("you have 7 stickers and give 3 away. how many do you have", "4"),
    ("a cat has 2 ears. how many ears do 3 cats have", "6"),
    ("you read 2 pages a day for 5 days. how many pages did you read", "10"),
    ("there are 8 cupcakes for 4 friends. how many does each get", "2"),
    ("you have 1 red ball and 3 blue balls. how many balls total", "4"),
    ("you pick 5 flowers then pick 5 more. how many flowers do you have", "10"),
    ("a hand has 5 fingers. how many fingers on 2 hands", "10"),
    ("you see 3 ducks and then 4 more. how many ducks total", "7"),
    ("you have 9 coins and lose 4. how many coins left", "5"),
    ("there are 2 rows of 5 chairs. how many chairs total", "10"),
    ("a car has 4 wheels. how many wheels on 3 cars", "12"),
    ("you buy 6 bananas and eat 1. how many are left", "5"),
    ("there are 3 fish in one bowl and 2 in another. how many fish total", "5"),
]

for problem, answer in word_problems:
    add(problem, "qa", 3, "commonsense", answer=answer)

# --- "X or Y" preference / category questions ---
or_questions = [
    ("is a tomato a fruit or a vegetable", "it is actually a fruit"),
    ("is a whale a fish or a mammal", "a mammal"),
    ("is a bat a bird or a mammal", "a mammal"),
    ("is the sun a planet or a star", "a star"),
    ("is glass a solid or a liquid", "a solid"),
    ("does a plant eat food or make its own food", "it makes its own food"),
    ("does sound travel faster in air or water", "in water"),
    ("is the earth closer to the sun or moon", "the moon is closer to earth"),
    ("do birds have teeth or beaks", "beaks"),
    ("do fish breathe with lungs or gills", "gills"),
]

for question, answer in or_questions:
    add(question, "qa", 3, "commonsense", answer=answer)

# --- habitat + adaptation combos ---
habitats = [
    ("desert", ["camel", "lizard", "cactus", "scorpion", "rattlesnake"],
     "very hot and dry"),
    ("ocean", ["fish", "whale", "shark", "jellyfish", "octopus"],
     "deep and full of salt water"),
    ("forest", ["deer", "bear", "owl", "squirrel", "wolf"],
     "full of trees"),
    ("arctic", ["polar bear", "penguin", "seal", "walrus", "snowy owl"],
     "very cold with ice and snow"),
    ("jungle", ["monkey", "parrot", "tiger", "snake", "toucan"],
     "warm and wet with tall trees"),
    ("pond", ["frog", "duck", "turtle", "fish", "dragonfly"],
     "a small body of fresh water"),
    ("grassland", ["lion", "zebra", "elephant", "giraffe", "cheetah"],
     "a big flat area with lots of grass"),
    ("farm", ["cow", "pig", "chicken", "horse", "goat"],
     "where people raise animals and grow crops"),
]

for habitat, animals, desc in habitats:
    add(f"the {habitat} is {desc}", "fact", 1, "commonsense")
    for animal in animals:
        a = a_an(animal)
        add(f"{a} lives in the {habitat}", "fact", 1, "animals")
        add(f"where does {a} live", "qa", 2, "animals",
            answer=f"in the {habitat}")

# --- what happens next (prediction) ---
what_next = [
    ("the ice cream is in the hot sun", "it will melt"),
    ("the baby is crying", "someone will pick it up"),
    ("dark clouds fill the sky", "it will rain soon"),
    ("the dog sees a cat", "the dog might chase the cat"),
    ("the kettle is boiling", "someone will pour the water"),
    ("the child yawns a lot", "the child will fall asleep"),
    ("the glass is at the edge of the table", "it might fall and break"),
    ("the plant has not been watered for days", "it will wilt"),
    ("the traffic light turns red", "the cars will stop"),
    ("the oven timer goes off", "the food is ready"),
    ("the bird flaps its wings", "it will fly away"),
    ("the boy blows the candles on his cake", "the flames go out"),
    ("water fills the bathtub to the top", "it will overflow"),
    ("the snow is melting", "spring is coming"),
    ("the kitten sees a ball of yarn", "it will play with it"),
]

for setup, prediction in what_next:
    add(f"{setup}. what happens next", "qa", 3, "commonsense",
        answer=prediction)
    add(f"if {setup} then {prediction}", "cause_effect", 2, "commonsense")

# --- "same and different" ---
same_different = [
    ("a dog and a cat", "are both animals", "dogs bark and cats meow"),
    ("an apple and a banana", "are both fruit", "apples are red and bananas are yellow"),
    ("a car and a bus", "are both vehicles", "a bus is bigger"),
    ("summer and winter", "are both seasons", "summer is hot and winter is cold"),
    ("a circle and a square", "are both shapes", "a circle is round and a square has corners"),
    ("rain and snow", "both come from clouds", "rain is water and snow is frozen"),
    ("a piano and a guitar", "are both instruments", "a piano has keys and a guitar has strings"),
    ("the sun and the moon", "are both in the sky", "the sun is a star and the moon is not"),
    ("a book and a tablet", "can both be read", "a book is made of paper"),
    ("a river and a lake", "are both water", "a river flows and a lake is still"),
]

for pair, same, diff in same_different:
    add(f"{pair} {same}", "fact", 2, "commonsense")
    add(f"{pair} are different because {diff}", "fact", 2, "commonsense")
    add(f"how are {pair} the same", "qa", 2, "commonsense", answer=same[4:])  # strip "are "
    add(f"how are {pair} different", "qa", 3, "commonsense", answer=diff)

# --- "X before Y" ordering knowledge ---
before_after = [
    ("plant a seed", "water it"),
    ("wake up", "eat breakfast"),
    ("get dirty", "take a bath"),
    ("cook food", "eat it"),
    ("feel sick", "go to the doctor"),
    ("get a cut", "put a bandage on it"),
    ("it rains", "puddles form"),
    ("spring", "summer"),
    ("the egg hatches", "the chick comes out"),
    ("the sun rises", "it gets light"),
    ("you throw a ball", "it goes up"),
    ("the alarm rings", "you wake up"),
    ("the caterpillar makes a cocoon", "the butterfly comes out"),
    ("you turn the key", "the door opens"),
    ("the ice cream truck comes", "children run outside"),
]

for first, second in before_after:
    add(f"{first} comes before {second}", "time", 2, "time")
    add(f"what comes before {second}", "qa", 2, "time", answer=first)
    add(f"what comes after {first}", "qa", 2, "time", answer=second)

# --- more spatial: preposition combos ---
prep_objects = ["the ball", "the cat", "the book", "the cup", "the toy"]
prep_locations = [
    ("on", "the table"), ("under", "the chair"), ("behind", "the door"),
    ("in front of", "the TV"), ("next to", "the lamp"), ("inside", "the box"),
    ("between", "the pillows"), ("above", "the shelf"),
]

for obj in prep_objects:
    for prep, loc in prep_locations:
        add(f"{obj} is {prep} {loc}", "spatial", 1, "spatial")

# --- emotional responses to events (more) ---
event_emotions = [
    ("you find a lost puppy", "worried but want to help"),
    ("you make a new friend", "happy and excited"),
    ("it is your first day at school", "nervous and excited"),
    ("you get a gold star", "proud and happy"),
    ("you see a big spider", "scared or surprised"),
    ("your friend moves away", "sad"),
    ("you learn to ride a bike", "proud"),
    ("someone shares their snack with you", "grateful and happy"),
    ("you break your favorite toy", "sad and upset"),
    ("you hear a spooky noise", "scared"),
    ("your team wins a game", "excited and happy"),
    ("you get a surprise gift", "surprised and happy"),
    ("you cannot find your mom in the store", "scared and worried"),
    ("you help someone who is sad", "good and kind"),
    ("you see fireworks", "amazed and excited"),
    ("you eat your favorite food", "happy and satisfied"),
    ("someone says you did a great job", "proud"),
    ("you trip and fall in front of people", "embarrassed"),
    ("you wait a long time in line", "bored and impatient"),
    ("you finally solve a hard puzzle", "proud and relieved"),
]

for event, emotion in event_emotions:
    add(f"when {event} you feel {emotion}", "social", 2, "social")
    add(f"how would you feel if {event}", "qa", 2, "social", answer=emotion)

# --- "why" questions (bulk) ---
why_questions = [
    ("why do we brush our teeth", "to keep them clean and healthy"),
    ("why do we eat food", "to get energy and grow"),
    ("why do we drink water", "because our body needs water to work"),
    ("why do we sleep", "so our body and brain can rest"),
    ("why do birds build nests", "to lay their eggs and raise their babies"),
    ("why do leaves fall in autumn", "because the tree is getting ready for winter"),
    ("why does it get dark at night", "because the sun is on the other side of the earth"),
    ("why do we wear clothes", "to stay warm and protected"),
    ("why do dogs wag their tails", "because they are happy"),
    ("why do fish have gills", "to breathe underwater"),
    ("why do we have bones", "to hold up our body"),
    ("why do we sweat", "to cool down our body"),
    ("why is the sky blue", "because of how sunlight hits the air"),
    ("why do we need air", "because we breathe oxygen to stay alive"),
    ("why do flowers smell nice", "to attract bees and butterflies"),
    ("why does the moon change shape", "because we see different parts lit by the sun"),
    ("why do we have eyebrows", "to keep sweat out of our eyes"),
    ("why do cats purr", "because they feel happy and safe"),
    ("why do we get goosebumps", "because our body is trying to warm up"),
    ("why do we have fingerprints", "to help us grip things"),
    ("why does bread rise", "because yeast makes tiny bubbles of gas"),
    ("why do boats float", "because they push water aside and are lighter overall"),
    ("why does metal feel cold", "because it takes heat from your hand quickly"),
    ("why do onions make you cry", "because they release a gas that stings your eyes"),
    ("why do stars twinkle", "because light bends through the air"),
    ("why do we hiccup", "because our diaphragm jumps"),
    ("why is grass green", "because of a chemical called chlorophyll"),
    ("why do leaves change color", "because chlorophyll breaks down in fall"),
    ("why does a ball bounce", "because it is made of springy material"),
    ("why do magnets stick to metal", "because of invisible magnetic force"),
]

for q, a in why_questions:
    add(q, "qa", 2, "commonsense", answer=a)

# --- "what is the difference between" ---
differences = [
    ("a pond and an ocean", "a pond is small and an ocean is huge"),
    ("day and night", "day has sunlight and night is dark"),
    ("a hill and a mountain", "a mountain is much taller"),
    ("a lake and a river", "a river flows but a lake stays still"),
    ("hot and warm", "hot is much warmer than warm"),
    ("a plant and an animal", "a plant makes food from sunlight but an animal eats food"),
    ("a puppy and a dog", "a puppy is a baby dog"),
    ("ice and snow", "ice is solid and smooth but snow is soft and fluffy"),
    ("a city and a village", "a city is bigger with more people"),
    ("a cloud and fog", "fog is a cloud that touches the ground"),
]

for pair, diff in differences:
    add(f"what is the difference between {pair}", "qa", 3, "commonsense",
        answer=diff)

# --- fill in additional physical world via templates ---
container_contents_extra = [
    ("a jar", ["pickles", "jam", "honey", "buttons", "coins"]),
    ("a pocket", ["keys", "coins", "lint", "a tissue"]),
    ("a wallet", ["money", "cards", "pictures"]),
    ("a toolbox", ["a hammer", "nails", "a wrench", "tape"]),
    ("a pencil case", ["pencils", "pens", "an eraser", "a sharpener"]),
    ("a lunchbox", ["a sandwich", "an apple", "a juice box"]),
    ("a suitcase", ["clothes", "shoes", "a toothbrush"]),
    ("an aquarium", ["fish", "water", "plants", "rocks"]),
    ("a nest", ["eggs", "twigs", "feathers"]),
    ("a treasure chest", ["gold coins", "jewels", "a map"]),
]

for container, contents in container_contents_extra:
    for c in contents:
        add(f"you might find {c} in {container}", "spatial", 1, "spatial")

# --- "first, next, last" for everyday events ---
event_orders = [
    ("eating an apple", "pick it up", "bite it", "chew and swallow"),
    ("opening a present", "find the present", "tear off the wrapping", "see what is inside"),
    ("making a bed", "pull up the sheet", "pull up the blanket", "put the pillow on top"),
    ("flying a kite", "hold the string", "run into the wind", "let the kite go up"),
    ("building a snowman", "roll a big snowball", "stack a smaller one on top", "add a face"),
    ("reading a book", "open the book", "read the pages", "close the book"),
    ("crossing the street", "look both ways", "wait for cars to stop", "walk across"),
    ("pouring juice", "get a cup", "pick up the bottle", "pour into the cup"),
    ("feeding a pet", "get the food bowl", "put food in the bowl", "give it to the pet"),
    ("taking a photo", "hold the camera", "look through it", "press the button"),
]

for task, first, middle, last in event_orders:
    add(f"when {task} first you {first} then you {middle} then you {last}",
        "sequence", 2, "time")
    add(f"what do you do first when {task}", "qa", 2, "time", answer=first)
    add(f"what do you do last when {task}", "qa", 2, "time", answer=last)

# --- holidays and celebrations ---
holidays = [
    ("birthday", "the day you were born", "cake and presents"),
    ("new year", "the start of a new year", "fireworks and counting down"),
    ("thanksgiving", "a day to be thankful", "a big family dinner"),
    ("halloween", "a spooky holiday in october", "costumes and candy"),
    ("valentine's day", "a day about love", "cards and hearts"),
    ("easter", "a spring holiday", "egg hunts and chocolate"),
    ("fourth of july", "a holiday about freedom", "fireworks and parades"),
    ("christmas", "a winter holiday", "gifts and a tree"),
]

for holiday, desc, tradition in holidays:
    add(f"{holiday} is {desc}", "fact", 1, "social")
    add(f"on {holiday} people enjoy {tradition}", "fact", 1, "social")
    add(f"what is {holiday}", "qa", 2, "social", answer=desc)

# --- "X lives for about Y" ---
lifespans = [
    ("a fly", "a few weeks"), ("a mouse", "about two years"),
    ("a dog", "about ten to fifteen years"),
    ("a cat", "about fifteen years"), ("a horse", "about twenty five years"),
    ("a person", "about seventy to eighty years"),
    ("a tortoise", "over one hundred years"),
    ("a goldfish", "about five to ten years"),
    ("an elephant", "about sixty years"),
    ("a parrot", "about fifty years"),
    ("a butterfly", "a few weeks"),
    ("an oak tree", "hundreds of years"),
]

for creature, span in lifespans:
    add(f"{creature} lives for {span}", "fact", 2, "biology")
    add(f"how long does {creature} live", "qa", 2, "biology", answer=span)

# --- more "if X then Y" chains (cause-effect combos) ---
if_then_chains = [
    ("you plant seeds in spring", "they grow in summer", "you pick them in fall"),
    ("it snows a lot", "you can build a snowman", "when it melts the snowman is gone"),
    ("you practice piano every day", "you get better at it", "soon you can play a song"),
    ("a caterpillar eats lots of leaves", "it gets big and fat", "then it turns into a butterfly"),
    ("the sun heats the ocean", "water turns to vapor", "the vapor makes clouds"),
    ("you save money every week", "your piggy bank gets full", "then you can buy something nice"),
    ("the baby bird grows feathers", "it learns to fly", "then it leaves the nest"),
    ("you mix flour and eggs and sugar", "you make cake batter", "you bake it into a cake"),
]

for step1, step2, step3 in if_then_chains:
    add(f"first {step1} then {step2} then {step3}", "cause_effect", 3, "commonsense")
    add(f"what happens after {step1}", "qa", 3, "commonsense", answer=step2)

# --- "X helps you Y" ---
helps_you = [
    ("a map", "find your way"),
    ("a dictionary", "find the meaning of words"),
    ("a calendar", "know what day it is"),
    ("a watch", "know what time it is"),
    ("medicine", "feel better when you are sick"),
    ("a coat", "stay warm in winter"),
    ("sunglasses", "protect your eyes from the sun"),
    ("a bandage", "cover a cut"),
    ("a recipe", "know how to cook something"),
    ("a teacher", "learn new things"),
    ("a friend", "feel happy and not alone"),
    ("sleep", "feel rested"),
    ("practice", "get better at things"),
    ("a magnifying glass", "see small things bigger"),
    ("a flashlight", "see in the dark"),
]

for thing, purpose in helps_you:
    add(f"{thing} helps you {purpose}", "fact", 1, "commonsense")
    add(f"how does {thing} help you", "qa", 2, "commonsense",
        answer=f"it helps you {purpose}")

# --- more combinatorial name + attribute stories ---
random.seed(43)
attributes = ["red", "blue", "big", "small", "happy", "fast", "green", "tall"]
things_owned = ["ball", "hat", "cup", "toy", "kite", "book", "bike", "bag"]

for _ in range(400):
    n1 = random.choice(names)
    n2 = random.choice([n for n in names if n != n1])
    a1 = random.choice(attributes)
    a2 = random.choice([a for a in attributes if a != a1])
    t1 = random.choice(things_owned)
    t2 = random.choice([t for t in things_owned if t != t1])
    add(f"{n1} has a {a1} {t1}. {n2} has a {a2} {t2}. what does {n1} have",
        "qa", 3, "commonsense", answer=f"a {a1} {t1}")

# =====================================================================
# 8. FINAL BULK EXPANSION TO REACH ~10,000
# =====================================================================

# --- more combinatorial QA: "X did Y at Z" ---
random.seed(43)
verbs_past = ["played", "ran", "sat", "jumped", "walked", "read", "drew",
              "sang", "slept", "danced"]
places_list = ["the park", "the beach", "the school", "the garden",
               "the kitchen", "the forest", "the library", "the pool",
               "the hill", "the store"]
times_list = ["in the morning", "in the afternoon", "at night",
              "on monday", "yesterday", "last week", "on a sunny day",
              "after lunch", "before dinner", "at noon"]

for _ in range(600):
    n1 = random.choice(names)
    n2 = random.choice([n for n in names if n != n1])
    v = random.choice(verbs_past)
    p1 = random.choice(places_list)
    p2 = random.choice([p for p in places_list if p != p1])
    t = random.choice(times_list)
    add(f"{n1} {v} at {p1} {t}. {n2} {v} at {p2}. where did {n1} {v.split()[0]}",
        "qa", 3, "commonsense", answer=f"at {p1}")

# --- object property combos: material + use + location ---
random.seed(43)
obj_material_use = [
    ("a wooden spoon", "wood", "stirring food", "the kitchen"),
    ("a metal key", "metal", "opening doors", "your pocket"),
    ("a glass jar", "glass", "storing things", "the pantry"),
    ("a rubber duck", "rubber", "playing in the bath", "the bathroom"),
    ("a plastic cup", "plastic", "drinking from", "the table"),
    ("a paper bag", "paper", "carrying groceries", "the store"),
    ("a cotton towel", "cotton", "drying yourself", "the bathroom"),
    ("a leather belt", "leather", "holding up pants", "the closet"),
    ("a clay pot", "clay", "holding a plant", "the garden"),
    ("a stone wall", "stone", "keeping things in or out", "outside"),
    ("a wool hat", "wool", "keeping your head warm", "the closet"),
    ("a silk scarf", "silk", "looking nice and staying warm", "a drawer"),
]

for obj, mat, use, loc in obj_material_use:
    add(f"{obj} is made of {mat}", "fact", 1, "commonsense")
    add(f"{obj} is used for {use}", "fact", 1, "commonsense")
    add(f"you can find {obj} in {loc}", "spatial", 1, "spatial")
    add(f"what is {obj} made of", "qa", 2, "commonsense", answer=mat)
    add(f"what is {obj} used for", "qa", 1, "commonsense", answer=use)

# --- "when it is [weather], you should..." ---
weather_actions = [
    ("raining", ["take an umbrella", "wear rain boots", "stay dry inside",
                 "jump in puddles", "watch from the window"]),
    ("snowing", ["wear a warm coat", "make a snowman", "go sledding",
                 "drink hot cocoa", "put on mittens"]),
    ("sunny and hot", ["drink lots of water", "wear sunscreen",
                       "swim in the pool", "eat ice cream", "play outside"]),
    ("windy", ["fly a kite", "hold onto your hat", "watch the leaves blow",
               "feel the breeze", "stay inside if it is too strong"]),
    ("foggy", ["be careful walking", "use headlights", "walk slowly",
               "stay close to home"]),
    ("icy", ["walk carefully", "do not run on ice", "wear boots with grip",
             "hold onto the railing"]),
]

for weather, actions in weather_actions:
    for act in actions:
        add(f"when it is {weather} you can {act}", "social", 1, "social")

# --- animal + body part + purpose combos ---
animal_parts_purpose = [
    ("a duck", "webbed feet", "to swim"),
    ("a woodpecker", "a strong beak", "to peck wood"),
    ("a giraffe", "a long neck", "to reach tall trees"),
    ("a porcupine", "sharp quills", "to protect itself"),
    ("a skunk", "a bad smell", "to keep enemies away"),
    ("a hummingbird", "tiny wings", "to hover near flowers"),
    ("a camel", "a hump", "to store fat for energy"),
    ("a chameleon", "color changing skin", "to hide from danger"),
    ("an armadillo", "a hard shell", "to protect itself"),
    ("a beaver", "flat tail", "to slap the water as a warning"),
    ("a hawk", "sharp claws", "to catch prey"),
    ("a pelican", "a big throat pouch", "to scoop up fish"),
    ("a mole", "big front paws", "to dig tunnels"),
    ("an anteater", "a long tongue", "to catch ants"),
    ("a gecko", "sticky toes", "to climb walls"),
    ("a firefly", "a glowing body", "to find a mate at night"),
]

for animal, part, purpose in animal_parts_purpose:
    add(f"{animal} has {part} {purpose}", "fact", 2, "biology")
    add(f"why does {animal} have {part}", "qa", 3, "biology",
        answer=purpose)
    add(f"what does {animal} use {part} for", "qa", 2, "biology",
        answer=purpose)

# --- more "which one does not belong" ---
random.seed(43)
odd_one_out_groups = [
    (["apple", "banana", "carrot", "grape"], "carrot", "the others are fruits"),
    (["dog", "cat", "bird", "table"], "table", "the others are animals"),
    (["red", "blue", "chair", "green"], "chair", "the others are colors"),
    (["car", "bus", "tree", "train"], "tree", "the others are vehicles"),
    (["piano", "guitar", "shoe", "drum"], "shoe", "the others are instruments"),
    (["shirt", "pants", "banana", "hat"], "banana", "the others are clothing"),
    (["water", "milk", "rock", "juice"], "rock", "the others are drinks"),
    (["summer", "winter", "pencil", "spring"], "pencil", "the others are seasons"),
    (["eye", "ear", "book", "nose"], "book", "the others are body parts"),
    (["hammer", "saw", "cloud", "drill"], "cloud", "the others are tools"),
    (["rose", "daisy", "fork", "tulip"], "fork", "the others are flowers"),
    (["soccer", "tennis", "lamp", "swimming"], "lamp", "the others are sports"),
    (["circle", "square", "dog", "triangle"], "dog", "the others are shapes"),
    (["oak", "pine", "truck", "maple"], "truck", "the others are trees"),
    (["eagle", "penguin", "cow", "parrot"], "cow", "the others are birds"),
]

for group, odd, reason in odd_one_out_groups:
    items_str = " and ".join(group)
    add(f"which does not belong: {items_str}", "qa", 3, "commonsense",
        answer=f"{odd} because {reason}")

# --- "X is the opposite action of Y" ---
opposite_actions = [
    ("open", "close"), ("push", "pull"), ("stand up", "sit down"),
    ("wake up", "fall asleep"), ("come in", "go out"),
    ("start", "stop"), ("build", "destroy"), ("fill", "empty"),
    ("tie", "untie"), ("lock", "unlock"), ("turn on", "turn off"),
    ("put on", "take off"), ("pick up", "put down"),
    ("give", "take"), ("teach", "learn"), ("ask", "answer"),
    ("remember", "forget"), ("win", "lose"), ("find", "lose"),
    ("laugh", "cry"), ("buy", "sell"), ("add", "subtract"),
    ("climb up", "climb down"), ("speed up", "slow down"),
]

for a, b in opposite_actions:
    add(f"the opposite of {a} is {b}", "fact", 1, "commonsense")
    add(f"what is the opposite of {a}", "qa", 1, "commonsense", answer=b)
    add(f"what is the opposite of {b}", "qa", 1, "commonsense", answer=a)

# --- more combinatorial adjective sentences with QA ---
random.seed(43)
adj_pairs = [("big", "small"), ("fast", "slow"), ("hot", "cold"),
             ("happy", "sad"), ("old", "new"), ("tall", "short"),
             ("clean", "dirty"), ("soft", "hard"), ("bright", "dark"),
             ("wet", "dry")]

common_nouns = ["ball", "dog", "cat", "house", "car", "tree", "bird",
                "fish", "flower", "rock", "boat", "hat", "box", "kite"]

for _ in range(500):
    a1, a2 = random.choice(adj_pairs)
    n1 = random.choice(common_nouns)
    n2 = random.choice([n for n in common_nouns if n != n1])
    add(f"the {n1} is {a1} but the {n2} is {a2}", "fact", 1, "commonsense")
    add(f"the {a1} {n1} and the {a2} {n2} are different", "fact", 1, "commonsense")

# --- cooking / food preparation ---
cooking_steps = [
    ("scrambled eggs", "crack eggs into a bowl", "mix them up", "cook in a pan"),
    ("a salad", "wash the lettuce", "chop vegetables", "add dressing"),
    ("toast", "put bread in the toaster", "wait for it to pop up", "add butter"),
    ("popcorn", "put kernels in a pot", "heat them up", "listen for the pops"),
    ("a smoothie", "put fruit in a blender", "add milk or juice", "blend until smooth"),
    ("pasta", "boil water", "put pasta in the water", "cook until soft"),
    ("a peanut butter sandwich", "get two slices of bread",
     "spread peanut butter on one", "put the other slice on top"),
    ("hot chocolate", "warm up milk", "add chocolate powder", "stir and drink"),
    ("cookies", "mix butter sugar and flour", "shape the dough",
     "bake in the oven"),
    ("pancakes", "mix flour eggs and milk", "pour batter on a hot pan",
     "flip when bubbles appear"),
]

for dish, s1, s2, s3 in cooking_steps:
    add(f"to make {dish}: first {s1} then {s2} then {s3}",
        "sequence", 2, "time")
    add(f"what is the first step to make {dish}", "qa", 2, "time", answer=s1)
    add(f"what is the last step to make {dish}", "qa", 2, "time", answer=s3)

# --- "X is bigger/smaller/faster/slower than Y" (more bulk) ---
size_order = [
    ("ant", "mouse", "cat", "dog", "horse", "elephant", "whale"),
    ("pebble", "rock", "boulder", "hill", "mountain"),
    ("puddle", "pond", "lake", "sea", "ocean"),
    ("seed", "pea", "apple", "watermelon", "pumpkin"),
    ("second", "minute", "hour", "day", "week", "month", "year"),
]

for chain in size_order:
    for i in range(len(chain)):
        for j in range(i + 1, len(chain)):
            a_item = a_an(chain[i])
            b_item = a_an(chain[j])
            add(f"{a_item} is smaller than {b_item}", "fact", 2, "commonsense")

# --- speed ordering ---
speed_chain = ["snail", "turtle", "person", "horse", "car", "train",
               "plane", "rocket"]
for i in range(len(speed_chain)):
    for j in range(i + 1, len(speed_chain)):
        a_item = a_an(speed_chain[i])
        b_item = a_an(speed_chain[j])
        add(f"{a_item} is slower than {b_item}", "fact", 2, "commonsense")

# --- "does X have Y" yes/no ---
has_doesnt = [
    ("a dog", "a tail", True), ("a fish", "legs", False),
    ("a bird", "feathers", True), ("a snake", "legs", False),
    ("a spider", "wings", False), ("a bat", "wings", True),
    ("a turtle", "a shell", True), ("a frog", "fur", False),
    ("a horse", "hooves", True), ("a cat", "scales", False),
    ("a tree", "roots", True), ("a rock", "leaves", False),
    ("a car", "wheels", True), ("a boat", "wheels", False),
    ("a person", "a heart", True), ("a robot", "feelings", False),
    ("a flower", "petals", True), ("a mushroom", "leaves", False),
    ("an owl", "big eyes", True), ("a worm", "teeth", False),
    ("a shark", "sharp teeth", True), ("a jellyfish", "bones", False),
    ("a crab", "claws", True), ("a starfish", "a brain", False),
    ("a penguin", "flippers", True), ("a whale", "gills", False),
]

for thing, part, has in has_doesnt:
    yn = "yes" if has else "no"
    if has:
        add(f"{thing} has {part}", "fact", 1, "commonsense")
    else:
        add(f"{thing} does not have {part}", "fact", 1, "commonsense")
    add(f"does {thing} have {part}", "qa", 2, "commonsense", answer=yn)

# --- "what comes in a group of X" ---
group_counts = [
    ("a pair", "two", ["shoes", "socks", "gloves", "earrings", "eyes", "ears"]),
    ("a dozen", "twelve", ["eggs", "donuts", "roses", "cookies"]),
    ("a trio", "three", ["musicians", "friends", "colors in a traffic light"]),
]

for group, count, examples in group_counts:
    add(f"{group} means {count}", "fact", 2, "commonsense")
    for ex in examples:
        add(f"{ex} often come in {group}", "fact", 2, "commonsense")

# --- simple logic / if-then reasoning ---
logic_chains = [
    ("all dogs have four legs. max is a dog", "max has four legs"),
    ("all birds have feathers. a robin is a bird", "a robin has feathers"),
    ("all fish live in water. a goldfish is a fish", "a goldfish lives in water"),
    ("all cats have tails. luna is a cat", "luna has a tail"),
    ("all flowers need sun. a daisy is a flower", "a daisy needs sun"),
    ("if it rains the street gets wet. it is raining",
     "the street is getting wet"),
    ("if you are tired you should sleep. you are tired",
     "you should sleep"),
    ("plants need water to grow. the plant has no water",
     "the plant will not grow"),
    ("ice melts in heat. the ice is in the sun", "the ice will melt"),
    ("birds fly south in winter. it is winter", "birds fly south"),
]

for premise, conclusion in logic_chains:
    add(f"{premise}. so {conclusion}", "cause_effect", 3, "commonsense")
    add(f"{premise}. what can you conclude", "qa", 3, "commonsense",
        answer=conclusion)

# --- more templated animal facts ---
animal_facts_extra = [
    ("a flamingo is pink because it eats shrimp", "biology"),
    ("a chameleon changes color to hide", "biology"),
    ("a salmon swims upstream to lay eggs", "biology"),
    ("a caterpillar turns into a butterfly", "biology"),
    ("a tadpole turns into a frog", "biology"),
    ("a bear sleeps all winter long", "biology"),
    ("a squirrel hides nuts for winter", "biology"),
    ("a camel can go days without water", "biology"),
    ("a koala sleeps up to twenty hours a day", "biology"),
    ("a peacock spreads its tail to show off", "biology"),
    ("ants work together as a team", "biology"),
    ("bees make honey from flower nectar", "biology"),
    ("a spider builds a web to catch food", "biology"),
    ("a hermit crab lives in a borrowed shell", "biology"),
    ("a sloth moves very slowly", "biology"),
    ("a hummingbird is the smallest bird", "biology"),
    ("an ostrich is the biggest bird", "biology"),
    ("an ostrich cannot fly but runs very fast", "biology"),
    ("a pufferfish puffs up to scare enemies", "biology"),
    ("a dolphin breathes air like us", "biology"),
    ("a sea turtle returns to the beach to lay eggs", "biology"),
    ("penguins huddle together to stay warm", "biology"),
    ("a mother kangaroo carries her baby in a pouch", "biology"),
    ("an eagle has very sharp eyesight", "biology"),
    ("a owl can rotate its head almost all the way around", "biology"),
    ("a cheetah is the fastest land animal", "biology"),
    ("a blue whale is the biggest animal ever", "biology"),
    ("an ant can carry fifty times its own weight", "biology"),
    ("a dog can hear sounds people cannot", "biology"),
    ("a cat always lands on its feet", "biology"),
]

for fact, cat in animal_facts_extra:
    add(fact, "fact", 2, cat)

# --- "what season is best for X" ---
season_best = [
    ("swimming outside", "summer"), ("building snowmen", "winter"),
    ("picking apples", "fall"), ("planting flowers", "spring"),
    ("flying kites", "spring"), ("going to the beach", "summer"),
    ("watching leaves change color", "fall"),
    ("drinking hot cocoa", "winter"), ("having a picnic", "summer"),
    ("raking leaves", "fall"), ("seeing baby animals born", "spring"),
    ("going sledding", "winter"), ("eating watermelon", "summer"),
    ("seeing snow", "winter"), ("wearing shorts", "summer"),
    ("wearing a heavy coat", "winter"),
]

for activity, season in season_best:
    add(f"the best season for {activity} is {season}", "fact", 1, "time")
    add(f"when is the best time for {activity}", "qa", 1, "time",
        answer=season)

# --- bulk "the X is [property]" physics statements ---
physics_properties = [
    ("water", ["wet", "clear", "a liquid", "needed for life"]),
    ("air", ["invisible", "all around us", "needed for breathing"]),
    ("the sun", ["bright", "hot", "far away", "a star"]),
    ("the moon", ["in the sky at night", "smaller than the sun", "rocky"]),
    ("dirt", ["brown", "under the grass", "where plants grow"]),
    ("wood", ["from trees", "can burn", "hard", "used to build things"]),
    ("iron", ["a metal", "strong", "heavy", "attracted to magnets"]),
    ("gold", ["a shiny metal", "yellow", "valuable", "rare"]),
    ("salt", ["white", "makes food taste good", "comes from the sea"]),
    ("oil", ["slippery", "a liquid", "does not mix with water"]),
    ("glass", ["see through", "hard", "breaks easily", "made from sand"]),
    ("ice", ["frozen water", "cold", "slippery", "melts when warm"]),
    ("fire", ["hot", "gives light", "needs air", "can be dangerous"]),
    ("smoke", ["gray", "comes from fire", "rises up", "hard to breathe"]),
    ("electricity", ["powers our lights", "can be dangerous",
                     "travels through wires"]),
]

for thing, props in physics_properties:
    for prop in props:
        add(f"{thing} is {prop}", "physics", 1, "commonsense")

# --- "you should not X because Y" safety ---
should_not = [
    ("run near a swimming pool", "you might slip and fall"),
    ("eat food off the ground", "it has germs on it"),
    ("touch a hot pan", "it will burn you"),
    ("play with matches", "you could start a fire"),
    ("swim alone", "it is not safe without help nearby"),
    ("take medicine without an adult", "you might take too much"),
    ("climb very high without help", "you could fall"),
    ("ride a bike without a helmet", "you could hurt your head"),
    ("lean out of a window", "you might fall out"),
    ("run across the street", "a car might not see you"),
    ("put small things in your mouth", "you could choke"),
    ("talk to strangers online", "they might not be who they say"),
    ("play with sharp knives", "you could cut yourself"),
    ("leave toys on the stairs", "someone could trip"),
    ("run with food in your mouth", "you could choke"),
]

for action, reason in should_not:
    add(f"you should not {action} because {reason}", "social", 2, "social")
    add(f"why should you not {action}", "qa", 2, "social",
        answer=f"because {reason}")

# --- "X can be Y or Z" (flexibility/variety) ---
can_be = [
    ("weather", "sunny or rainy or snowy or windy"),
    ("water", "a solid or a liquid or a gas"),
    ("food", "hot or cold or warm"),
    ("a person's mood", "happy or sad or angry or calm"),
    ("a day", "long or short"),
    ("a sound", "loud or soft"),
    ("a color", "light or dark"),
    ("an animal", "big or small or tiny"),
    ("a story", "funny or scary or sad or happy"),
    ("a road", "straight or curvy"),
]

for thing, options in can_be:
    add(f"{thing} can be {options}", "fact", 1, "commonsense")

# --- templated "X verb Y at/in Z" sentences (bulk spatial) ---
random.seed(43)
subjects = ["the boy", "the girl", "the cat", "the dog", "the bird",
            "the baby", "the man", "the woman", "the child", "the rabbit"]
spatial_verbs = ["sits", "stands", "hides", "plays", "sleeps",
                 "runs", "jumps", "waits", "looks", "rests"]
spatial_locs = ["on the grass", "by the lake", "near the door",
                "under the tree", "on the bench", "behind the fence",
                "in the yard", "on the rock", "by the road",
                "next to the flowers", "in the shade", "on the path"]

for _ in range(500):
    subj = random.choice(subjects)
    verb = random.choice(spatial_verbs)
    loc = random.choice(spatial_locs)
    add(f"{subj} {verb} {loc}", "spatial", 1, "spatial")

# --- profession + tool + workplace combos ---
prof_tool_place = [
    ("a doctor", "a stethoscope", "a hospital"),
    ("a chef", "a knife and a pan", "a kitchen"),
    ("a teacher", "books and a board", "a school"),
    ("a firefighter", "a hose", "a fire station"),
    ("a farmer", "a tractor", "a farm"),
    ("a pilot", "a plane", "an airport"),
    ("a painter", "a brush and paint", "a studio"),
    ("a police officer", "a badge", "a police station"),
    ("a dentist", "a drill and mirror", "a dental office"),
    ("a carpenter", "a hammer and saw", "a workshop"),
    ("a librarian", "books", "a library"),
    ("a mechanic", "a wrench", "a garage"),
    ("a gardener", "a shovel and seeds", "a garden"),
    ("a baker", "an oven", "a bakery"),
    ("a fisherman", "a fishing rod", "a lake or the sea"),
    ("a musician", "an instrument", "a stage"),
    ("a scientist", "a microscope", "a lab"),
    ("an astronaut", "a space suit", "a space station"),
]

for prof, tool, place in prof_tool_place:
    add(f"{prof} uses {tool}", "fact", 1, "social")
    add(f"{prof} works at {place}", "fact", 1, "social")
    add(f"what tool does {prof} use", "qa", 2, "social", answer=tool)
    add(f"where does {prof} work", "qa", 2, "social", answer=place)

# --- "when you are sick you..." ---
sick_facts = [
    "you might have a fever",
    "you should rest in bed",
    "you should drink lots of water",
    "a doctor can help you feel better",
    "medicine can help you heal",
    "you should cover your mouth when you cough",
    "you should stay home so others do not get sick",
    "soup and rest are good for you",
    "your body fights germs to get better",
    "you might feel tired and weak",
]

for f in sick_facts:
    add(f"when you are sick {f}", "social", 1, "social")

# --- more detailed counting / math QA ---
number_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

for a_n in range(1, 11):
    for b_n in range(1, 11):
        s = a_n + b_n
        if s <= 20:
            add(f"what is {a_n} plus {b_n}", "qa", 2, "commonsense",
                answer=str(s))
        if a_n >= b_n:
            add(f"what is {a_n} minus {b_n}", "qa", 2, "commonsense",
                answer=str(a_n - b_n))

# --- final padding: combinatorial "X is next to Y" ---
random.seed(43)
room_objects = ["the lamp", "the clock", "the vase", "the photo",
                "the plant", "the radio", "the candle", "the mirror"]
furniture_list = ["the table", "the shelf", "the desk", "the couch",
                  "the bed", "the chair", "the counter", "the dresser"]

for _ in range(300):
    obj = random.choice(room_objects)
    furn = random.choice(furniture_list)
    prep = random.choice(["on", "next to", "near", "behind", "in front of"])
    add(f"{obj} is {prep} {furn}", "spatial", 1, "spatial")

# =====================================================================
# 9. FINAL PADDING: combinatorial templates to reach 10,000+
# =====================================================================

# --- "X is Y but Z is not" contrastive pairs ---
contrastive = [
    ("a bird", "can fly", "a penguin", "cannot fly"),
    ("a dog", "is a pet", "a wolf", "is wild"),
    ("ice", "is cold", "fire", "is hot"),
    ("day", "is bright", "night", "is dark"),
    ("a car", "has an engine", "a bicycle", "does not"),
    ("wood", "floats", "a rock", "sinks"),
    ("glass", "is see through", "wood", "is not"),
    ("sugar", "is sweet", "lemon", "is sour"),
    ("a rabbit", "hops", "a snake", "slithers"),
    ("a tree", "has leaves", "a cactus", "has spines"),
    ("the sun", "is a star", "the moon", "is not a star"),
    ("a frog", "starts as a tadpole", "a dog", "starts as a puppy"),
    ("a mammal", "has fur", "a fish", "has scales"),
    ("summer", "is hot", "winter", "is cold"),
    ("an eagle", "can see far", "a mole", "cannot see well"),
]

for a_thing, a_prop, b_thing, b_prop in contrastive:
    add(f"{a_thing} {a_prop} but {b_thing} {b_prop}", "fact", 2, "commonsense")

# --- more "what do you do at/when" combos ---
do_at_events = [
    ("a birthday party", ["eat cake", "sing happy birthday", "open presents",
                           "play games", "blow out candles"]),
    ("the beach", ["build sandcastles", "swim in the ocean", "collect shells",
                    "lay in the sun", "splash in the waves"]),
    ("school", ["read books", "write in notebooks", "listen to the teacher",
                "eat lunch", "play at recess"]),
    ("the park", ["swing on the swings", "slide down the slide",
                  "play catch", "run in the grass", "climb the monkey bars"]),
    ("bedtime", ["brush your teeth", "put on pajamas", "read a story",
                 "hug your parents", "close your eyes"]),
    ("a grocery store", ["push a cart", "pick out food", "pay at the counter",
                          "carry bags to the car"]),
    ("the dentist", ["sit in a chair", "open your mouth wide",
                     "get your teeth checked", "get a sticker"]),
    ("a rainy day inside", ["read a book", "draw pictures", "play board games",
                             "watch a movie", "build with blocks"]),
    ("a farm visit", ["see cows and chickens", "pet a horse",
                      "pick vegetables", "ride on a hay wagon"]),
    ("a camping trip", ["set up a tent", "build a campfire",
                        "cook marshmallows", "look at stars", "tell stories"]),
]

for event, acts in do_at_events:
    for act in acts:
        add(f"at {event} you {act}", "fact", 1, "social")
    add(f"what do you do at {event}", "qa", 2, "social",
        answer=" and ".join(acts[:2]))

# --- more name + action + time QA combos ---
random.seed(43)
did_actions = [
    ("drew a picture", "drew"), ("built a tower", "built"),
    ("ate a snack", "ate"), ("found a rock", "found"),
    ("made a card", "made"), ("sang a song", "sang"),
    ("told a joke", "told"), ("wrote a letter", "wrote"),
    ("caught a ball", "caught"), ("rode a bike", "rode"),
]
when_list = ["on monday", "on tuesday", "on wednesday", "on thursday",
             "on friday", "in the morning", "in the afternoon", "after school",
             "before lunch", "at recess"]

for _ in range(500):
    n1 = random.choice(names)
    n2 = random.choice([n for n in names if n != n1])
    a1, v1 = random.choice(did_actions)
    a2, v2 = random.choice([(a, v) for a, v in did_actions if a != a1])
    w = random.choice(when_list)
    add(f"{n1} {a1} {w}. {n2} {a2} {w}. what did {n1} do {w}",
        "qa", 3, "commonsense", answer=a1)

# --- "living vs nonliving" ---
living = ["dog", "tree", "bird", "fish", "flower", "cat", "person",
          "grass", "worm", "bee", "frog", "mushroom"]
nonliving = ["rock", "water", "car", "table", "ball", "shoe",
             "cup", "cloud", "pencil", "book", "sand", "door"]

for thing in living:
    a = a_an(thing)
    add(f"{a} is a living thing", "fact", 1, "biology")
    add(f"is {a} living or nonliving", "qa", 2, "biology", answer="living")

for thing in nonliving:
    a = a_an(thing)
    add(f"{a} is a nonliving thing", "fact", 1, "commonsense")
    add(f"is {a} living or nonliving", "qa", 2, "commonsense",
        answer="nonliving")

# --- recycling and environment ---
env_facts = [
    "we should recycle paper and plastic and glass",
    "recycling helps the earth",
    "trees give us clean air",
    "we should not litter",
    "picking up trash keeps the park clean",
    "turning off lights saves energy",
    "walking or biking is good for the planet",
    "plants clean the air we breathe",
    "we should use less water when we can",
    "animals need clean water to live",
    "pollution makes the air dirty",
    "planting trees helps the earth",
    "the earth is our home so we should take care of it",
    "we can reuse bags instead of throwing them away",
    "solar panels get energy from the sun",
    "wind turbines get energy from the wind",
]

for f in env_facts:
    add(f, "social", 2, "social")

# --- "X rhymes with Y" ---
rhymes = [
    ("cat", "hat"), ("dog", "log"), ("sun", "fun"), ("tree", "bee"),
    ("moon", "spoon"), ("star", "car"), ("rain", "train"), ("boat", "goat"),
    ("cake", "lake"), ("book", "cook"), ("hill", "fill"), ("day", "play"),
    ("fish", "wish"), ("bear", "chair"), ("mouse", "house"),
    ("frog", "fog"), ("night", "light"), ("king", "ring"),
    ("bed", "red"), ("snow", "grow"),
]

for w1, w2 in rhymes:
    add(f"{w1} rhymes with {w2}", "fact", 1, "commonsense")
    add(f"what rhymes with {w1}", "qa", 1, "commonsense", answer=w2)

# --- single word vocabulary definitions ---
vocab_words = [
    ("brave", "not afraid to do hard things"),
    ("curious", "wanting to know more"),
    ("gentle", "soft and careful"),
    ("enormous", "very very big"),
    ("tiny", "very very small"),
    ("delicious", "very tasty"),
    ("ancient", "very very old"),
    ("fragile", "easy to break"),
    ("fierce", "strong and scary"),
    ("patient", "able to wait without getting upset"),
    ("clever", "smart and quick thinking"),
    ("clumsy", "often dropping or bumping into things"),
    ("grumpy", "in a bad mood"),
    ("cheerful", "happy and bright"),
    ("enormous", "very large"),
    ("cozy", "warm and comfortable"),
    ("swift", "very fast"),
    ("sturdy", "strong and not easy to break"),
    ("slippery", "hard to hold onto"),
    ("fluffy", "soft and light like a cloud"),
]

for word, meaning in vocab_words:
    add(f"{word} means {meaning}", "fact", 1, "commonsense")
    add(f"what does {word} mean", "qa", 1, "commonsense", answer=meaning)

# --- "what comes in pairs" ---
pairs_things = ["shoes", "socks", "gloves", "earrings", "eyes", "ears",
                "hands", "feet", "wings on a bird", "wheels on a bicycle",
                "parents", "salt and pepper"]
for p in pairs_things:
    add(f"{p} come in pairs", "fact", 1, "commonsense")

# --- more combinatorial spatial: "the [color] [thing] is [prep] the [place]" ---
random.seed(43)
sc_colors = ["red", "blue", "green", "yellow", "brown", "white", "black"]
sc_things = ["cup", "ball", "book", "hat", "shoe", "toy", "bag", "pen"]
sc_preps = ["on", "under", "next to", "behind", "in front of", "near"]
sc_places = ["the table", "the chair", "the bed", "the door", "the window",
             "the shelf", "the box", "the couch"]

for _ in range(500):
    c = random.choice(sc_colors)
    t = random.choice(sc_things)
    p = random.choice(sc_preps)
    loc = random.choice(sc_places)
    text = f"the {c} {t} is {p} {loc}"
    add(text, "spatial", 1, "spatial")
    add(f"where is the {c} {t}. it is {p} {loc}",
        "qa", 2, "spatial", answer=f"{p} {loc}")

# ── final stats ──────────────────────────────────────────────────────
random.shuffle(entries)

# Deduplicate by text field
seen = set()
unique = []
for e in entries:
    key = e["text"].lower().strip()
    if key not in seen:
        seen.add(key)
        unique.append(e)

entries = unique

# ── write ─────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "text_commonsense.json")

with open(out_path, "w") as f:
    json.dump(entries, f, indent=1)

# Stats
from collections import Counter
cats = Counter(e["category"] for e in entries)
types = Counter(e["type"] for e in entries)
levels = Counter(e["level"] for e in entries)

print(f"Total entries: {len(entries)}")
print(f"\nBy category: {dict(sorted(cats.items()))}")
print(f"\nBy type:     {dict(sorted(types.items()))}")
print(f"\nBy level:    {dict(sorted(levels.items()))}")
