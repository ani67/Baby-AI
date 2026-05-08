"""Expand curriculum.json + curriculum_interleaved.json with the v0.5
post-commit deeper-corpus sources.

Reads the structured manifest below, produces:
  - curriculum.json: append new book steps before the self_examination tail
  - curriculum_interleaved.json: append new source entries to `sources`

Idempotent — re-running with sources already present is a no-op.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Domain weights for the interleaved curriculum (per spec).
DOMAIN_WEIGHTS = {
    "philosophy": 1.2,
    "history":    1.0,
    "literature": 1.2,
    "essays":     1.5,   # intimate voice — highest weight
    "science":    0.8,
    "eastern":    1.0,
    "mythology":  0.8,
}


# Each entry: (name, source_path, domain, gutenberg_id_or_None, wiki_category_or_None, notes)
SOURCES: list[tuple[str, str, str, int | None, str | None, str]] = [

    # ---- PHILOSOPHY — Ancient to Modern ----
    ("plato_republic",                   "data/books/plato_republic.txt",                  "philosophy", 1497, None, "justice, the ideal state, the allegory of the cave"),
    ("plato_symposium",                  "data/books/plato_symposium.txt",                 "philosophy", 1600, None, "love, beauty, the nature of eros"),
    ("plato_phaedo",                     "data/books/plato_phaedo.txt",                    "philosophy", 1658, None, "the soul, death, immortality"),
    ("aristotle_poetics",                "data/books/aristotle_poetics.txt",               "philosophy", 1974, None, "art, tragedy, catharsis, mimesis"),
    ("aristotle_politics",               "data/books/aristotle_politics.txt",              "philosophy", 6762, None, "the state, citizenship, justice in practice"),
    ("epictetus_enchiridion",            "data/books/epictetus_enchiridion.txt",           "philosophy", 45109, None, "stoic practice, what is in our control, freedom through acceptance"),
    ("lucretius_nature_of_things",       "data/books/lucretius_nature_of_things.txt",      "philosophy", 785, None, "atoms, void, death, nature without gods"),
    ("cicero_on_duties",                 "data/books/cicero_on_duties.txt",                "philosophy", 47001, None, "moral obligation, virtue in public life"),
    ("augustine_confessions",            "data/books/augustine_confessions.txt",           "philosophy", 3296, None, "self-examination, time, memory, God — first great autobiography"),
    ("descartes_meditations",            "data/books/descartes_meditations.txt",           "philosophy", 59, None, "doubt, the cogito, mind and body"),
    ("spinoza_ethics",                   "data/books/spinoza_ethics.txt",                  "philosophy", 3800, None, "God as nature, determinism, human freedom"),
    ("hume_enquiry_understanding",       "data/books/hume_enquiry_understanding.txt",      "philosophy", 9662, None, "causation, empiricism, limits of reason"),
    ("kant_prolegomena",                 "data/books/kant_prolegomena.txt",                "philosophy", 52821, None, "what can we know, synthetic a priori, the limits of metaphysics"),
    ("nietzsche_thus_spoke_zarathustra", "data/books/nietzsche_thus_spoke_zarathustra.txt","philosophy", 1998, None, "will to power, eternal recurrence, the overman — philosophy as literature"),
    ("nietzsche_beyond_good_evil",       "data/books/nietzsche_beyond_good_evil.txt",      "philosophy", 4363, None, "morality as construction, master and slave values"),
    ("schopenhauer_world_as_will",       "data/books/schopenhauer_world_as_will.txt",      "philosophy", 38427, None, "will as fundamental reality, suffering, art as escape"),
    ("mill_utilitarianism",              "data/books/mill_utilitarianism.txt",             "philosophy", 11224, None, "greatest happiness, consequentialism"),
    ("mill_on_liberty",                  "data/books/mill_on_liberty.txt",                 "philosophy", 34901, None, "individual freedom, tyranny of majority"),
    ("rousseau_social_contract",         "data/books/rousseau_social_contract.txt",        "philosophy", 46333, None, "general will, legitimacy of government"),
    ("wiki_continental_philosophy",      "data/wiki/continental_philosophy.txt",           "philosophy", None, "Continental_philosophy", ""),
    ("wiki_analytic_philosophy",         "data/wiki/analytic_philosophy.txt",              "philosophy", None, "Analytic_philosophy", ""),
    ("wiki_existentialism",              "data/wiki/existentialism.txt",                   "philosophy", None, "Existentialism", ""),
    ("wiki_eastern_philosophy",          "data/wiki/eastern_philosophy.txt",               "philosophy", None, "Eastern_philosophy", ""),

    # ---- HISTORY — Deep and Wide ----
    ("thucydides_peloponnesian_war",     "data/books/thucydides_peloponnesian_war.txt",    "history", 7142, None, "war, power, human nature — the first political realist"),
    ("suetonius_twelve_caesars",         "data/books/suetonius_twelve_caesars.txt",        "history", 6400, None, "power corrupts, personality shapes empire"),
    ("tacitus_annals",                   "data/books/tacitus_annals.txt",                  "history", 2364, None, "tyranny, resistance, the cost of empire"),
    ("gibbon_decline_fall_vol1",         "data/books/gibbon_decline_fall_vol1.txt",        "history", 890, None, "how civilizations collapse — the most important history book"),
    ("machiavelli_prince",               "data/books/machiavelli_prince.txt",              "history", 1232, None, "power, realism, the gap between how men live and how they ought"),
    ("tocqueville_democracy_america",    "data/books/tocqueville_democracy_america.txt",   "history", 816, None, "democracy, equality, tyranny of the majority in practice"),
    ("paine_rights_of_man",              "data/books/paine_rights_of_man.txt",             "history", 1120, None, "revolution, human rights, the legitimacy of government"),
    ("wiki_byzantine_empire",            "data/wiki/byzantine_empire.txt",                 "history", None, "Byzantine_Empire", ""),
    ("wiki_islamic_golden_age",          "data/wiki/islamic_golden_age.txt",               "history", None, "Islamic_Golden_Age", ""),
    ("wiki_renaissance",                 "data/wiki/renaissance.txt",                      "history", None, "Renaissance", ""),
    ("wiki_enlightenment",               "data/wiki/enlightenment.txt",                    "history", None, "Age_of_Enlightenment", ""),
    ("wiki_industrial_revolution",       "data/wiki/industrial_revolution.txt",            "history", None, "Industrial_Revolution", ""),
    ("wiki_colonialism",                 "data/wiki/colonialism.txt",                      "history", None, "Colonialism", ""),
    ("wiki_20th_century",                "data/wiki/20th_century.txt",                     "history", None, "20th_century", ""),

    # ---- LITERATURE AND POETRY — Voice and Register ----
    ("shakespeare_sonnets",              "data/books/shakespeare_sonnets.txt",             "literature", 1041, None, "love, time, mortality, beauty — compressed emotional language"),
    ("shakespeare_hamlet",               "data/books/shakespeare_hamlet.txt",              "literature", 1524, None, "doubt, action, mortality, the mind turned on itself"),
    ("shakespeare_king_lear",            "data/books/shakespeare_king_lear.txt",           "literature", 1532, None, "power, madness, love, the destruction of the self"),
    ("dante_inferno",                    "data/books/dante_inferno.txt",                   "literature", 8800, None, "sin, consequence, the moral architecture of the universe"),
    ("homer_iliad",                      "data/books/homer_iliad.txt",                     "literature", 2199, None, "war, glory, grief, the cost of pride — the first great story"),
    ("homer_odyssey",                    "data/books/homer_odyssey.txt",                   "literature", 1727, None, "home, wandering, identity across time"),
    ("virgil_aeneid",                    "data/books/virgil_aeneid.txt",                   "literature", 227, None, "duty, fate, the founding of civilization"),
    ("ovid_metamorphoses",               "data/books/ovid_metamorphoses.txt",              "literature", 21765, None, "transformation, desire, the instability of form"),
    ("dostoevsky_brothers_karamazov",    "data/books/dostoevsky_brothers_karamazov.txt",   "literature", 28054, None, "faith, doubt, free will, the problem of evil — the greatest novel"),
    ("tolstoy_anna_karenina",            "data/books/tolstoy_anna_karenina.txt",           "literature", 1399, None, "love, society, moral consequence, the inner life"),
    ("tolstoy_death_of_ivan_ilyich",     "data/books/tolstoy_death_of_ivan_ilyich.txt",    "literature", 600, None, "mortality, authenticity, the examined life — short and devastating"),
    ("chekhov_short_stories",            "data/books/chekhov_short_stories.txt",           "literature", 7986, None, "human smallness, compassion, irony, the unspoken"),
    ("kafka_metamorphosis",              "data/books/kafka_metamorphosis.txt",             "literature", 5200, None, "alienation, identity, the absurd"),
    ("melville_moby_dick",               "data/books/melville_moby_dick.txt",              "literature", 2701, None, "obsession, nature, meaning in the void"),
    ("austen_pride_prejudice",           "data/books/austen_pride_prejudice.txt",          "literature", 1342, None, "social observation, irony, the comedy of human self-deception"),
    ("dickens_great_expectations",       "data/books/dickens_great_expectations.txt",      "literature", 1400, None, "class, identity, the illusion of self-improvement"),
    ("hardy_tess",                       "data/books/hardy_tess.txt",                      "literature", 110, None, "fate, injustice, the cruelty of social structure"),
    ("whitman_leaves_of_grass",          "data/books/whitman_leaves_of_grass.txt",         "literature", 1322, None, "the self, democracy, the body, American voice — expansive"),
    ("keats_poems",                      "data/books/keats_poems.txt",                     "literature", 23684, None, "beauty, mortality, negative capability — the richest poetic vocabulary"),
    ("blake_poems",                      "data/books/blake_poems.txt",                     "literature", 574, None, "innocence and experience, vision, the tiger"),
    ("rumi_masnavi",                     "data/books/rumi_masnavi.txt",                    "literature", 51662, None, "love, longing, the divine — compressed spiritual register"),
    ("tao_te_ching",                     "data/books/tao_te_ching.txt",                    "literature", 216, None, "paradox, emptiness, the unnamed — 81 verses of compressed wisdom"),
    ("bhagavad_gita",                    "data/books/bhagavad_gita.txt",                   "literature", 2388, None, "duty, action without attachment, the self"),
    ("wiki_modernist_literature",        "data/wiki/modernist_literature.txt",             "literature", None, "Modernist_literature", ""),
    ("wiki_poetry",                      "data/wiki/poetry.txt",                           "literature", None, "Poetry", ""),

    # ---- ESSAYS AND LETTERS — Intimate Voice ----
    ("montaigne_essays",                 "data/books/montaigne_essays.txt",                "essays", 3600, None, "the first essayist — self-examination, uncertainty as virtue, the examined life"),
    ("bacon_essays",                     "data/books/bacon_essays.txt",                    "essays", 575, None, "truth, death, adversity, beauty — dense aphoristic wisdom"),
    ("emerson_essays_first_series",      "data/books/emerson_essays_first_series.txt",     "essays", 2944, None, "self-reliance, the over-soul, nature — American transcendentalism"),
    ("emerson_essays_second_series",     "data/books/emerson_essays_second_series.txt",    "essays", 2945, None, "experience, politics, character"),
    ("thoreau_walden",                   "data/books/thoreau_walden.txt",                  "essays", 205, None, "simplicity, nature, deliberate living — the examined life in practice"),
    ("virginia_woolf_room",              "data/books/virginia_woolf_room.txt",             "essays", 5786, None, "women, fiction, freedom of mind — essayistic and intimate"),
    ("orwell_essays",                    "data/books/orwell_essays.txt",                   "essays", 34409, None, "clarity, politics, the corruption of language — the most important modern essayist"),
    ("letters_keats",                    "data/books/letters_keats.txt",                   "essays", 35688, None, "negative capability, beauty, the poetic life — intimate first person"),
    ("letters_seneca_lucilius",          "data/books/letters_seneca_lucilius.txt",         "essays", 19942, None, "how to live, time, friendship — the most intimate ancient voice"),
    ("wiki_essay",                       "data/wiki/essay.txt",                            "essays", None, "Essays", ""),

    # ---- SCIENCE AND NATURE — Empirical Wonder ----
    ("darwin_origin_species",            "data/books/darwin_origin_species.txt",           "science", 1228, None, "natural selection, the tree of life — the most important scientific idea"),
    ("darwin_descent_of_man",            "data/books/darwin_descent_of_man.txt",           "science", 2300, None, "human evolution, morality as evolved"),
    ("faraday_forces_of_matter",         "data/books/faraday_forces_of_matter.txt",        "science", 14472, None, "wonder at physical forces — science as revelation"),
    ("huxley_mans_place_in_nature",      "data/books/huxley_mans_place_in_nature.txt",     "science", 2931, None, "humanity in the natural order"),
    ("wiki_consciousness",               "data/wiki/consciousness.txt",                    "science", None, "Consciousness", "the hard problem — connects to what the mind itself is"),
    ("wiki_cognitive_science",           "data/wiki/cognitive_science.txt",                "science", None, "Cognitive_science", ""),
    ("wiki_evolutionary_biology",        "data/wiki/evolutionary_biology.txt",             "science", None, "Evolutionary_biology", ""),
    ("wiki_physics",                     "data/wiki/physics.txt",                          "science", None, "Physics", ""),
    ("wiki_cosmology",                   "data/wiki/cosmology.txt",                        "science", None, "Physical_cosmology", "the scale and nature of the universe"),

    # ---- EASTERN THOUGHT ----
    ("confucian_analects",               "data/books/confucian_analects.txt",              "eastern", 4094, None, "virtue, social harmony, the superior person"),
    ("zhuangzi",                         "data/books/zhuangzi.txt",                        "eastern", 56956, None, "Taoist philosophy, transformation, the relativity of all things"),
    ("wiki_buddhist_philosophy",         "data/wiki/buddhist_philosophy.txt",              "eastern", None, "Buddhist_philosophy", ""),
    ("wiki_hindu_philosophy",            "data/wiki/hindu_philosophy.txt",                 "eastern", None, "Hindu_philosophy", ""),
    ("wiki_confucianism",                "data/wiki/confucianism.txt",                     "eastern", None, "Confucianism", ""),

    # ---- MYTHOLOGY AND SACRED TEXTS ----
    ("bullfinch_mythology",              "data/books/bullfinch_mythology.txt",             "mythology", 4928, None, "greek and roman myths retold — the stories underneath western thought"),
    ("egyptian_book_of_dead",            "data/books/egyptian_book_of_dead.txt",           "mythology", 7145, None, "death, the afterlife, judgment — oldest sacred text"),
    ("upanishads",                       "data/books/upanishads.txt",                      "mythology", 37016, None, "brahman, atman, the self as universal — oldest philosophy"),
    ("wiki_comparative_mythology",       "data/wiki/comparative_mythology.txt",            "mythology", None, "Comparative_mythology", ""),
    ("wiki_religion",                    "data/wiki/religion.txt",                         "mythology", None, "Religion", ""),
]


def build_book_step(name: str, source: str, domain: str,
                    gid: int | None, wiki: str | None, notes: str) -> dict:
    step: dict = {
        "name":   name,
        "source": source,
        "domain": domain,
        "type":   "book",
    }
    if gid is not None:
        step["gutenberg_id"] = gid
    if wiki is not None:
        step["wiki_category"] = wiki
    if notes:
        step["notes"] = notes
    return step


def build_interleaved_entry(name: str, source: str, domain: str,
                            notes: str) -> dict:
    weight = DOMAIN_WEIGHTS[domain]
    out: dict = {
        "name":   name,
        "source": source,
        "domain": domain,
        "weight": weight,
    }
    if notes:
        out["notes"] = notes
    return out


def merge_curriculum(curr_path: str) -> tuple[int, int]:
    """Insert new book steps into curriculum.json's sequence right before
    the existing self_examination step. Idempotent on `name`."""
    with open(curr_path, "r", encoding="utf-8") as f:
        curr = json.load(f)
    seq = curr["sequence"]
    existing_names = {s.get("name") for s in seq}

    # Find self_examination index — new entries land just before it so the
    # tail dialogue still runs after everything is ingested.
    idx = next(
        (i for i, s in enumerate(seq) if s.get("name") == "self_examination"),
        len(seq),
    )

    new_steps = []
    skipped = 0
    for (name, source, domain, gid, wiki, notes) in SOURCES:
        if name in existing_names:
            skipped += 1
            continue
        new_steps.append(build_book_step(name, source, domain, gid, wiki, notes))

    if new_steps:
        curr["sequence"] = seq[:idx] + new_steps + seq[idx:]
        with open(curr_path, "w", encoding="utf-8") as f:
            json.dump(curr, f, indent=2)
            f.write("\n")
    return len(new_steps), skipped


def merge_interleaved(path: str) -> tuple[int, int]:
    """Append new sources to curriculum_interleaved.json's `sources`.
    Idempotent on `name`."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sources = cfg["sources"]
    existing_names = {s.get("name") for s in sources}

    new_entries = []
    skipped = 0
    for (name, source, domain, _gid, _wiki, notes) in SOURCES:
        if name in existing_names:
            skipped += 1
            continue
        new_entries.append(build_interleaved_entry(name, source, domain, notes))

    if new_entries:
        cfg["sources"] = sources + new_entries
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
    return len(new_entries), skipped


def main() -> int:
    curr_path     = os.path.join(ROOT, "curriculum.json")
    inter_path    = os.path.join(ROOT, "curriculum_interleaved.json")

    n1, s1 = merge_curriculum(curr_path)
    print(f"[curriculum.json]            +{n1} new book steps  ({s1} already present)")
    n2, s2 = merge_interleaved(inter_path)
    print(f"[curriculum_interleaved.json] +{n2} new sources    ({s2} already present)")

    # Domain breakdown
    by_domain: dict[str, int] = {}
    for (_n, _s, d, *_rest) in SOURCES:
        by_domain[d] = by_domain.get(d, 0) + 1
    print()
    print(f"[manifest] {len(SOURCES)} new sources by domain:")
    for d in sorted(by_domain):
        w = DOMAIN_WEIGHTS[d]
        print(f"    {d:12s}  count={by_domain[d]:>2d}  weight={w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
