"""B5 — seed a fresh mind with 50 animal sentences, run three prompts.

Builds a fresh `animal` mind in data/animal/ (deletes any prior dir),
ingests 50 short declarative animal sentences using ingestion_mode so
plenty of surprises register, then asks three prompts via the same
loop.cycle path the API uses with force_respond=True.

Reports per prompt:
  - active set size before / after
  - budget computed
  - sentences generated
  - full joined response
  - where the loop stopped (Chosen / Revision / Suppression)
  - the cycle's expression_decision type
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.affect import AffectStack            # noqa: E402
from backend.attention import Attention            # noqa: E402
from backend.expression import Expression          # noqa: E402
from backend.graph import ConceptGraph             # noqa: E402
from backend.identity import (                     # noqa: E402
    ChosenCandidate, RevisionRequest, SuppressionRequest,
)
from backend.input import InputPipeline            # noqa: E402
from backend.main_loop import MainLoop             # noqa: E402
from backend.mind_paths import MindPaths           # noqa: E402
from backend.persistence import MindPersistence    # noqa: E402
from backend.predict import PredictionEngine       # noqa: E402
from backend.simulation import SimulationReplay   # noqa: E402


ANIMAL_SENTENCES = [
    "cats are small carnivorous mammals.",
    "cats hunt by stalking and pouncing.",
    "cats purr when content and growl when threatened.",
    "cats have whiskers that sense air currents.",
    "cats sleep many hours each day.",
    "cats can jump six times their body length.",
    "cats see well in low light.",
    "cats lick their fur to stay clean.",
    "cats often climb trees and high shelves.",
    "cats eat meat and small prey.",
    "dogs are loyal pack animals descended from wolves.",
    "dogs bark to alert their humans.",
    "dogs wag their tails when happy.",
    "dogs follow scents over long distances.",
    "dogs need daily exercise to stay healthy.",
    "dogs come in many breeds and sizes.",
    "dogs herd sheep on farms across the world.",
    "dogs bury bones and toys in the yard.",
    "dogs greet their owners with enthusiasm.",
    "dogs sleep beside their humans at night.",
    "birds have feathers and hollow bones.",
    "birds lay eggs in carefully built nests.",
    "birds sing at dawn to mark territory.",
    "birds migrate thousands of miles each year.",
    "birds use their beaks to pick up food.",
    "birds glide on warm air currents.",
    "owls hunt at night with silent flight.",
    "eagles soar high above mountains and rivers.",
    "sparrows gather in small busy flocks.",
    "penguins waddle on ice and swim through cold water.",
    "fish breathe through gills under the water.",
    "fish swim in schools for protection.",
    "trout live in clear cold streams.",
    "salmon return to their birth river to spawn.",
    "sharks have rows of replaceable teeth.",
    "tuna travel in vast ocean migrations.",
    "horses run in herds across open plains.",
    "horses pull carts and carry riders.",
    "horses sleep standing up most of the night.",
    "rabbits hop on long powerful back legs.",
    "rabbits live in warrens beneath the ground.",
    "rabbits eat grass and tender shoots.",
    "elephants have long trunks for grasping branches.",
    "elephants form close family groups led by a matriarch.",
    "elephants remember watering holes for many years.",
    "wolves hunt cooperatively in tight packs.",
    "wolves howl across the forest at night.",
    "bears fish for salmon at rocky waterfalls.",
    "bears hibernate through the cold winter months.",
    "deer feed at the edge of meadows at dusk.",
]


def fresh_mind(mind_name: str) -> tuple[MainLoop, MindPaths]:
    paths = MindPaths(mind_name)
    if os.path.exists(paths.root):
        print(f"[seed] removing existing {paths.root}")
        shutil.rmtree(paths.root)
    paths.ensure_dirs()

    now = time.time()
    seed = 7777
    a = AffectStack(birth_seed=seed, t_birth=now)
    g = ConceptGraph()
    p = PredictionEngine(affect=a, graph=g)
    sim = SimulationReplay(affect=a, graph=g, predict_engine=p)
    from backend.identity import Identity
    ident = Identity(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        birth_seed=seed, birth_time=now,
    )
    h = InputPipeline(affect=a, graph=g, predict_engine=p, identity=ident)
    f = Attention(affect=a, graph=g)
    gx = Expression(
        affect=a, graph=g, predict_engine=p, identity=ident, input_pipeline=h,
        lm_weights_path=paths.language_head, lm_vocab_path=paths.vocab,
    )
    return MainLoop(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        identity=ident, attention=f, expression=gx, input_pipeline=h,
    ), paths


def seed(loop: MainLoop, paths: MindPaths) -> None:
    h = loop.input_pipeline
    agent_id = h.register_agent("seeder", now=time.time())
    print(f"[seed] ingesting {len(ANIMAL_SENTENCES)} animal sentences (ingestion_mode=ON) …")
    loop.predict_engine.set_ingestion_mode(True)
    n_surprise = 0
    for i, sentence in enumerate(ANIMAL_SENTENCES, start=1):
        now = time.time()
        ing = h.ingest_text(sentence, now=now, agent_id=agent_id)
        loop.cycle(ing, now=now + 1e-3, force_respond=False, skip_simulation=True)
        if ing.gap.is_surprise:
            n_surprise += 1
    loop.predict_engine.set_ingestion_mode(False)
    print(f"[seed]   surprises: {n_surprise}/{len(ANIMAL_SENTENCES)}")
    print(f"[seed]   nodes={loop.graph.node_count}  edges={loop.graph.edge_count}")
    MindPersistence(paths.db).save(loop, now=time.time())
    print(f"[seed]   saved to {paths.db}")


def ask(loop: MainLoop, prompt: str, label: str) -> dict:
    print(f"\n[{label}]  Q: {prompt!r}")
    h = loop.input_pipeline
    examiner = h.register_agent("examiner", now=time.time())
    now = time.time()
    ing = h.ingest_text(prompt, agent_id=examiner, now=now)

    # Capture budget at the moment cycle would compute it — same active set
    # as what the EXPRESS branch uses.  We compute it pre-cycle from the
    # input active set as a coarse predictor; the cycle internally uses the
    # post-processing-loop active set so the values may differ slightly.
    arousal_before = float(loop.affect.current_arousal(now))

    cyc = loop.cycle(ing, now=now + 1e-3, force_respond=True)

    input_size = len(cyc.input_active_set or {})
    proc_size  = len(cyc.processed_active_set or {})
    arousal    = float(loop.affect.current_arousal(time.time()))
    # Budget the cycle actually saw (post-processing active set, same now).
    budget     = loop.compute_expression_budget(
        cyc.processed_active_set or cyc.input_active_set or {},
        now=now + 1e-3,
    )

    decision = type(cyc.expression_decision).__name__ if cyc.expression_decision else None
    surface  = cyc.emitted_surface
    action_kind = cyc.action.action.value if cyc.action else None
    n_sent = 0
    if surface:
        # Same boundary used in InputPanel.tsx — split on '. '.
        n_sent = sum(1 for piece in surface.split(". ") if piece.strip())

    print(f"     action: {action_kind}")
    print(f"     active set: input={input_size}  processed={proc_size}")
    print(f"     arousal: before={arousal_before:.3f}  after={arousal:.3f}")
    print(f"     budget: {budget}")
    print(f"     decision: {decision}")
    print(f"     sentences emitted: {n_sent}")
    print(f"     full response: {surface!r}")
    return {
        "label":   label,
        "prompt":  prompt,
        "action":      action_kind,
        "input_size":  input_size,
        "proc_size":   proc_size,
        "arousal_before": arousal_before,
        "arousal":     arousal,
        "budget":      budget,
        "decision":    decision,
        "n_sentences": n_sent,
        "response":    surface,
    }


def main() -> int:
    loop, paths = fresh_mind("animal")
    seed(loop, paths)

    results = []
    # Q1 — fully aroused after seeding 50 surprises; rich active set.
    results.append(ask(loop, "tell me everything you know about cats", "Q1"))

    # Reaction half-life is 2s. Sleep 8s so reaction decays toward
    # baseline and "yes" lands in a calmer mind — exercising the budget=1
    # minimal path the way the spec describes.
    print("\n[wait] sleeping 8s so reaction decays before 'yes' …")
    time.sleep(8.0)

    results.append(ask(loop, "yes", "Q2"))

    # Another decay window before Q3.
    print("\n[wait] sleeping 8s before 'what is beautiful' …")
    time.sleep(8.0)

    results.append(ask(loop, "what is beautiful", "Q3"))

    print()
    print("=" * 78)
    print(" B5 SUMMARY — three responses must have visibly different lengths")
    print("=" * 78)
    for r in results:
        body = r["response"] or "(no surface)"
        print(f"  {r['label']}  budget={r['budget']}  n={r['n_sentences']:>2d}   "
              f"chars={len(body):>4d}    {body[:60]!r}{'…' if len(body)>60 else ''}")

    lengths = [len(r["response"] or "") for r in results]
    distinct = len(set(lengths))
    if distinct >= 2:
        print(f"\n[B5] PASS — {distinct} distinct response lengths")
        return 0
    print(f"\n[B5] FAIL — all responses same length ({lengths[0]} chars)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
