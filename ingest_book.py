"""Feed a book into the mind, sentence by sentence.

Usage
-----
    python3 ingest_book.py path/to/book.txt
    python3 ingest_book.py path/to/book.txt --limit 200
    python3 ingest_book.py path/to/book.txt --reset

Pre-processing
--------------
- Strips Project-Gutenberg-style START/END boilerplate if present
- Drops lines that are clearly ALL-CAPS chapter titles, pure page
  numbers, or "Chapter X." / "PART II" markers
- Collapses whitespace and joins paragraph-internal newlines
- Splits into sentences on `.!?` followed by whitespace + capital
  letter, with a small abbreviation guard (Mr., Dr., e.g., …)
- Filters sentences shorter than 5 or longer than 40 words

Pipeline
--------
For each kept sentence:
    InputPipeline.ingest_text(text, agent=<source_file>, now)
    MainLoop.cycle(ingest_result, now, force_respond=False)

Every 500 sentences: loop.idle(now, max_replays=3)
Every 2000 sentences: loop.sleep(now, 10s) + persist.save()
Every 100 sentences: progress line printed

Resumable
---------
Progress lives in data/ingest_progress.json:
    {"source_file": "...", "sentences_processed": N, "total_sentences": M}
On restart with the same `source_file`, the script skips the first N kept
sentences and continues. With `--reset`, both progress and mind.db are
discarded.

The backend MUST be stopped while ingestion runs (both share data/mind.db).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.affect import AffectStack                     # noqa: E402
from backend.attention import Attention                     # noqa: E402
from backend.expression import Expression                   # noqa: E402
from backend.graph import ConceptGraph                      # noqa: E402
from backend.identity import Identity                       # noqa: E402
from backend.input import InputPipeline                     # noqa: E402
from backend.main_loop import MainLoop                      # noqa: E402
from backend.persistence import MindPersistence             # noqa: E402
from backend.predict import PredictionEngine                # noqa: E402
from backend.simulation import SimulationReplay             # noqa: E402


from backend.mind_paths import MindPaths     # noqa: E402

DEFAULT_MIND_NAME = "default"

PROGRESS_PRINT_EVERY = 100
IDLE_EVERY           = 500
SLEEP_EVERY          = 2000
SLEEP_DURATION_S     = 10.0
IDLE_REPLAYS         = 3

MIN_WORDS = 5
MAX_WORDS = 40


# ============================================================
# Pre-processing
# ============================================================

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF.+?\*\*\*", re.DOTALL)
_GUTENBERG_END   = re.compile(r"\*\*\*\s*END OF.+?\*\*\*",   re.DOTALL)
_CHAPTER_LINE    = re.compile(r"^\s*(chapter|part|book|section)\s+[ivxlcdm0-9]+", re.IGNORECASE)
_ROMAN_LINE      = re.compile(r"^\s*[IVX]+\.?\s*$")
_PAGE_NUMBER     = re.compile(r"^\s*\d{1,4}\s*$")

_ABBREVS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "co", "inc",
    "ltd", "vs", "etc", "e.g", "i.e", "cf", "no", "vol", "ch", "p",
    "pp", "fig", "viz", "ed",
}

# rfind "the last token before a sentence-ending dot" check — used to
# stitch back sentences that were split across a known abbreviation.
_LAST_WORD_RE = re.compile(r"([A-Za-z.]+)\.?\s*$")


def strip_book(raw: str) -> str:
    text = raw.replace("\r\n", "\n")
    # Strip Gutenberg boilerplate.
    m_start = _GUTENBERG_START.search(text)
    m_end   = _GUTENBERG_END.search(text)
    if m_start:
        text = text[m_start.end():]
    if m_end:
        text = text[:m_end.start()]

    # Line-by-line filter for chapter markers, page numbers, all-caps titles.
    kept_lines: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            kept_lines.append("")
            continue
        if _CHAPTER_LINE.match(s):
            continue
        if _ROMAN_LINE.match(s):
            continue
        if _PAGE_NUMBER.match(s):
            continue
        # Heuristic: ALL-CAPS lines under ~60 chars are titles.
        if s == s.upper() and len(s) < 60 and any(c.isalpha() for c in s):
            continue
        kept_lines.append(s)
    cleaned = "\n".join(kept_lines)

    # Paragraphs are separated by blank lines; within a paragraph, fold
    # newlines into spaces so sentence splits work cleanly.
    paragraphs = re.split(r"\n\s*\n", cleaned)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p)


def split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation + whitespace + capital letter, with
    an abbreviation guard: if the previous sentence ended on a known
    abbreviation, stitch it back to the next.
    """
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", text)
    out: list[str] = []
    for piece in pieces:
        s = piece.strip()
        if not s:
            continue
        if out:
            m = _LAST_WORD_RE.search(out[-1])
            if m and m.group(1).rstrip(".").lower() in _ABBREVS:
                out[-1] = out[-1] + " " + s
                continue
        out.append(s)
    return out


def keep_sentence(s: str) -> bool:
    n = len(s.split())
    return MIN_WORDS <= n <= MAX_WORDS


def load_sentences(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = strip_book(raw)
    sentences = split_sentences(text)
    return [s for s in sentences if keep_sentence(s)]


# ============================================================
# Mind construction / restoration
# ============================================================

def construct_mind(birth_seed: int = 42) -> MainLoop:
    now = time.time()
    a = AffectStack(birth_seed=birth_seed, t_birth=now)
    g = ConceptGraph()
    p = PredictionEngine(affect=a, graph=g)
    sim = SimulationReplay(affect=a, graph=g, predict_engine=p)
    ident = Identity(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        birth_seed=birth_seed, birth_time=now,
    )
    h = InputPipeline(affect=a, graph=g, predict_engine=p, identity=ident)
    f = Attention(affect=a, graph=g)
    gx = Expression(
        affect=a, graph=g, predict_engine=p, identity=ident, input_pipeline=h,
    )
    return MainLoop(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        identity=ident, attention=f, expression=gx, input_pipeline=h,
    )


# ============================================================
# Progress persistence
# ============================================================

def load_progress(paths: MindPaths, source_file: str) -> int:
    if not os.path.exists(paths.ingest_progress):
        return 0
    try:
        with open(paths.ingest_progress, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0
    if data.get("source_file") != source_file:
        return 0
    return int(data.get("sentences_processed", 0))


def save_progress(paths: MindPaths, source_file: str, processed: int, total: int) -> None:
    os.makedirs(os.path.dirname(paths.ingest_progress) or ".", exist_ok=True)
    payload = {
        "source_file": source_file,
        "sentences_processed": processed,
        "total_sentences": total,
    }
    tmp = paths.ingest_progress + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, paths.ingest_progress)


# ============================================================
# Main loop
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="path to plain text book")
    ap.add_argument("--mind", default=DEFAULT_MIND_NAME,
                    help="mind name (paths under data/{mind}/)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max sentences to process this run (after resume offset)")
    ap.add_argument("--reset", action="store_true",
                    help="discard the mind's db and progress before starting")
    ap.add_argument("--no-sleep", action="store_true",
                    help="skip the periodic sleep envelope")
    args = ap.parse_args()

    paths = MindPaths(args.mind)
    paths.ensure_dirs()

    source_file = os.path.abspath(args.file)
    if not os.path.exists(source_file):
        print(f"file not found: {source_file}", file=sys.stderr)
        return 1

    if args.reset:
        for p in (paths.db, paths.ingest_progress, paths.surprised_log):
            if os.path.exists(p):
                os.remove(p)
                print(f"removed {p}")

    print(f"loading sentences from {source_file} …")
    sentences = load_sentences(source_file)
    total = len(sentences)
    print(f"  {total:,} sentences kept (after filter {MIN_WORDS}-{MAX_WORDS} words)")

    skip = load_progress(paths, source_file)
    if skip > 0:
        print(f"  resuming from sentence {skip:,}")

    # Construct or restore mind.
    if os.path.exists(paths.db):
        print(f"loading mind from {paths.db} …")
        loop = MindPersistence.load(paths.db)
        print(f"  restored: {loop.graph.node_count} nodes, "
              f"{loop.graph.edge_count} edges, cycle_count={loop.cycle_count}")
    else:
        print(f"constructing fresh mind '{args.mind}' …")
        loop = construct_mind()

    persist = MindPersistence(paths.db)

    # Register the source as an agent so refers_to chains anchor cleanly.
    agent_handle = os.path.splitext(os.path.basename(source_file))[0][:32]
    agent_id = loop.input_pipeline.register_agent(agent_handle, now=time.time())
    print(f"  source agent: {agent_handle} (cid={agent_id})")

    # Establish baselines for "this run" reporting.
    nodes_before = loop.graph.node_count
    surprises_before = loop.predict_engine.surprise_count

    end = total if args.limit is None else min(total, skip + args.limit)
    if skip >= end:
        print("nothing to do — already past the requested cut")
        return 0

    # Open the training-corpus log in append mode. Each surprised sentence
    # gets a JSONL record: {sentence, affect, surprise_score, concept_id, t}.
    # Phase 5 LM trains on this — "the language head learns from the same
    # signal as the mind."
    os.makedirs(os.path.dirname(paths.surprised_log) or ".", exist_ok=True)
    train_log = open(paths.surprised_log, "a", encoding="utf-8")

    # Also stream the kept-sentence list to paths.book_text_log for vocab building.
    # Truncated and rewritten on every fresh run (no resume offset).
    if skip == 0:
        with open(paths.book_text_log, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

    print(f"\ningesting sentences {skip:,} … {end:,} of {total:,}\n")
    t_started = time.perf_counter()

    # Phase 7: bulk ingestion uses the relaxed 1.0σ surprise threshold.
    # Restored to default in the finally branches so live conversation
    # afterward uses the calibrated 1.5σ default.
    loop.predict_engine.set_ingestion_mode(True)
    print("  ingestion_mode=ON (1.0σ surprise threshold)")

    try:
        for i in range(skip, end):
            sentence = sentences[i]
            now = time.time()
            ingest = loop.input_pipeline.ingest_text(
                sentence, agent_id=agent_id, now=now,
            )
            cycle_result = loop.cycle(ingest, now=now + 1e-3, force_respond=False)

            # Log surprised sentences (for LM training corpus).
            if ingest.gap.is_surprise:
                composite = loop.affect.composite(now)
                train_log.write(json.dumps({
                    "sentence":       sentence,
                    "affect":         [float(x) for x in composite],
                    "surprise_score": float(ingest.gap.surprise_score),
                    "concept_id":     ingest.gap.concept_id,
                    "t":              float(now),
                }) + "\n")
                train_log.flush()

            n_done = i + 1

            if n_done % PROGRESS_PRINT_EVERY == 0:
                arousal = loop.affect.current_arousal(time.time())
                print(
                    f"  [{n_done:>6,d} / {total:,}]  "
                    f"nodes={loop.graph.node_count:>5d}  "
                    f"edges={loop.graph.edge_count:>5d}  "
                    f"surprises={loop.predict_engine.surprise_count:>5d}  "
                    f"arousal={arousal:.3f}"
                )

            if n_done % IDLE_EVERY == 0:
                replayed = loop.idle(now=time.time(), max_replays=IDLE_REPLAYS)
                print(f"    [idle] replayed {replayed.replayed_count}")

            if (not args.no_sleep) and n_done % SLEEP_EVERY == 0:
                slept = loop.sleep(now=time.time(), duration_seconds=SLEEP_DURATION_S)
                print(
                    f"    [sleep] replays={slept.replays_fired} "
                    f"abstractions={slept.abstractions_formed} "
                    f"duration={slept.duration_actual:.2f}s"
                )
                persist.save(loop, now=time.time())
                save_progress(paths, source_file, n_done, total)
                print(f"    [save] mind.db + progress at sentence {n_done:,}")
    except KeyboardInterrupt:
        print("\ninterrupted — saving before exit …")
        loop.predict_engine.set_ingestion_mode(False)
        persist.save(loop, now=time.time())
        save_progress(paths, source_file, i, total)
        print(f"[save] mind.db + progress at sentence {i:,}")
        train_log.close()
        return 130

    loop.predict_engine.set_ingestion_mode(False)
    train_log.close()

    # Final save.
    persist.save(loop, now=time.time())
    save_progress(paths, source_file, end, total)

    # Report.
    duration = time.perf_counter() - t_started
    nodes_added = loop.graph.node_count - nodes_before
    surprises_added = loop.predict_engine.surprise_count - surprises_before
    density = (loop.graph.edge_count / max(1, loop.graph.node_count))
    print()
    print(f"ingested {end - skip:,} sentences in {duration:.1f}s "
          f"({(end - skip) / max(duration, 1e-6):.1f}/s)")
    print(f"  concepts written this run     : {nodes_added}")
    print(f"  surprises this run            : {surprises_added}")
    print(f"  total graph: {loop.graph.node_count} nodes, "
          f"{loop.graph.edge_count} edges  (density {density:.2f} edges/node)")
    print(f"  pin count                     : {loop.graph.pin_count}")
    print(f"  replay buffer                 : {loop.simulation.buffer_size}")

    # Top 10 by activation_count, excluding boot anchors.
    excluded_names = {"self", "unknown"}
    interesting = [
        n for n in loop.graph.nodes.values()
        if n.name not in excluded_names
        and not n.name.startswith("tpl.")
        and not n.name.startswith("agent:")
        and not n.name.startswith("abstraction:")
    ]
    interesting.sort(key=lambda n: -n.activation_count)
    print(f"\n  top 10 most-activated concepts:")
    for n in interesting[:10]:
        print(f"    cid={n.concept_id:<5d} act={n.activation_count:<3d}  {n.name!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
