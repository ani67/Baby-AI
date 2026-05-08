"""Run a list of prompts against a saved mind, with the conditioned
decoder active, printing each (decision, response) and writing JSONL.

Usage
-----
    python3 scripts/ask_questions.py --mind first --prompts "what are you,do you dream"
    python3 scripts/ask_questions.py --mind first --prompts-file prompts.txt --out data/first/dialogue_v05.jsonl

The script:
  - Loads the mind via MindPersistence
  - Forces a reload of the language head so the v2 (or v1) GPT-2 file is
    picked up
  - Reports which decoder was loaded
  - Runs each prompt with force_respond=True
  - Prints `[N] Q: <prompt>` then `     -> (<decision>) <surface>`
  - Writes one JSONL record per prompt to --out (default: stdout-only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.mind_paths import MindPaths            # noqa: E402
from backend.persistence import MindPersistence     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind", required=True)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--prompts",
                     help="comma-separated prompts (use a single bash-quoted arg)")
    grp.add_argument("--prompts-file",
                     help="path to a file with one prompt per line")
    ap.add_argument("--out", default=None, help="write JSONL records to this path")
    ap.add_argument("--examiner", default="examiner",
                    help="agent handle to register as the speaker")
    args = ap.parse_args()

    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        print("no prompts; nothing to do", file=sys.stderr)
        return 1

    paths = MindPaths(args.mind)
    print(f"[ask] loading mind '{args.mind}' from {paths.db}")
    loop = MindPersistence.load(paths.db)

    # Force the loader to pick up whichever language-head delta is on disk.
    # The Expression __init__ may have set a stale CD path if older save
    # formats are involved; re-point here just in case.
    loop.expression._lm_weights_path = paths.language_head
    loop.expression._lm_vocab_path = paths.vocab
    if hasattr(loop.expression, "_cd_root"):
        loop.expression._cd_root = os.path.dirname(paths.language_head) or "."
    if hasattr(loop.expression, "_cd_weights_path"):
        from backend.language_head import CONDITIONED_DECODER_FILENAME
        loop.expression._cd_weights_path = os.path.join(
            loop.expression._cd_root or ".", CONDITIONED_DECODER_FILENAME,
        )
    loop.expression.reload_language_head()

    cd_loaded   = loop.expression._conditioned_decoder is not None
    lstm_loaded = loop.expression._language_head is not None
    print(f"[ask] conditioned decoder loaded: {cd_loaded}")
    print(f"[ask] LSTM head loaded:           {lstm_loaded}")
    print(f"[ask] graph:                      {loop.graph.node_count} nodes,"
          f" {loop.graph.edge_count} edges")
    print()

    h = loop.input_pipeline
    examiner_id = h.register_agent(args.examiner, now=time.time())

    log_f = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        log_f = open(args.out, "w", encoding="utf-8")

    decisions_count: dict[str, int] = {}
    for i, prompt in enumerate(prompts, start=1):
        now = time.time()
        ing = h.ingest_text(prompt, agent_id=examiner_id, now=now)
        cyc = loop.cycle(ing, now=now + 1e-3, force_respond=True)
        expr = cyc.expression_decision
        surface = cyc.emitted_surface
        decision = type(expr).__name__ if expr else None
        decisions_count[decision or "None"] = decisions_count.get(decision or "None", 0) + 1
        rec = {
            "i":        i,
            "t":        now,
            "prompt":   prompt,
            "decision": decision,
            "answer":   surface,
        }
        print(f"[{i:>2d}] Q: {prompt!r}")
        print(f"      -> ({decision}) {surface!r}")
        if log_f:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    if log_f:
        log_f.close()
        print()
        print(f"[ask] wrote {len(prompts)} records to {args.out}")

    print()
    print(f"[ask] decision counts: {decisions_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
