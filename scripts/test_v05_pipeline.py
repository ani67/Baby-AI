"""Final v0.5 pipeline test.

Per spec:
  - Start API with mind 'first' and v2 language head active.
  - Ask 5 core questions via POST /ingest with force_respond=True.
  - Print budget, sentence count, and exact full response for each.

Procedure:
  1. Spawn `python -m uvicorn backend.api:app` on port 8765 with
     MIND_NAME=first.
  2. Poll /state until ready (or 60s timeout).
  3. POST each question to /ingest.
  4. Save per-question payload (budget, n_sentences, decision, surface)
     to data/first/dialogue_v05_pipeline.jsonl.
  5. Print the report.
  6. Tear down the API.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PROMPTS = [
    "what are you",
    "what is beautiful",
    "do you dream",
    "what do you want",
    "what is a story",
]


def wait_until_ready(url: str, timeout_s: float = 60.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(0.4)
    return False


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        return json.loads(resp.read())


def main() -> int:
    log_path = os.path.join(ROOT, ".logs/v05_pipeline_api.log")
    out_path = os.path.join(ROOT, "data/first/dialogue_v05_pipeline.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MIND_NAME"] = "first"

    # Start API in background.
    print("[v05-pipeline] launching API …")
    log_f = open(log_path, "w", encoding="utf-8")
    api_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "backend.api:app",
            "--host", "127.0.0.1", "--port", "8765", "--no-access-log",
        ],
        cwd=ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT,
    )

    try:
        # Wait until /state responds.
        ready = wait_until_ready("http://127.0.0.1:8765/state", timeout_s=90.0)
        if not ready:
            print("[v05-pipeline] FAIL — API never came up; tail of log:")
            log_f.flush()
            with open(log_path, "r", encoding="utf-8") as f:
                tail = f.read()[-2000:]
            print(tail)
            return 1

        # Quick state probe.
        with urllib.request.urlopen("http://127.0.0.1:8765/state", timeout=5.0) as resp:
            state = json.loads(resp.read())
        print(f"[v05-pipeline] mind ready  nodes={state['node_count']}  "
              f"edges={state['edge_count']}  cycles={state['cycle_count']}")

        results: list[dict] = []
        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, prompt in enumerate(PROMPTS, start=1):
                payload = {
                    "text": prompt,
                    "agent_handle": "examiner",
                    "force_respond": True,
                }
                t0 = time.perf_counter()
                resp = post_json("http://127.0.0.1:8765/ingest", payload)
                dt = time.perf_counter() - t0

                rec = {
                    "i":         i,
                    "t":         time.time(),
                    "prompt":    prompt,
                    "action":    (resp.get("action") or {}).get("kind"),
                    "arousal":   resp.get("arousal"),
                    "input_set_size":     len(resp.get("active_set") or {}),
                    "processed_set_size": resp.get("processed_active_set_size"),
                    "budget":             resp.get("budget"),
                    "n_sentences":        resp.get("n_sentences"),
                    "decision":           (resp.get("expression") or {}).get("type"),
                    "emitted_surface":    resp.get("emitted_surface"),
                    "expression_gap":     (resp.get("expression") or {}).get("expression_gap"),
                    "duration_s":         dt,
                }
                results.append(rec)
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

                surface = rec["emitted_surface"]
                print()
                print(f"[Q{i}] {prompt!r}    ({dt:.2f}s)")
                print(f"      action={rec['action']}  arousal={rec['arousal']:.3f}  "
                      f"input_set={rec['input_set_size']}  proc_set={rec['processed_set_size']}")
                print(f"      budget={rec['budget']}  decision={rec['decision']}  "
                      f"sentences={rec['n_sentences']}")
                if surface:
                    sentences = [s for s in surface.split(". ") if s.strip()]
                    if len(sentences) <= 1:
                        print(f"      response: {surface!r}")
                    else:
                        for j, s in enumerate(sentences, start=1):
                            print(f"      [{j}] {s!r}")
                else:
                    print(f"      response: (no surface)")

        print()
        print("=" * 78)
        print(f" v0.5 PIPELINE — {len(results)} questions through POST /ingest")
        print("=" * 78)
        for r in results:
            body = r["emitted_surface"] or "(no surface)"
            preview = (body[:64] + "…") if len(body) > 64 else body
            print(f"  Q{r['i']}  budget={r['budget']}  n={r['n_sentences']:>2d}  "
                  f"dec={r['decision']:>10s}    {preview!r}")
        print()
        print(f"[v05-pipeline] saved {len(results)} records to {out_path}")
        print(f"[v05-pipeline] API log:  {log_path}")
        return 0

    finally:
        try:
            api_proc.send_signal(signal.SIGTERM)
            api_proc.wait(timeout=5.0)
        except Exception:
            api_proc.kill()
            api_proc.wait(timeout=5.0)
        log_f.close()


if __name__ == "__main__":
    sys.exit(main())
