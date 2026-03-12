# Current State (as of session end ~step 579)

## What's Working
- Full learning loop: curriculum → teacher → encode → forward → FF update → growth → viz
- 247 training items: 141 images (20 categories) + 106 text concepts
- 3D visualization with force-directed layout, legend, auto-rotate
- Growth system: BUD, CONNECT, PRUNE, INSERT, EXTEND, DORMANT all functional
- Growth capped at 64 active clusters
- Checkpoint persistence across restarts (every 100 steps)
- Model answer (M) displayed alongside teacher answer (T) in dialogue
- WebSocket real-time updates: graph diffs, activations, positions, dialogue
- Image uploads via URL (single + bulk)
- "Talk to it" chat with decoded model output
- Positive/negative markers in dialogue (✓/✗)

## Current Graph Stats (last seen)
- Step: ~579
- Stage: 1
- Clusters: 68 (cap is 64, exceeded slightly before cap was added)
- Nodes: 544
- Edges: 2278
- Layers: 5

## Known Limitations / Future Work
- **Curriculum doesn't rescan disk** — `_load_stage_0()` runs once at init. New images added to disk while running aren't discovered until restart.
- **Decoder output is gibberish** — TextDecoder does nearest-neighbor in CLIP vocab. At ~580 steps the model hasn't differentiated enough for coherent output. Needs much more training.
- **All clusters same color (orange)** — Most clusters are "arbitration" type (low internal density, low external ratio). Cluster types should diversify as structure evolves.
- **No plasticity decay implemented** — PlasticitySchedule exists but doesn't meaningfully reduce plasticity over time.
- **Curiosity scoring is basic** — CuriosityScorer doesn't use model state effectively. Could drive more targeted curriculum selection.
- **Stages 2-4 not meaningfully different** — Stage 2 unlocks EXTEND, stages 3-4 have no behavioral changes. Just number filters for curriculum.
- **Edge count very high (2278)** — PRUNE threshold may need tuning. Currently: strength < 0.02 AND steps_since_activation > 300.
- **No video support** — VideoEncoder exists but no video items in curriculum.

## Stages Explained
- **Stage 0**: Pure exposure. All positive except every 3rd step (noise). Growth aggressive.
- **Stage 1**: Model vs teacher comparison. Positive if cosine similarity high, negative if low. Growth selective. **This is where real learning happens.**
- **Stage 2+**: Same as 1 but EXTEND can add new layers. Not meaningfully different yet.

## How to Run
```bash
# Full system (recommended)
cd /Users/ani/Frameo/Baby && bash start.sh

# Manual
cd backend && PYTHONUNBUFFERED=1 nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level warning > /tmp/baby-backend.log 2>&1 &
cd frontend && npm run dev

# Seed more training data
cd backend && python3 seed_data.py

# Check status
curl -s localhost:8000/status
curl -X POST localhost:8000/start
curl -X POST localhost:8000/stage -H 'Content-Type: application/json' -d '{"stage": 1}'
```

## Recommendation for Next Session
- Let it train at stage 1 for several hundred more steps
- Monitor model answers (M lines) for any improvement
- If model answers stay random after 1000+ steps, investigate:
  - Whether is_positive scoring is actually working (cosine similarity threshold)
  - Whether weight changes are happening (check delta_summary)
  - Whether the decoder is the bottleneck (it maps to CLIP vocab space which may not capture what the model learned)
