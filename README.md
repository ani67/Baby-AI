# AI-Experiments

A continuous local AI system that learns and self-corrects on an M1 Mac. Built as 10 modular components inspired by biological brain architecture — episodic memory, importance scoring, sleep consolidation, internal state monitoring, and multi-teacher learning.

## What it does

A Llama-3.2-3B model runs locally via MLX with LoRA adapters. When you chat with it, it:

1. Retrieves relevant memories and facts to augment its response
2. Stores the interaction as an episodic memory
3. Scores the interaction's importance (like an amygdala)
4. Monitors its own uncertainty, novelty, and coherence (proto-self)
5. When uncertain, asks teacher models (Ollama phi4-mini + Gemini Flash) for training data
6. Periodically consolidates memories during "sleep" — pruning low-value episodes, strengthening important ones
7. Learns from corrections immediately — user feedback becomes both a fact and a training signal

All inference and training happen simultaneously via a double-buffer system. No downtime.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Component 10: Orchestrator                                  │
│  Ties everything together. chat() is the main entry point.   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ 1. Base      │  │ 2. LoRA     │  │ 3. Double-Buffer     │ │
│  │ Inference    │  │ Training    │  │ (train + infer       │ │
│  │ (Llama 3B)  │  │ (112 params)│  │  simultaneously)     │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ 4. Episodic  │  │ 5. Importance│ │ 6. Consolidation    │ │
│  │ Store       │  │ Scorer      │  │ ("Sleep")            │ │
│  │ (ChromaDB)  │  │ (Amygdala)  │  │ Prune + strengthen   │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ 7. State     │  │ 8. Teacher  │  │ 9. Knowledge Store   │ │
│  │ Monitor     │  │ Ensemble    │  │ (Facts, not weights)  │ │
│  │ (Proto-self)│  │ (Ollama +   │  │ Anti-hallucination    │ │
│  │             │  │  Gemini)    │  │                      │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Components

| # | Component | What it does |
|---|-----------|-------------|
| 1 | **Base Inference** | Loads Llama-3.2-3B via MLX. ~0.6s per response. |
| 2 | **LoRA Training** | 112 trainable parameters. Single-step fine-tuning from corrections. |
| 3 | **Double-Buffer** | Two LoRA adapters (A serves, B trains). Atomic swap. Metal lock for thread safety. |
| 4 | **Episodic Store** | ChromaDB + JSON persistence. Similarity retrieval. Survives restarts. |
| 5 | **Importance Scorer** | Scores interactions 0-1. Drives learning rate (0.1→1e-5, 1.0→5e-4). |
| 6 | **Consolidation** | "Sleep" cycle every 100 episodes or 24hrs. Prunes low-value memories, replays important ones. Safety: reverts if loss increases >10%. |
| 7 | **State Monitor** | Tracks uncertainty, novelty, coherence, confidence via ring buffers. Flags uncertainty > 0.7 to trigger teacher queries. |
| 8 | **Teacher Ensemble** | Ollama phi4-mini (local) + Gemini Flash (API). Consensus via word overlap. Rate-limited (10/hr). Background worker thread. |
| 9 | **Knowledge Store** | Separate fact store. User corrections → facts at 0.95 confidence. Augments prompts with relevant facts. |
| 10 | **Orchestrator** | `chat()` entry point. Retrieves context, runs inference, stores episode, triggers background learning. Self-narrative from state + history. |

## Stack

- **Model**: Llama-3.2-3B (~3.5GB) via MLX
- **Training**: LoRA via MLX-LM (112 parameters)
- **Memory**: ChromaDB (all-MiniLM-L6-v2 embeddings)
- **Teachers**: Ollama (phi4-mini) + Gemini Flash API
- **Backend**: FastAPI
- **Frontend**: Next.js
- **Tests**: pytest (82 tests)

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Start (launches backend + frontend)
bash start.sh

# Stop
bash stop.sh

# Run tests
pytest tests/
```

Requires: Python 3.11+, Node.js 18+, Ollama with phi4-mini model.

## Design Documents

The `Context/` folder contains the architectural thinking behind each component:

- `build-spec.md` — Master spec with component interfaces and session prompts
- `continuous-training-paradigm.md` — Why the freeze-then-fine-tune cycle is wrong
- `memory-and-wiring-problem.md` — Why memory needs episodic + knowledge layers
- `brain-architecture-scaling.md` — Biological grounding for importance and state monitoring
- `model-dialogue-training.md` — Why teacher ensemble beats search-based training
- `from-scratch-minimum-model.md` — Why the model is small and facts live outside weights
- `modality-emergence-self.md` — Self-narrative and multimodality roadmap
