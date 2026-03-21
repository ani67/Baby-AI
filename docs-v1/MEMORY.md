# Baby AI - Project Memory

## Overview
Curiosity-driven developmental AI system. FastAPI backend + React/Three.js frontend.
Forward-Forward learning (gradient-free), growing neural graph, teacher-student loop.

## Key Files Reference
See [architecture.md](./architecture.md) for detailed file map.
See [fixes-log.md](./fixes-log.md) for all bugs found and fixed.
See [current-state.md](./current-state.md) for where things stand.

## Quick Reference
- Start: `cd /Users/ani/Frameo/Baby && bash start.sh`
- Manual backend: `cd backend && PYTHONUNBUFFERED=1 python3 -m uvicorn main:app --host 0.0.0.0 --port 8000`
- Frontend: `cd frontend && npm run dev`
- Seed data: `cd backend && python3 seed_data.py`
- API: `curl -s localhost:8000/status`
- Checkpoint saves every 100 steps, restores on startup
- Growth capped at 64 active clusters
- Projection interval: 10 steps (config.py)
