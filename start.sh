#!/usr/bin/env bash
# Start both servers. Usage: ./start.sh
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data .logs

# 1. Backend on :8765
echo "[start] backend on http://127.0.0.1:8765"
PYTHONUNBUFFERED=1 python3 -m uvicorn backend.api:app \
  --host 127.0.0.1 --port 8765 --no-access-log \
  > .logs/backend.log 2>&1 &
BACK_PID=$!

cleanup() {
  echo
  echo "[start] stopping backend ($BACK_PID) and frontend ($FRONT_PID)…"
  kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait briefly for /state to respond before starting the frontend.
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -sf http://127.0.0.1:8765/state > /dev/null; then break; fi
  sleep 0.4
done

# 2. Frontend
cd frontend
if [ ! -d node_modules ]; then
  echo "[start] installing frontend deps (first run)…"
  npm install --silent
fi
echo "[start] frontend on http://127.0.0.1:5173"
npm run dev &
FRONT_PID=$!

wait "$FRONT_PID" "$BACK_PID"
