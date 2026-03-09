#!/bin/bash
# Start Kaida Reed — backend + frontend, single command
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Colors
DIM='\033[2m'
RESET='\033[0m'
BLUE='\033[0;34m'

echo -e "${BLUE}kaida${RESET} starting..."

# Kill anything already running on our ports
lsof -ti:8000 | xargs kill 2>/dev/null || true
lsof -ti:3000 | xargs kill 2>/dev/null || true
sleep 1

# PID file for stop.sh
PIDFILE="$DIR/.kaida.pids"
rm -f "$PIDFILE"

# Activate venv and start backend
source "$DIR/.venv/bin/activate"
uvicorn app:app --port 8000 2>&1 | sed "s/^/${DIM}[backend]${RESET} /" &
BACKEND_PID=$!
echo "$BACKEND_PID" >> "$PIDFILE"

# Start frontend
cd "$DIR/ui"
npm run dev -- --port 3000 2>&1 | sed "s/^/${DIM}[frontend]${RESET} /" &
FRONTEND_PID=$!
echo "$FRONTEND_PID" >> "$PIDFILE"

cd "$DIR"

# Wait for backend to be ready
echo -e "${DIM}waiting for backend...${RESET}"
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/status > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Wait for frontend to be ready
echo -e "${DIM}waiting for frontend...${RESET}"
for i in $(seq 1 30); do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo -e "${BLUE}kaida${RESET} ready at ${BLUE}http://localhost:3000${RESET}"
echo -e "${DIM}backend: http://localhost:8000  |  press ctrl+c to stop${RESET}"

# Open browser
open http://localhost:3000 2>/dev/null || true

# Cleanup on Ctrl+C
cleanup() {
    echo ""
    echo -e "${DIM}shutting down...${RESET}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    rm -f "$PIDFILE"
    echo -e "${BLUE}kaida${RESET} stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
