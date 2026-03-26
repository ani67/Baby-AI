#!/usr/bin/env bash
set -e

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
DIM='\033[2m'
RESET='\033[0m'
BOLD='\033[1m'

ok()   { echo -e "${GREEN}✓${RESET}  $1"; }
info() { echo -e "${DIM}   $1${RESET}"; }
warn() { echo -e "${YELLOW}⚠${RESET}  $1"; }
fail() { echo -e "${RED}✗${RESET}  $1"; exit 1; }

# ─── Config ───────────────────────────────────────────────────────────────────
TEACHER_MODEL="${TEACHER_MODEL:-llava}"
BACKEND_PORT=8000
FRONTEND_PORT=5180
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
LOG_DIR=".logs"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
OLLAMA_LOG="$LOG_DIR/ollama.log"
HEALTH_URL="http://localhost:$BACKEND_PORT/health"
HEALTH_TIMEOUT=60    # seconds to wait for backend to be ready

# PIDs — tracked so Ctrl+C can kill them all
OLLAMA_PID=""
BACKEND_PID=""
FRONTEND_PID=""

# ─── Cleanup on exit ─────────────────────────────────────────────────────────
cleanup() {
    echo ""
    info "Shutting down..."

    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null && info "Frontend stopped"
    [ -n "$BACKEND_PID"  ] && kill "$BACKEND_PID"  2>/dev/null && info "Backend stopped"

    # Only stop Ollama if WE started it (not if it was already running)
    if [ -n "$OLLAMA_PID" ]; then
        kill "$OLLAMA_PID" 2>/dev/null
        info "Ollama stopped"
    fi

    info "Logs saved to $LOG_DIR/"
    echo ""
}
trap cleanup EXIT INT TERM

# ─── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  developmental ai${RESET}"
echo -e "${DIM}  ─────────────────${RESET}"
echo ""

# ─── 1. Dependency checks ─────────────────────────────────────────────────────
info "Checking dependencies..."

command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install Python 3.11+."
command -v node    >/dev/null 2>&1 || fail "node not found. Install Node.js 18+."
command -v npm     >/dev/null 2>&1 || fail "npm not found."
# Ollama only required for live/ollama curriculum
CURRICULUM_SOURCE="${CURRICULUM_SOURCE:-precomputed}"
if [ "$CURRICULUM_SOURCE" != "precomputed" ]; then
    command -v ollama  >/dev/null 2>&1 || fail "ollama not found. Install from https://ollama.ai"
fi

# Python version check (3.11+)
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
    fail "Python 3.11+ required. Found $PYTHON_VERSION."
fi

ok "Dependencies found (Python $PYTHON_VERSION, Node $(node --version))"

# ─── 2. Directory check ───────────────────────────────────────────────────────
[ -d "$BACKEND_DIR"  ] || fail "Backend directory '$BACKEND_DIR' not found. Run from project root."
[ -d "$FRONTEND_DIR" ] || fail "Frontend directory '$FRONTEND_DIR' not found. Run from project root."

mkdir -p "$LOG_DIR"
mkdir -p "$BACKEND_DIR/data/stage0"

# ─── 3. Ollama (only for live curriculum) ────────────────────────────────────
if [ "$CURRICULUM_SOURCE" != "precomputed" ]; then
    info "Starting Ollama..."

    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        ok "Ollama already running"
        OLLAMA_PID=""
    else
        ollama serve >> "$OLLAMA_LOG" 2>&1 &
        OLLAMA_PID=$!

        OLLAMA_READY=0
        for i in $(seq 1 30); do
            if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                OLLAMA_READY=1
                break
            fi
            sleep 0.5
        done

        if [ "$OLLAMA_READY" -eq 0 ]; then
            fail "Ollama failed to start. Check $OLLAMA_LOG"
        fi
        ok "Ollama started (pid $OLLAMA_PID)"
    fi

    info "Checking teacher model ($TEACHER_MODEL)..."
    if ollama list | grep -q "$TEACHER_MODEL"; then
        ok "Model $TEACHER_MODEL already cached"
    else
        warn "Model $TEACHER_MODEL not found — pulling now (this may take a while)..."
        ollama pull "$TEACHER_MODEL" || fail "Failed to pull $TEACHER_MODEL"
        ok "Model $TEACHER_MODEL ready"
    fi
else
    ok "Using precomputed embeddings — skipping Ollama"
fi

# ─── 4. Backend ───────────────────────────────────────────────────────────────
info "Starting backend..."

# Kill any stale process on the backend port
STALE_PID=$(lsof -ti:$BACKEND_PORT -sTCP:LISTEN 2>/dev/null | head -1)
if [ -n "$STALE_PID" ]; then
    warn "Killing stale process on port $BACKEND_PORT (pid $STALE_PID)"
    kill "$STALE_PID" 2>/dev/null
    sleep 1
fi

# Check for virtual environment or uv
if [ -f "$BACKEND_DIR/.venv/bin/activate" ]; then
    PYTHON="$BACKEND_DIR/.venv/bin/python"
    UV_RUN=""
elif command -v uv >/dev/null 2>&1 && [ -f "$BACKEND_DIR/pyproject.toml" ]; then
    PYTHON="uv run python"
    UV_RUN="uv run"
    # Install dependencies if needed
    (cd "$BACKEND_DIR" && uv sync --quiet 2>>"../$BACKEND_LOG") || {
        warn "uv sync had issues — check $BACKEND_LOG"
    }
else
    PYTHON="python3"
    UV_RUN=""
    # pip install if requirements.txt exists
    if [ -f "$BACKEND_DIR/requirements.txt" ]; then
        info "Installing Python dependencies..."
        pip install -q -r "$BACKEND_DIR/requirements.txt" --break-system-packages \
            2>>"$BACKEND_LOG" || warn "Some pip installs failed — check $BACKEND_LOG"
    fi
fi

# Start the backend
(
    cd "$BACKEND_DIR"
    export PYTHONUNBUFFERED=1
    if [ -n "$UV_RUN" ]; then
        uv run uvicorn main:app \
            --host 0.0.0.0 \
            --port "$BACKEND_PORT" \
            --log-level warning \
            >> "../$BACKEND_LOG" 2>&1
    else
        $PYTHON -m uvicorn main:app \
            --host 0.0.0.0 \
            --port "$BACKEND_PORT" \
            --log-level warning \
            >> "../$BACKEND_LOG" 2>&1
    fi
) &
BACKEND_PID=$!

# Wait for backend health endpoint
info "Waiting for backend to be ready..."
BACKEND_READY=0
for i in $(seq 1 $HEALTH_TIMEOUT); do
    if curl -s "$HEALTH_URL" >/dev/null 2>&1; then
        BACKEND_READY=1
        break
    fi
    # Check if process died
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        fail "Backend process died. Check $BACKEND_LOG"
    fi
    sleep 1
done

if [ "$BACKEND_READY" -eq 0 ]; then
    fail "Backend not ready after ${HEALTH_TIMEOUT}s. Check $BACKEND_LOG"
fi

ok "Backend ready at http://localhost:$BACKEND_PORT (pid $BACKEND_PID)"

# ─── 5. Frontend ──────────────────────────────────────────────────────────────
info "Starting frontend..."

# Install node_modules if needed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    info "Installing frontend dependencies (first run)..."
    (cd "$FRONTEND_DIR" && npm install --silent >> "../$FRONTEND_LOG" 2>&1) || \
        fail "npm install failed. Check $FRONTEND_LOG"
fi

(cd "$FRONTEND_DIR" && npm run dev >> "../$FRONTEND_LOG" 2>&1) &
FRONTEND_PID=$!

# Wait for Vite to be ready (it's fast — 3s max)
FRONTEND_READY=0
for i in $(seq 1 10); do
    if curl -s "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1; then
        FRONTEND_READY=1
        break
    fi
    sleep 0.5
done

if [ "$FRONTEND_READY" -eq 0 ]; then
    warn "Frontend may still be starting. Check $FRONTEND_LOG if browser is blank."
fi

ok "Frontend ready at http://localhost:$FRONTEND_PORT (pid $FRONTEND_PID)"

# ─── 6. Open browser ─────────────────────────────────────────────────────────
# macOS: open, Linux: xdg-open
if command -v open >/dev/null 2>&1; then
    open "http://localhost:$FRONTEND_PORT"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:$FRONTEND_PORT" &
fi

# ─── Ready ────────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${GREEN}${BOLD}ready.${RESET}"
echo ""
echo -e "  ${DIM}frontend${RESET}  http://localhost:$FRONTEND_PORT"
echo -e "  ${DIM}backend${RESET}   http://localhost:$BACKEND_PORT"
echo -e "  ${DIM}logs${RESET}      $LOG_DIR/"
echo ""
echo -e "  ${DIM}Ctrl+C to stop${RESET}"
echo ""

# ─── Wait ─────────────────────────────────────────────────────────────────────
# Hold until Ctrl+C or one of the processes dies
wait $BACKEND_PID $FRONTEND_PID
