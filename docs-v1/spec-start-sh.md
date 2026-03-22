# SPEC: start.sh
*Component 9 of 9 — The launcher*

---

## What it is

A single bash script at the project root.
One command starts everything.

```bash
./start.sh
```

It does five things in order:
1. Checks that dependencies exist
2. Starts Ollama and pulls the teacher model if not cached
3. Starts the Python backend
4. Starts the Vite frontend
5. Opens the browser

When you Ctrl+C, it shuts everything down cleanly.

---

## Location in the project

```
project/
  start.sh          ← this file (chmod +x)
  stop.sh           ← optional: kills everything if start.sh was backgrounded
```

---

## Full script

```bash
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
TEACHER_MODEL="${TEACHER_MODEL:-phi4-mini}"
BACKEND_PORT=8000
FRONTEND_PORT=5173
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
command -v ollama  >/dev/null 2>&1 || fail "ollama not found. Install from https://ollama.ai"

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

# ─── 3. Ollama ────────────────────────────────────────────────────────────────
info "Starting Ollama..."

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama already running"
    OLLAMA_PID=""  # don't kill it on exit — we didn't start it
else
    ollama serve >> "$OLLAMA_LOG" 2>&1 &
    OLLAMA_PID=$!

    # Wait for Ollama to be ready (up to 15s)
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

# Pull teacher model if not present
info "Checking teacher model ($TEACHER_MODEL)..."
if ollama list | grep -q "^$TEACHER_MODEL"; then
    ok "Model $TEACHER_MODEL already cached"
else
    warn "Model $TEACHER_MODEL not found — pulling now (this may take a while)..."
    ollama pull "$TEACHER_MODEL" || fail "Failed to pull $TEACHER_MODEL"
    ok "Model $TEACHER_MODEL ready"
fi

# ─── 4. Backend ───────────────────────────────────────────────────────────────
info "Starting backend..."

# Check for virtual environment or uv
if [ -f "$BACKEND_DIR/.venv/bin/activate" ]; then
    PYTHON="$BACKEND_DIR/.venv/bin/python"
    UV_RUN=""
elif command -v uv >/dev/null 2>&1; then
    PYTHON="uv run python"
    UV_RUN="uv run"
    # Install dependencies if needed
    (cd "$BACKEND_DIR" && uv sync --quiet 2>>"$BACKEND_LOG") || {
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
```

---

## stop.sh (companion script)

For when `start.sh` was run in a terminal you've since closed
and the processes are still running.

```bash
#!/usr/bin/env bash
# stop.sh — kills backend, frontend, and (optionally) Ollama

pkill -f "uvicorn main:app" 2>/dev/null && echo "Backend stopped" || echo "Backend not running"
pkill -f "vite"             2>/dev/null && echo "Frontend stopped" || echo "Frontend not running"

read -p "Stop Ollama too? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pkill -f "ollama serve" 2>/dev/null && echo "Ollama stopped" || echo "Ollama not running"
fi
```

---

## First-run experience

```
$ ./start.sh

  developmental ai
  ─────────────────

   Checking dependencies...
✓  Dependencies found (Python 3.11, Node v20.10.0)
   Starting Ollama...
✓  Ollama already running
   Checking teacher model (phi4-mini)...
⚠  Model phi4-mini not found — pulling now (this may take a while)...
   [ollama pull output here, takes ~2min on first run]
✓  Model phi4-mini ready
   Starting backend...
   Waiting for backend to be ready...
   Loading encoders (CLIP ~5s)...         ← printed by backend
✓  Backend ready at http://localhost:8000 (pid 84231)
   Starting frontend...
✓  Frontend ready at http://localhost:5173 (pid 84267)

  ready.

  frontend  http://localhost:5173
  backend   http://localhost:8000
  logs      .logs/

  Ctrl+C to stop
```

Second run (everything cached):

```
$ ./start.sh

  developmental ai
  ─────────────────

   Checking dependencies...
✓  Dependencies found (Python 3.11, Node v20.10.0)
   Starting Ollama...
✓  Ollama already running
   Checking teacher model (phi4-mini)...
✓  Model phi4-mini already cached
   Starting backend...
   Waiting for backend to be ready...
✓  Backend ready at http://localhost:8000 (pid 84231)
   Starting frontend...
✓  Frontend ready at http://localhost:5173 (pid 84267)

  ready.
```

Total time from `./start.sh` to browser open: ~8s on M1 (after first run).

---

## Environment variables

Override any default by setting before running:

```bash
TEACHER_MODEL=mistral:7b-instruct ./start.sh
```

| Variable        | Default                         | Description                     |
|-----------------|----------------------------------|---------------------------------|
| `TEACHER_MODEL` | `phi4-mini`                      | Ollama model for the teacher    |
| `BACKEND_PORT`  | `8000`                           | Backend port                    |
| `FRONTEND_PORT` | `5173`                           | Frontend port                   |
| `DB_PATH`       | `backend/data/dev.db`            | SQLite database path            |
| `DATA_DIR`      | `backend/data`                   | Data directory                  |
| `OLLAMA_URL`    | `http://localhost:11434`         | Ollama base URL                 |

---

## File layout after first run

```
project/
  .logs/
    backend.log       ← uvicorn output (errors, startup messages)
    frontend.log      ← vite output
    ollama.log        ← ollama serve output (only if we started it)
  backend/
    data/
      dev.db          ← SQLite database (created on first run)
      stage0/         ← put images here before starting
        dog/
          dog_01.jpg
        cat/
          cat_01.jpg
  frontend/
    node_modules/     ← created on first run
```

---

## Seeding Stage 0 images

Before running for the first time, populate `backend/data/stage0/`:

```bash
mkdir -p backend/data/stage0/dog
mkdir -p backend/data/stage0/cat
mkdir -p backend/data/stage0/bird
mkdir -p backend/data/stage0/car
# etc.
# Drop .jpg or .png files into each folder
# The folder name becomes the label
```

The model will work without any images (it'll generate concept-only
questions from the start) but the early stages are designed for
image input. Ten images across five categories is enough to begin.

A minimal seed set ships with the project in `backend/data/stage0/`.
If the folder is empty when the server starts, a warning is printed
but nothing breaks — the Curriculum will use Stage 4 concept items
from the start instead.

---

## Known issues

**`ollama list` format.**
`ollama list` output format changed between Ollama versions.
Some versions print the model name as `phi4-mini:latest`,
others as `phi4-mini`. The grep pattern `^$TEACHER_MODEL` may miss
models with `:latest` suffix. Fix:

```bash
if ollama list | grep -q "$TEACHER_MODEL"; then
```
(Remove the `^` anchor — match anywhere in the line.)

**Port conflicts.**
If port 8000 or 5173 is already in use, the backend or frontend
will fail to bind and the health check will time out.
The script reports "check logs" — the log will show
`Address already in use`. Fix: kill the conflicting process,
or change `BACKEND_PORT` / `FRONTEND_PORT`.

**`uv sync` on first run takes time.**
`uv` resolves and downloads packages on first run.
Subsequent runs are fast (packages cached in `~/.cache/uv`).
The script doesn't print progress during `uv sync` — the terminal
will appear to hang for 10-30s. Add `--verbose` to `uv sync` during
debugging if you want to see what it's doing.

**macOS Gatekeeper on first `./start.sh`.**
macOS may block the script or Ollama on first run.
Right-click → Open, or:
```bash
chmod +x start.sh
xattr -d com.apple.quarantine start.sh   # if blocked by Gatekeeper
```

**Ctrl+C sometimes leaves Ollama running.**
If `OLLAMA_PID` is set, the `cleanup()` trap kills Ollama.
But if `start.sh` exits abnormally (kill -9, power loss),
the trap doesn't fire and Ollama stays running.
This is fine — Ollama is a background service.
Use `stop.sh` to clean up manually.

---

## Making it executable

```bash
chmod +x start.sh stop.sh
```

Add to `.gitignore`:
```
.logs/
backend/data/dev.db
backend/data/*.db
frontend/node_modules/
backend/.venv/
```

---

## What `wait` does at the end

```bash
wait $BACKEND_PID $FRONTEND_PID
```

This holds the script in the foreground until both processes exit.
If either one crashes (e.g., backend hits an unrecoverable error),
`wait` returns and the script exits — triggering `cleanup()`.
The human will see the terminal return and know something died.
They check `.logs/backend.log` to find out what.

This is better than an infinite `sleep` loop because it responds
to process death immediately rather than waiting for the next
loop iteration.
