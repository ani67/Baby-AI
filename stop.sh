#!/bin/bash
# Stop Kaida Reed — kill backend + frontend
DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/.kaida.pids"

DIM='\033[2m'
RESET='\033[0m'
BLUE='\033[0;34m'

if [ -f "$PIDFILE" ]; then
    while read -r pid; do
        kill "$pid" 2>/dev/null
    done < "$PIDFILE"
    rm -f "$PIDFILE"
fi

# Also kill by port as fallback
lsof -ti:8000 | xargs kill 2>/dev/null || true
lsof -ti:3000 | xargs kill 2>/dev/null || true

echo -e "${BLUE}kaida${RESET} stopped"
