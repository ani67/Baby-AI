#!/usr/bin/env bash
# stop.sh — kills backend, frontend, and (optionally) Ollama

pkill -f "uvicorn main:app" 2>/dev/null && echo "Backend stopped" || echo "Backend not running"
pkill -f "vite"             2>/dev/null && echo "Frontend stopped" || echo "Frontend not running"

read -p "Stop Ollama too? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pkill -f "ollama serve" 2>/dev/null && echo "Ollama stopped" || echo "Ollama not running"
fi
