"""FastAPI backend for the Continuous Local AI system."""

import sys
import time

sys.path.insert(0, "src")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import component3
import component10


@asynccontextmanager
async def lifespan(app: FastAPI):
    component10.start()
    yield
    component10.stop()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str


class CorrectionRequest(BaseModel):
    prompt: str
    correction: str


@app.post("/chat")
def chat(req: ChatRequest):
    t0 = time.time()
    response = component10.chat(req.prompt)
    elapsed_ms = round((time.time() - t0) * 1000)
    return {"response": response, "elapsed_ms": elapsed_ms}


@app.get("/status")
def status():
    s = component10.get_system_status()
    return {
        "narrative": s.self_narrative,
        "episodes_stored": s.episodes_stored,
        "facts_stored": s.facts_stored,
        "training_queue_size": s.training_queue_size,
        "uptime_seconds": round(s.uptime_seconds, 1),
    }


@app.post("/correct")
def correct(req: CorrectionRequest):
    component3.submit_correction(req.prompt, req.correction)
    return {"ok": True}


@app.get("/history")
def history():
    return [
        {"prompt": p, "response": r}
        for p, r in component10._conversation_history
    ]


@app.get("/sessions")
def list_sessions():
    return {
        "sessions": component10.get_sessions(),
        "current": component10._current_session_id,
    }


@app.post("/sessions/new")
def new_session():
    session_id = component10.new_session()
    return {"id": session_id}


@app.post("/sessions/{session_id}/switch")
def switch_session(session_id: str):
    found = component10.switch_session(session_id)
    if not found:
        return {"ok": False, "error": "session not found"}
    return {"ok": True}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    found = component10.delete_session(session_id)
    if not found:
        return {"ok": False, "error": "session not found"}
    return {"ok": True, "current": component10._current_session_id}


@app.get("/sleep-status")
def sleep_status():
    return component10.get_sleep_status()


@app.post("/sleep")
def trigger_sleep():
    import threading
    threading.Thread(target=component10._run_sleep_cycle, daemon=True).start()
    return {"ok": True}
