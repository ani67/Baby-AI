"""
Tests for Backend API — uses FastAPI TestClient with mock components.
"""

import os
import sys
import io

# Enable test mode before importing app — uses mock components (no CLIP, no Ollama)
os.environ["TESTING"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app


def make_test_jpeg() -> bytes:
    """Create a minimal valid JPEG in memory."""
    from PIL import Image
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# Use a module-level client so lifespan runs once
_client = None


def get_client():
    global _client
    if _client is None:
        _client = TestClient(app, raise_server_exceptions=True)
        _client.__enter__()
    return _client


def teardown_module():
    global _client
    if _client is not None:
        _client.__exit__(None, None, None)
        _client = None


# ── Tests ──


def test_health():
    client = get_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("PASS: test_health")


def test_status_returns_idle():
    client = get_client()
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"
    print("PASS: test_status_returns_idle")


def test_step_returns_step_result():
    client = get_client()
    r = client.post("/step")
    assert r.status_code == 200
    body = r.json()
    assert "question" in body
    assert "answer" in body
    print("PASS: test_step_returns_step_result")


def test_stage_out_of_range():
    client = get_client()
    r = client.post("/stage", json={"stage": 99})
    assert r.status_code == 400
    print("PASS: test_stage_out_of_range")


def test_chat_returns_string():
    client = get_client()
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 200
    assert isinstance(r.json()["message"], str)
    print("PASS: test_chat_returns_string")


def test_image_upload_jpeg():
    client = get_client()
    img_bytes = make_test_jpeg()
    r = client.post(
        "/image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"label": "test"},
    )
    assert r.status_code == 200
    assert r.json()["label"] == "test"
    print("PASS: test_image_upload_jpeg")


def test_image_upload_rejects_pdf():
    client = get_client()
    r = client.post(
        "/image",
        files={"file": ("bad.pdf", b"%PDF", "application/pdf")},
    )
    assert r.status_code == 400
    print("PASS: test_image_upload_rejects_pdf")


def test_websocket_connects():
    client = get_client()
    # Run one step to generate some graph state
    client.post("/step")

    with client.websocket_connect("/ws") as ws:
        # Connection doesn't crash — success
        pass
    print("PASS: test_websocket_connects")


def test_start_pause_resume():
    client = get_client()
    r = client.post("/start")
    assert r.json()["state"] in ("running", "idle")

    r = client.post("/pause")
    assert r.json()["state"] == "paused"

    r = client.post("/resume")
    # After resume, state should be running (or still paused if task hasn't started yet)
    assert r.json()["state"] in ("running", "paused")

    client.post("/pause")
    print("PASS: test_start_pause_resume")


def test_reset_returns_to_idle():
    client = get_client()
    client.post("/step")
    r = client.post("/reset")
    assert r.json()["state"] == "idle"
    assert r.json()["step"] == 0
    print("PASS: test_reset_returns_to_idle")


if __name__ == "__main__":
    test_health()
    test_status_returns_idle()
    test_step_returns_step_result()
    test_stage_out_of_range()
    test_chat_returns_string()
    test_image_upload_jpeg()
    test_image_upload_rejects_pdf()
    test_websocket_connects()
    test_start_pause_resume()
    test_reset_returns_to_idle()

    teardown_module()
    print("\nAll 10 tests passed.")
