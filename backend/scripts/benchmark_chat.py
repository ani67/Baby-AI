"""Ask 3 benchmark questions and log responses. Run periodically to track evolution."""
import json, os, time, sys

CHAT_REQ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chat_request.json")
CHAT_RESP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chat_response.json")
STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "shared_state.json")
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "benchmark_chat_log.jsonl")

QUESTIONS = [
    "what is a dog",
    "the sky is",
    "two plus three equals",
]

def ask(question, timeout=30):
    if os.path.exists(CHAT_RESP):
        os.unlink(CHAT_RESP)
    with open(CHAT_REQ, "w") as f:
        json.dump({"message": question, "type": "chat"}, f)
    for _ in range(timeout):
        time.sleep(1)
        if os.path.exists(CHAT_RESP):
            with open(CHAT_RESP) as f:
                resp = json.load(f)
            os.unlink(CHAT_RESP)
            return resp.get("message", "")
    return "(no response)"

def get_step():
    try:
        with open(STATE_FILE) as f:
            return json.load(f).get("step", 0)
    except Exception:
        return 0

if __name__ == "__main__":
    step = get_step()
    print(f"\n=== BENCHMARK @ step {step} ===")
    entry = {"step": step, "timestamp": time.time(), "responses": {}}
    for q in QUESTIONS:
        answer = ask(q)
        print(f'  Q: "{q}"')
        print(f'  A: "{answer}"')
        print()
        entry["responses"][q] = answer
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Logged to {LOG_FILE}")
