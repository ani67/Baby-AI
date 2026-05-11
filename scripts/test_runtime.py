import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.mind_runtime import MindRuntime

print("[test_runtime] loading runtime ...")
rt = MindRuntime.load('first')
rt.start()
print(f"  node_count={rt.graph.node_count}")

time.sleep(2.0)  # warmup

rt.send("what is justice")
resp = rt.receive(timeout=10.0)
if resp:
    print(f"  response: '{resp.text}'  gap={resp.gap:.3f}  "
          f"active={resp.active_concept_count}  arousal={resp.arousal:.3f}")
else:
    print("  no response within 10s (expression may be suppressed)")

status = rt.status()
print(f"  status: {status}")

rt.stop()
print("test_runtime: PASS")
