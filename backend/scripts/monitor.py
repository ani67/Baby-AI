"""
Convergence monitor — watches training progress from shared_state.json.

Prints a compact dashboard every N seconds showing:
- Step count + speed (steps/min)
- Neuron count (active, growth rate)
- Distillation convergence (text cos_sim, trend)
- Reasoning accuracy (per task type)
- Generation quality

Usage:
    python -m scripts.monitor              # poll every 10s
    python -m scripts.monitor --interval 5  # poll every 5s
    python -m scripts.monitor --log convergence.csv  # also write CSV
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "shared_state.json"
BOX_WIDTH = 58


# ── State reading ──

def read_state() -> dict | None:
    """Read shared_state.json. Returns None if missing or corrupt."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def state_age_seconds() -> float:
    """Seconds since the state file was last written."""
    try:
        return time.time() - os.path.getmtime(STATE_FILE)
    except FileNotFoundError:
        return float("inf")


# ── Value extraction ──

def get_nested(d: dict, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


# ── Formatting helpers ──

def bar(value: float, width: int = 10) -> str:
    """Render a 0..1 value as a filled/empty bar."""
    if value is None:
        return "-" * width
    filled = int(round(value * width))
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def fmt_val(v, decimals: int = 3) -> str:
    """Format a metric value, showing --- for None."""
    if v is None:
        return "---"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def trend_arrow(trend: float | None) -> str:
    if trend is None:
        return " "
    if trend > 0.005:
        return "\u2191"
    if trend < -0.005:
        return "\u2193"
    return "\u2192"


def pad_line(text: str) -> str:
    """Pad a content line to fit inside the box."""
    inner = BOX_WIDTH - 4  # minus "| " and " |"
    return "\u2551 " + text.ljust(inner) + " \u2551"


# ── Milestone checks ──

MILESTONES = [
    ("text_cos_sim",       0.50, "TEXT ENCODER 50% CONVERGED"),
    ("text_cos_sim",       0.85, "TEXT ENCODER READY -- can replace CLIP"),
    ("reasoning_accuracy", 0.60, "REASONING ABOVE RANDOM"),
    ("neurons",            5000, "SCALING MILESTONE"),
]


def check_milestones(current: dict, reached: set[str]) -> list[str]:
    """Return list of newly-reached milestone messages."""
    new = []
    vals = {
        "text_cos_sim": get_nested(current, "metrics", "distillation", "text_cosine_sim"),
        "reasoning_accuracy": get_nested(current, "metrics", "reasoning", "overall_accuracy"),
        "neurons": get_nested(current, "graph_summary", "node_count"),
    }
    for key, threshold, message in MILESTONES:
        tag = f"{key}>{threshold}"
        if tag in reached:
            continue
        v = vals.get(key)
        if v is not None and v >= threshold:
            reached.add(tag)
            new.append(message)
    return new


# ── Dashboard rendering ──

def render_dashboard(state: dict, steps_per_min: float | None, age: float) -> list[str]:
    """Build the dashboard lines."""
    now = datetime.now().strftime("%H:%M:%S")
    step = get_nested(state, "step", default="?")
    neurons = get_nested(state, "graph_summary", "node_count", default="?")

    # Distillation
    text_sim = get_nested(state, "metrics", "distillation", "text_cosine_sim")
    text_samples = get_nested(state, "metrics", "distillation", "text_samples", default=0)
    text_trend = get_nested(state, "metrics", "distillation", "text_cosine_sim_trend")
    vision_sim = get_nested(state, "metrics", "distillation", "vision_cosine_sim")

    # Reasoning
    total_tasks = get_nested(state, "metrics", "reasoning", "total_tasks", default=0)
    overall_acc = get_nested(state, "metrics", "reasoning", "overall_accuracy")

    # Generation
    vocab = get_nested(state, "metrics", "generation", "vocab_size", default=0)
    relevance = get_nested(state, "metrics", "generation", "response_relevance")

    # Speed string
    speed_str = f"+{steps_per_min:,.0f}/min" if steps_per_min is not None else "measuring..."

    # Staleness warning
    stale = " [STALE]" if age > 60 else ""

    top    = "\u2554" + "\u2550" * (BOX_WIDTH - 2) + "\u2557"
    mid    = "\u2560" + "\u2550" * (BOX_WIDTH - 2) + "\u2563"
    bottom = "\u255a" + "\u2550" * (BOX_WIDTH - 2) + "\u255d"

    lines = [
        top,
        pad_line(f"BABY AI MONITOR{stale}".ljust(BOX_WIDTH - 20) + now),
        mid,
        pad_line(f"BRAIN   step={step:,}  {speed_str}  neurons={neurons:,}" if isinstance(step, int) and isinstance(neurons, int)
                 else f"BRAIN   step={step}  {speed_str}  neurons={neurons}"),
        pad_line(f"DISTILL text={fmt_val(text_sim)} {bar(text_sim)} ({text_samples} samples)  trend: {trend_arrow(text_trend)}"),
        pad_line(f"        vision={fmt_val(vision_sim)}"),
        pad_line(f"REASON  tasks={total_tasks}  overall={fmt_val(overall_acc)}"),
        pad_line(f"GEN     vocab={vocab}  relevance={fmt_val(relevance)}"),
        bottom,
    ]
    return lines


# ── CSV logging ──

CSV_COLUMNS = [
    "timestamp", "step", "neurons", "text_cos_sim",
    "text_samples", "reasoning_tasks", "reasoning_accuracy",
]


def init_csv(path: str) -> None:
    """Write CSV header if file does not exist."""
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)


def append_csv(path: str, state: dict) -> None:
    """Append one row to the CSV log."""
    row = [
        datetime.now().isoformat(),
        get_nested(state, "step"),
        get_nested(state, "graph_summary", "node_count"),
        get_nested(state, "metrics", "distillation", "text_cosine_sim"),
        get_nested(state, "metrics", "distillation", "text_samples"),
        get_nested(state, "metrics", "reasoning", "total_tasks"),
        get_nested(state, "metrics", "reasoning", "overall_accuracy"),
    ]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ── Main loop ──

def main() -> None:
    parser = argparse.ArgumentParser(description="Baby AI convergence monitor")
    parser.add_argument("--interval", type=float, default=10, help="Poll interval in seconds (default: 10)")
    parser.add_argument("--log", type=str, default=None, help="Path to CSV log file")
    args = parser.parse_args()

    if args.log:
        init_csv(args.log)

    prev_step: int | None = None
    prev_time: float | None = None
    reached_milestones: set[str] = set()

    print(f"Monitoring {STATE_FILE}")
    print(f"Poll interval: {args.interval}s" + (f"  CSV: {args.log}" if args.log else ""))
    print()

    try:
        while True:
            state = read_state()
            age = state_age_seconds()

            if state is None:
                print("Waiting for training to start (no shared_state.json)...")
                time.sleep(args.interval)
                continue

            # Compute speed
            cur_step = get_nested(state, "step")
            now = time.time()
            steps_per_min = None
            if prev_step is not None and cur_step is not None and prev_time is not None:
                dt = now - prev_time
                if dt > 0:
                    steps_per_min = (cur_step - prev_step) / (dt / 60)
            prev_step = cur_step
            prev_time = now

            # Render
            dashboard = render_dashboard(state, steps_per_min, age)
            # Clear previous output and print
            sys.stdout.write("\033[2J\033[H")  # clear screen, cursor to top
            for line in dashboard:
                print(line)

            # Milestones
            new_milestones = check_milestones(state, reached_milestones)
            for msg in new_milestones:
                print(f"\n  *** {msg} ***")

            # CSV
            if args.log:
                append_csv(args.log, state)

            sys.stdout.flush()
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
