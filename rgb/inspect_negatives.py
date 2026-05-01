"""Inspect specific RGB negative cases to see if the rescue was justified.

Loads rgb_results.json and prints the full distractor refs for the
target rgb_ids — answers "is the claim's answer actually absent from
all 5 distractors, or did it sneak in?"

Usage:
    python3 rgb/inspect_negatives.py 17 19 21
"""
import json
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parent / "results" / "rgb_results.json"


def main(targets: list[str]) -> int:
    plan = json.loads(PLAN.read_text())
    seen: set[int] = set()
    for p in plan["plans"]:
        if p["group"] != "negative":
            continue
        if str(p["rgb_id"]) not in targets:
            continue
        if p["rgb_id"] in seen:
            continue
        seen.add(p["rgb_id"])
        print("=" * 80)
        print(f"rgb_id={p['rgb_id']}  query: {p['query']}")
        print(f"response: {p['response']}")
        print()
        for i, r in enumerate(p["references"]):
            content = r["content"]
            print(f"--- distractor {i} ({len(content)} chars) ---")
            print(content[:800] + ("..." if len(content) > 800 else ""))
            print()
    return 0


if __name__ == "__main__":
    targets = sys.argv[1:] or ["17", "19", "21"]
    sys.exit(main(targets))
