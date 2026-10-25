"""Recompute the October 2026 RGB table from the per-case results — no API key, no rescoring.

    python3 rgb/verify_2026_10.py

Reads rgb/results/rgb_2026-10_four_scorers.json (one row per case with every scorer's score)
and prints, for each scorer: cases scored, recall (grounded cases approved), false approvals
(distractor cases approved), precision and AUC. A case is approved when its faithfulness is at
least 0.5; DeepEval's hallucination score is inverted (1 - score) first. Then prints the matched
operating point (RAGAS's threshold raised to 0.6). Standard library only.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "results" / "rgb_2026-10_four_scorers.json"
SCORERS = [
    ("variA/Bly 4.2.1", lambda r: r["variably_4_2_1"]),
    ("RAGAS faithfulness", lambda r: r["ragas_faithfulness"]),
    ("DeepEval Hallucination", lambda r: None if r["deepeval_hallucination"] is None else 1 - r["deepeval_hallucination"]),
    ("DeepEval Faithfulness", lambda r: r["deepeval_faithfulness"]),
    ("variA/Bly April 2026 (history)", lambda r: r["variably_april_2026"]),
]


def auc(pos, neg):
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def operating_point(cases, get, threshold):
    scored = [(get(r), r["ground_truth"]) for r in cases if get(r) is not None]
    pos = [s for s, t in scored if t == 1]
    neg = [s for s, t in scored if t == 0]
    tp = sum(s >= threshold for s in pos)
    fp = sum(s >= threshold for s in neg)
    return tp, len(pos), fp, len(neg)


def main():
    cases = json.loads(DATA.read_text())["cases"]
    print(f"{'Scorer':32} {'scored':>8} {'recall':>14} {'false approvals':>18} {'precision':>10} {'AUC':>6}")
    for name, get in SCORERS:
        scored = [(get(r), r["ground_truth"]) for r in cases if get(r) is not None]
        pos = [s for s, t in scored if t == 1]
        neg = [s for s, t in scored if t == 0]
        tp = sum(s >= 0.5 for s in pos)
        fp = sum(s >= 0.5 for s in neg)
        print(f"{name:32} {len(scored):>4}/{len(cases):<3} {100 * tp / len(pos):6.1f}% ({tp}/{len(pos)})"
              f" {100 * fp / len(neg):8.1f}% ({fp}/{len(neg)}) {100 * tp / (tp + fp):9.1f}% {auc(pos, neg):6.3f}")

    # Matched operating point: RAGAS's scores are continuous, so its threshold can be raised to
    # the point variA/Bly reaches at its default.
    print("\nMatched operating point")
    for name, get, t in [("variA/Bly 4.2.1, default (0.5)", SCORERS[0][1], 0.5),
                         ("RAGAS, threshold 0.6", SCORERS[1][1], 0.6),
                         ("RAGAS, default (0.5)", SCORERS[1][1], 0.5)]:
        tp, npos, fp, nneg = operating_point(cases, get, t)
        print(f"{name:32} correct approved {100 * tp / npos:5.1f}% ({tp}/{npos})"
              f"  distractors approved {100 * fp / nneg:5.1f}% ({fp}/{nneg})  precision {100 * tp / (tp + fp):5.1f}%")


if __name__ == "__main__":
    main()
