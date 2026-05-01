"""Final comparison summary at n=296: variA/Bly (reverted) vs RAGAS.

Reads rgb_labeled.json (current variA/Bly scores after revert) and
rgb_ragas_comparison.json (RAGAS scores from the previous $17.76 run —
those don't change since RAGAS is independent of our scorer).

Computes per-scorer:
  - mean faithfulness on positives / negatives
  - AUC
  - accuracy at threshold 0.5
  - false-positive rate on distractors at threshold 0.5
  - false-negative rate on positives at threshold 0.5

Output is the marketing-grade tradeoff table for the public benchmarks
page.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

OUT_DIR = Path(__file__).resolve().parent / "results"
LABELED_PATH = OUT_DIR / "rgb_labeled.json"
RAGAS_PATH = OUT_DIR / "rgb_ragas_comparison.json"


def auc(pairs):
    if not pairs:
        return 0.0
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs_sorted if y == 1)
    n_neg = len(pairs_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = sum(rank for rank, (_, y) in enumerate(pairs_sorted, start=1) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    # Index labeled.json by (rgb_id, group) — the post-revert run has DIFFERENT
    # evaluation_ids than the RAGAS comparison run (since each runner invocation
    # creates fresh observation_ids). RAGAS scores are a pure function of
    # (response, references), so the (rgb_id, group) pair is the stable key.
    labeled_rows = json.loads(LABELED_PATH.read_text())
    labeled_by_pair = {(int(r["rgb_id"]), r["group"]): r for r in labeled_rows}
    ragas_doc = json.loads(RAGAS_PATH.read_text())

    rows = []
    for r in ragas_doc:
        key = (int(r["rgb_id"]), r["group"])
        lab = labeled_by_pair.get(key)
        if lab is None:
            continue
        rows.append({
            "rgb_id": key[0],
            "group": r["group"],
            "ground_truth": r["ground_truth"],
            "variably": lab["faithfulness"],
            "ragas": r["ragas_faithfulness"],
        })

    print(f"Joined {len(rows)} pairs (variA/Bly post-revert vs RAGAS)")
    print(f"  variA/Bly labeled: {len(labeled_rows)} rows")
    print(f"  RAGAS comparison:  {len(ragas_doc)} rows")
    if not rows:
        print("\nNo pairs joined. Both files exist but no (rgb_id, group) match.")
        print("Sample labeled key: ", next(iter(labeled_by_pair)) if labeled_by_pair else None)
        print("Sample ragas key:   ", (ragas_doc[0]['rgb_id'], ragas_doc[0]['group']) if ragas_doc else None)
        return 1
    print()

    pos = [r for r in rows if r["ground_truth"] == 1]
    neg = [r for r in rows if r["ground_truth"] == 0]

    for name, key in [("variA/Bly (reverted)", "variably"), ("RAGAS", "ragas")]:
        pos_scores = [r[key] for r in pos]
        neg_scores = [r[key] for r in neg]
        all_pairs = [(r[key], r["ground_truth"]) for r in rows]
        a = auc(all_pairs)
        # threshold 0.5
        tp = sum(1 for r in pos if r[key] >= 0.5)
        fn = sum(1 for r in pos if r[key] < 0.5)
        fp = sum(1 for r in neg if r[key] >= 0.5)
        tn = sum(1 for r in neg if r[key] < 0.5)
        acc = (tp + tn) / len(rows)
        fpr = fp / len(neg) if neg else 0.0
        fnr = fn / len(pos) if pos else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        print(f"=== {name} ===")
        print(f"  Mean faithfulness positives: {mean(pos_scores):.3f}")
        print(f"  Mean faithfulness negatives: {mean(neg_scores):.3f}")
        print(f"  AUC:                          {a:.3f}")
        print(f"  Accuracy @ 0.5:               {acc:.3f}")
        print(f"  Precision @ 0.5:              {precision:.3f}")
        print(f"  Recall @ 0.5:                 {recall:.3f}")
        print(f"  False-positive rate (negs):   {fpr:.3f}")
        print(f"  False-negative rate (pos):    {fnr:.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
