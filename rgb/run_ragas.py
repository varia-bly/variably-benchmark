"""RAGAS faithfulness comparison on the same RGB pairs we submitted to variA/Bly.

Reads rgb_results.json (the plan with response + references) and
rgb_labeled.json (variA/Bly's scores per evaluation_id), runs RAGAS
faithfulness on each (question, response, contexts) triple, then joins
the two and reports head-to-head metrics:

  - Pearson r between variA/Bly and RAGAS faithfulness scores
  - AUC of each scorer as a binary classifier of grounded/not-grounded
  - Agreement rate at threshold 0.5
  - Where they disagree (specific evaluation_ids + scores)

Cost: ~$0.03 per pair (RAGAS faithfulness uses one OpenAI call per
sample). For 60 pairs: ~$1.80 GPT-4o-mini.

Usage:
    OPENAI_API_KEY=sk-... python3 rgb/run_ragas.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

OUT_DIR = Path(__file__).resolve().parent / "results"
PLAN_PATH = OUT_DIR / "rgb_results.json"
LABELED_PATH = OUT_DIR / "rgb_labeled.json"
RAGAS_OUT = OUT_DIR / "rgb_ragas_comparison.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, default=PLAN_PATH)
    p.add_argument("--labeled", type=Path, default=LABELED_PATH)
    p.add_argument("--out", type=Path, default=RAGAS_OUT)
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of pairs to score (for cost control)")
    p.add_argument("--dry-run", action="store_true",
                   help="show cost estimate without calling RAGAS")
    return p.parse_args()


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient. Stdlib only."""
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def auc(pairs: list[tuple[float, int]]) -> float:
    """Rank-based ROC AUC for (predicted, label) pairs."""
    if not pairs:
        return 0.0
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs_sorted if y == 1)
    n_neg = len(pairs_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = sum(rank for rank, (_, y) in enumerate(pairs_sorted, start=1) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def score_with_ragas(plans: list[dict]) -> dict[str, float]:
    """Score each plan via RAGAS faithfulness.

    RAGAS faithfulness asks GPT-4o-mini: "given these contexts, are the
    statements in the answer verifiable from the contexts?" Returns a
    score in [0, 1]. We use the same model variA/Bly's calibration
    expects — gpt-4o-mini — so the comparison is apples-to-apples on
    judge cost.

    Returns map from evaluation_id -> faithfulness score.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        from datasets import Dataset
    except ImportError as e:
        raise SystemExit(
            f"missing RAGAS deps: {e}\n"
            "install: pip install ragas datasets langchain-openai"
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY env var is required for RAGAS")

    rows = []
    for p in plans:
        rows.append({
            "user_input": p["query"],
            "response": p["response"],
            "retrieved_contexts": [r["content"] for r in p["references"]],
        })
    ds = Dataset.from_list(rows)

    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    print(f"Scoring {len(rows)} pairs via RAGAS faithfulness "
          f"(gpt-4o-mini, ~$0.03 per pair)...")
    t0 = time.time()
    result = evaluate(ds, metrics=[faithfulness], llm=judge)
    elapsed = time.time() - t0
    print(f"  RAGAS done in {elapsed:.1f}s")

    df = result.to_pandas()
    out: dict[str, float] = {}
    for plan, row in zip(plans, df.itertuples(index=False)):
        out[plan["evaluation_id"]] = float(row.faithfulness)
    return out


def main() -> int:
    args = parse_args()
    plan_doc = json.loads(args.plan.read_text())
    plans = [p for p in plan_doc["plans"] if p.get("evaluation_id")]
    labeled = {r["evaluation_id"]: r for r in json.loads(args.labeled.read_text())}

    if args.limit:
        plans = plans[: args.limit]

    n = len(plans)
    cost = round(n * 0.03, 2)
    print(f"Plan: {n} pairs to score via RAGAS")
    print(f"  Cost: ~${cost:.2f} OpenAI (gpt-4o-mini)")
    print(f"  ETA:  ~{max(2, n // 6)} min")

    if args.dry_run:
        print("\n--dry-run: nothing scored.")
        return 0

    if cost > 1.0:
        confirm = input(f"\nProceed with ~${cost:.2f} OpenAI spend? [y/N] ")
        if confirm.strip().lower() != "y":
            print("aborted")
            return 1

    ragas_scores = score_with_ragas(plans)

    # Join: per-plan, attach (variA/Bly score, RAGAS score, ground_truth)
    joined = []
    for p in plans:
        eid = p["evaluation_id"]
        if eid not in labeled or eid not in ragas_scores:
            continue
        joined.append({
            "evaluation_id": eid,
            "rgb_id": p["rgb_id"],
            "group": p["group"],
            "ground_truth": p["ground_truth"],
            "variably_faithfulness": labeled[eid]["faithfulness"],
            "ragas_faithfulness": ragas_scores[eid],
        })

    args.out.write_text(json.dumps(joined, indent=2))
    print(f"\nWrote {args.out} ({len(joined)} pairs)")

    # Aggregates
    if not joined:
        print("No joinable rows — bail.")
        return 1

    var_scores = [r["variably_faithfulness"] for r in joined]
    ragas_scores_list = [r["ragas_faithfulness"] for r in joined]
    truth = [r["ground_truth"] for r in joined]

    pearson_r = pearson(var_scores, ragas_scores_list)
    var_auc = auc(list(zip(var_scores, truth)))
    ragas_auc = auc(list(zip(ragas_scores_list, truth)))

    var_correct = sum(1 for r in joined
                      if (r["variably_faithfulness"] >= 0.5) == (r["ground_truth"] == 1))
    ragas_correct = sum(1 for r in joined
                        if (r["ragas_faithfulness"] >= 0.5) == (r["ground_truth"] == 1))
    agreement = sum(1 for r in joined
                    if (r["variably_faithfulness"] >= 0.5) == (r["ragas_faithfulness"] >= 0.5))

    print("\n=== variA/Bly vs RAGAS — head-to-head ===")
    print(f"  Pearson r between scores:           {pearson_r:.3f}")
    print(f"  Agreement at threshold 0.5:         {agreement / len(joined):.3f}")
    print()
    print(f"  variA/Bly AUC:                      {var_auc:.3f}")
    print(f"  RAGAS AUC:                          {ragas_auc:.3f}")
    print(f"  variA/Bly accuracy @ 0.5:           {var_correct / len(joined):.3f}")
    print(f"  RAGAS accuracy @ 0.5:               {ragas_correct / len(joined):.3f}")

    # Top disagreements (where the two scorers disagree by > 0.5)
    disagreements = sorted(joined,
                           key=lambda r: abs(r["variably_faithfulness"] - r["ragas_faithfulness"]),
                           reverse=True)[:5]
    print("\n=== Top 5 disagreements (|variably - ragas| largest) ===")
    for r in disagreements:
        delta = r["variably_faithfulness"] - r["ragas_faithfulness"]
        print(f"  rgb_id={r['rgb_id']:>3}  group={r['group']:<8}  truth={r['ground_truth']}  "
              f"variably={r['variably_faithfulness']:.2f}  ragas={r['ragas_faithfulness']:.2f}  "
              f"Δ={delta:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
