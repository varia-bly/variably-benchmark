"""Analyze RGB benchmark results after async scoring completes.

Reads the rgb_results.json plan written by rgb_runner.py, polls the
deployed dev DB for dimensional_scores by evaluation_id, computes:

  - mean faithfulness on positive vs negative groups
  - AUC of faithfulness as a binary classifier of grounded/not-grounded
  - accuracy at threshold 0.5
  - per-sample (eval_id, group, faithfulness, hallucination_rate, misinformation, retrieval_relevance)

Output: prints a summary table to stdout and writes a labeled
calibration training set to rgb/results/rgb_labeled.json
(usable as a labeled set for downstream calibration).

Usage:
    DATABASE_URL="postgres://..." python3 rgb/analyze.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

OUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_PLAN = OUT_DIR / "rgb_results.json"
LABELED_OUT = OUT_DIR / "rgb_labeled.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN,
                   help="rgb_results.json from the runner")
    p.add_argument("--out", type=Path, default=LABELED_OUT,
                   help="where to write the labeled calibration set")
    p.add_argument("--wait-sec", type=int, default=0,
                   help="poll the DB this many seconds before reading; "
                        "useful if you ran the runner just now")
    return p.parse_args()


def fetch_scores(eval_ids: list[str]) -> dict[str, dict]:
    """Pull dimensional_scores for the given evaluation_ids."""
    import psycopg2
    import psycopg2.extras

    if not eval_ids:
        return {}

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL env var is required")

    out: dict[str, dict] = {}
    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT evaluation_id, faithfulness, hallucination_rate, misinformation,
                       intent_alignment, cross_encoder_relevance, retrieval_relevance,
                       attribution_accuracy, context_utilization
                FROM dimensional_scores
                WHERE evaluation_id = ANY(%s)
                """,
                (eval_ids,),
            )
            for row in cur.fetchall():
                out[row["evaluation_id"]] = dict(row)
    return out


def recover_eval_ids_by_metadata(plans: list[dict]) -> dict[tuple, str]:
    """For each (rgb_id, group) pair, find the matching evaluation_id in the
    DB by metadata. Used when the runner didn't capture observation_id at
    submission time.

    Returns a mapping {(rgb_id, group): evaluation_id}.
    Looks up evaluations created in the last 4 hours that match
    metadata.benchmark='rgb' and the (rgb_id, group) pair.
    """
    import psycopg2
    import psycopg2.extras

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL env var is required")

    needs_recovery = [(p["rgb_id"], p["group"]) for p in plans
                      if not p.get("evaluation_id")]
    if not needs_recovery:
        return {}

    print(f"Recovering {len(needs_recovery)} eval_ids by metadata lookup...")
    found: dict[tuple, str] = {}
    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            for rgb_id, group in needs_recovery:
                cur.execute(
                    """
                    SELECT evaluation_id, created_at
                    FROM prompt_evaluations
                    WHERE metadata->>'benchmark' = 'rgb'
                      AND (metadata->>'rgb_id')::int = %s
                      AND metadata->>'group' = %s
                      AND created_at > NOW() - INTERVAL '4 hours'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (rgb_id, group),
                )
                row = cur.fetchone()
                if row:
                    found[(rgb_id, group)] = row["evaluation_id"]
    print(f"  Recovered {len(found)} / {len(needs_recovery)}")
    return found


def auc(pairs: list[tuple[float, int]]) -> float:
    """Compute ROC AUC for (predicted_score, label) pairs.

    Trapezoidal AUC via rank statistics (no sklearn dep). label in {0, 1}.
    """
    if not pairs:
        return 0.0
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs_sorted if y == 1)
    n_neg = len(pairs_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # rank-based AUC: average rank of positives / n_neg
    rank_sum = 0.0
    for rank, (_, y) in enumerate(pairs_sorted, start=1):
        if y == 1:
            rank_sum += rank
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    plans = plan["plans"]
    print(f"Loaded {len(plans)} plan rows from {args.plan}")

    # If the runner didn't capture observation_id (older runs), look up by
    # metadata and patch the plan in memory + on disk.
    missing_ids = [p for p in plans if not p.get("evaluation_id")]
    if missing_ids:
        recovered = recover_eval_ids_by_metadata(plans)
        if recovered:
            for p in plans:
                if not p.get("evaluation_id"):
                    p["evaluation_id"] = recovered.get((p["rgb_id"], p["group"]))
            # Persist recovered IDs back to the plan so re-runs are idempotent
            plan["plans"] = plans
            args.plan.write_text(json.dumps(plan, indent=2))
            print(f"  Patched plan file with {sum(1 for p in plans if p.get('evaluation_id'))} ids")

    eval_ids = [p["evaluation_id"] for p in plans if p.get("evaluation_id")]
    if not eval_ids:
        raise SystemExit("no evaluation_ids found and metadata recovery returned nothing")

    if args.wait_sec > 0:
        import time
        print(f"Waiting {args.wait_sec}s for async scoring to settle...")
        time.sleep(args.wait_sec)

    print(f"Fetching scores for {len(eval_ids)} evaluations...")
    scores = fetch_scores(eval_ids)
    print(f"Got scores for {len(scores)} / {len(eval_ids)} evaluations")
    if len(scores) < len(eval_ids):
        missing = set(eval_ids) - set(scores)
        print(f"  Missing: {len(missing)} (still scoring? rerun in 60s)")
        for mid in list(missing)[:3]:
            print(f"    - {mid}")

    # Join scores with plan
    labeled = []
    for p in plans:
        eid = p.get("evaluation_id")
        s = scores.get(eid)
        if not s:
            continue
        labeled.append({
            "evaluation_id": eid,
            "rgb_id": p["rgb_id"],
            "subset": p["subset"],
            "group": p["group"],
            "ground_truth": p["ground_truth"],
            "faithfulness": float(s["faithfulness"] or 0),
            "hallucination_rate": float(s["hallucination_rate"] or 0),
            "misinformation": float(s["misinformation"] or 0),
            "intent_alignment": float(s["intent_alignment"] or 0),
            "cross_encoder_relevance": float(s["cross_encoder_relevance"] or 0),
            "retrieval_relevance": float(s["retrieval_relevance"] or 0),
            "attribution_accuracy": float(s["attribution_accuracy"] or 0),
        })

    args.out.write_text(json.dumps(labeled, indent=2))
    print(f"\nWrote labeled set: {args.out} ({len(labeled)} rows)")

    # Summary
    pos = [r for r in labeled if r["ground_truth"] == 1]
    neg = [r for r in labeled if r["ground_truth"] == 0]
    if not pos or not neg:
        print("\nNot enough samples in both groups for a summary; need at least 1 of each.")
        return 0

    pos_f = [r["faithfulness"] for r in pos]
    neg_f = [r["faithfulness"] for r in neg]

    pairs = [(r["faithfulness"], r["ground_truth"]) for r in labeled]
    auc_score = auc(pairs)

    # Accuracy at threshold 0.5
    correct = sum(
        1 for r in labeled
        if (r["faithfulness"] >= 0.5 and r["ground_truth"] == 1)
        or (r["faithfulness"] < 0.5 and r["ground_truth"] == 0)
    )
    acc = correct / len(labeled)

    print("\n=== RGB binary-classification summary ===")
    print(f"  positives (refs contain answer):   n={len(pos)}  mean faithfulness = {mean(pos_f):.3f}")
    print(f"  negatives (refs distract):         n={len(neg)}  mean faithfulness = {mean(neg_f):.3f}")
    print(f"  separation (pos - neg):                            {mean(pos_f) - mean(neg_f):.3f}")
    print(f"  AUC:                                {auc_score:.3f}")
    print(f"  accuracy @ threshold 0.5:           {acc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
