"""Diagnose why some positive RGB cases score below 1.0.

Reads rgb_labeled.json (produced by rgb_analyze.py), picks the
positive-but-low-scoring evals, pulls their claim_analyses, and
aggregates failure reasons + per-claim detail.

Usage:
    DATABASE_URL=... python3 rgb/diagnose.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

LABELED = Path(__file__).resolve().parent / "results" / "rgb_labeled.json"


def main() -> int:
    if not LABELED.exists():
        raise SystemExit(f"missing {LABELED} — run rgb_analyze.py first")

    rows = json.loads(LABELED.read_text())
    pos = [r for r in rows if r["ground_truth"] == 1]
    pos_low = sorted([r for r in pos if r["faithfulness"] < 1.0],
                     key=lambda r: r["faithfulness"])
    print(f"Positives total: {len(pos)}; below-1.0: {len(pos_low)}")

    if not pos_low:
        print("All positives scored 1.0 — nothing to diagnose.")
        return 0

    import psycopg2
    import psycopg2.extras

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL env var is required")

    eval_ids = [r["evaluation_id"] for r in pos_low]

    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT pe.evaluation_id,
                       pe.llm_response,
                       ca.claim_index,
                       ca.is_grounded,
                       ca.failure_reason,
                       ca.entailment_score,
                       ca.contradiction_score,
                       ca.numeric_mismatches,
                       ca.claim_text,
                       ca.supporting_ref_excerpt
                FROM prompt_evaluations pe
                JOIN claim_analyses ca ON ca.evaluation_id = pe.evaluation_id
                WHERE pe.evaluation_id = ANY(%s)
                ORDER BY pe.evaluation_id, ca.claim_index
                """,
                (eval_ids,),
            )
            claims_by_eval: dict[str, list[dict]] = {}
            for row in cur.fetchall():
                claims_by_eval.setdefault(row["evaluation_id"], []).append(dict(row))

    # 1. Aggregate failure reasons across all below-1.0 positives
    print("\n=== Failure reason histogram (positives below 1.0) ===")
    reasons = Counter()
    n_claims = 0
    for eid, claims in claims_by_eval.items():
        for c in claims:
            n_claims += 1
            reasons[c["failure_reason"] or "GROUNDED"] += 1
    for reason, count in reasons.most_common():
        pct = count / n_claims * 100
        print(f"  {reason:<25} {count:>4}  ({pct:.1f}%)")
    print(f"  {'TOTAL claims':<25} {n_claims:>4}")

    # 2. Per-eval detail, lowest faithfulness first
    print("\n=== Per-eval breakdown (lowest faithfulness first, top 10) ===")
    for r in pos_low[:10]:
        eid = r["evaluation_id"]
        claims = claims_by_eval.get(eid, [])
        if not claims:
            continue
        response = claims[0]["llm_response"]
        print(f"\n--- rgb_id={r['rgb_id']:>3}  faithfulness={r['faithfulness']:.3f}  claims={len(claims)}")
        print(f"    response: {response[:120]}")
        for c in claims:
            mark = "✓" if c["is_grounded"] else "✗"
            ent = float(c["entailment_score"] or 0)
            reason = c["failure_reason"] or ""
            nums = list(c["numeric_mismatches"] or [])
            nums_s = f"  nums_missing={nums}" if nums else ""
            print(f"    [{mark}] ent={ent:.3f}  {reason:<22}{nums_s}")
            print(f"        claim: {c['claim_text'][:120]}")
            if c["supporting_ref_excerpt"]:
                print(f"        ref:   {c['supporting_ref_excerpt'][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
