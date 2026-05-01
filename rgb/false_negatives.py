"""Targeted diagnostic for false negatives where variA/Bly=0 but truth=1.

For each false negative:
  - Print the full response text
  - Print all 5 positive references (the ones that SHOULD ground the response)
  - Print every claim and its grounding decision from claim_analyses
  - Compare with RAGAS's score (from rgb_ragas_comparison.json) to confirm
    the case is a real false negative (not just our test design noise)

The goal: tell whether each miss is
  (a) NLI rejecting a real paraphrase entailment,
  (b) numerics missing despite the union-of-refs fix,
  (c) the response containing a sub-claim not actually in any ref
      (the LLM hallucinated a detail), or
  (d) something else.

Usage:
    DATABASE_URL=... python3 rgb/false_negatives.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "results"
PLAN = OUT_DIR / "rgb_results.json"
LABELED = OUT_DIR / "rgb_labeled.json"
RAGAS_COMPARISON = OUT_DIR / "rgb_ragas_comparison.json"


def main() -> int:
    plan = json.loads(PLAN.read_text())
    labeled = {r["evaluation_id"]: r for r in json.loads(LABELED.read_text())}
    ragas = (
        {r["evaluation_id"]: r for r in json.loads(RAGAS_COMPARISON.read_text())}
        if RAGAS_COMPARISON.exists() else {}
    )

    # Find false negatives: ground_truth=1 (positive) but variA/Bly faithfulness < 0.5
    false_negatives = []
    for p in plan["plans"]:
        if p.get("ground_truth") != 1:
            continue
        eid = p.get("evaluation_id")
        if not eid or eid not in labeled:
            continue
        if labeled[eid]["faithfulness"] < 0.5:
            false_negatives.append(p)

    print(f"False negatives (positives variA/Bly scored < 0.5): {len(false_negatives)}\n")
    if not false_negatives:
        return 0

    # Pull claim_analyses for these eval_ids
    import psycopg2
    import psycopg2.extras

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL env var is required")

    eval_ids = [p["evaluation_id"] for p in false_negatives]
    claims_by_eval: dict[str, list[dict]] = {}
    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT evaluation_id, claim_index, is_grounded, failure_reason,
                       entailment_score, contradiction_score, numeric_mismatches,
                       claim_text, supporting_ref_id, supporting_ref_excerpt
                FROM claim_analyses
                WHERE evaluation_id = ANY(%s)
                ORDER BY evaluation_id, claim_index
                """,
                (eval_ids,),
            )
            for row in cur.fetchall():
                claims_by_eval.setdefault(row["evaluation_id"], []).append(dict(row))

    # Print full detail per false negative
    for p in false_negatives:
        eid = p["evaluation_id"]
        rgb_id = p["rgb_id"]
        var_score = labeled[eid]["faithfulness"]
        ragas_score = ragas.get(eid, {}).get("ragas_faithfulness", "N/A")
        ragas_str = f"{ragas_score:.2f}" if isinstance(ragas_score, (int, float)) else "N/A"

        print("=" * 80)
        print(f"rgb_id={rgb_id}  variA/Bly={var_score:.2f}  RAGAS={ragas_str}")
        print(f"\nQuery:    {p['query']}")
        print(f"Response: {p['response']}")
        print()

        for i, ref in enumerate(p["references"]):
            content = ref["content"]
            print(f"--- positive ref [{i}] ({len(content)} chars) ---")
            print(content[:1000] + ("..." if len(content) > 1000 else ""))
            print()

        print("Claims extracted by atomic decomposition:")
        for c in claims_by_eval.get(eid, []):
            mark = "✓" if c["is_grounded"] else "✗"
            reason = c["failure_reason"] or "GROUNDED"
            ent = float(c["entailment_score"] or 0)
            con = float(c["contradiction_score"] or 0)
            nums = list(c["numeric_mismatches"] or [])
            nums_s = f"  nums_missing={nums}" if nums else ""
            print(f"  [{mark}] {reason:<22}  ent={ent:.3f}  con={con:.3f}{nums_s}")
            print(f"      claim: {c['claim_text']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
