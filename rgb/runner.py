"""RGB Benchmark runner for variA/Bly grounding scoring.

Improvement #05. Runs the RGB benchmark
(https://github.com/chen700564/RGB) through variA/Bly's grounding
pipeline and produces per-sample (predicted_score, ground_truth) pairs
that double as a labeled set for #07 calibration.

Test design — binary classification on known truth
==================================================

Each RGB record has:
  - query        the user's question
  - answer       the correct short answer (list of equivalent forms)
  - positive[5]  five chunks that contain the answer (RAG-retrieved-good)
  - negative[35] thirty-five chunks that don't (distractors)

We submit each sample TWICE through variA/Bly observe mode:

  Positive case:  (query, answer, references=positive[:5])
                  variA/Bly should score this as grounded (faithfulness high)
                  ground truth: 1 (grounded)

  Negative case:  (query, answer, references=negative[:5])
                  variA/Bly should score this as NOT grounded
                  ground truth: 0 (not grounded)

After both runs land in the DB we compute:
  - mean faithfulness on positive vs negative groups
  - AUC of faithfulness as a classifier of grounded/not-grounded
  - accuracy at threshold 0.5
  - per-sample (eval_id, group, faithfulness, hallucination_rate, misinformation)

Output rows go to rgb/results/rgb_results.json.
That file doubles as a labeled set for downstream calibration.

Cost: variA/Bly-only path is FREE (local NLI + cross-encoder, no LLM
calls). RAGAS opt-in via --ragas costs ~$0.03 per pair.

Usage:
    VARIABLY_API_KEY=vb_dev_... \\
    VARIABLY_BASE_URL=https://api.variably.tech \\
    python3 rgb/runner.py --limit 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RGB_ROOT = Path(__file__).resolve().parent / "data" / "RGB"
LLM_RESPONSE_CACHE = OUT_DIR / "rgb_llm_responses.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB Benchmark runner for variA/Bly")
    p.add_argument("--subset", choices=["en", "en_fact", "en_int", "en_refine", "all"],
                   default="en", help="RGB subset to run (default: en, the 300-sample noise-robustness set)")
    p.add_argument("--limit", type=int, default=30,
                   help="number of samples (default: 30 = 60 submissions = ~10 min)")
    p.add_argument("--out", type=Path, default=OUT_DIR / "rgb_results.json")
    p.add_argument("--ragas", action="store_true",
                   help="also score via RAGAS LLM-as-judge (consumes OpenAI credit)")
    p.add_argument("--llm-response", action="store_true",
                   help="generate a realistic RAG-style response via GPT-4o-mini "
                        "given (query, top-5 positive chunks) instead of using a "
                        "synthetic template. ~$0.001 per sample. Responses are "
                        "cached in rgb_llm_responses.json for re-runs.")
    p.add_argument("--dry-run", action="store_true",
                   help="print plan + cost estimate without submitting")
    return p.parse_args()


def load_rgb_samples(subset: str, limit: int) -> List[dict]:
    """Load RGB samples from the data/ subdir.

    RGB stores each subset as JSONL — one record per line. Each record
    has id, query, answer (list of valid alternative answers), positive,
    negative.
    """
    if not RGB_ROOT.exists():
        raise SystemExit(
            f"RGB dataset not found at {RGB_ROOT}.\n"
            "Clone with: git clone --depth 1 https://github.com/chen700564/RGB rgb/data/RGB"
        )

    files = (
        ["en.json", "en_fact.json", "en_int.json", "en_refine.json"]
        if subset == "all"
        else [f"{subset}.json"]
    )

    samples: List[dict] = []
    for fname in files:
        fpath = RGB_ROOT / "data" / fname
        if not fpath.exists():
            raise SystemExit(f"missing RGB file {fpath}")
        with fpath.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # RGB's `answer` schema is mixed:
                #   sample 0: [["form1", "form2", ...]]  — list of valid alternatives
                #   sample 1: ["medical"]                — flat list of strings
                # We pick the first canonical form; handle both shapes.
                ans_field = rec.get("answer") or []
                if not ans_field:
                    canonical = ""
                else:
                    first = ans_field[0]
                    if isinstance(first, list):
                        canonical = first[0] if first else ""
                    else:
                        canonical = first  # already a string
                samples.append({
                    "id": rec["id"],
                    "subset": fname.replace(".json", ""),
                    "query": rec["query"],
                    "answer": canonical,
                    "positive": rec.get("positive", []),
                    "negative": rec.get("negative", []),
                })
                if len(samples) >= limit:
                    return samples
    return samples


def _build_response(query: str, answer: str) -> str:
    """Synthetic-template fallback when --llm-response isn't used.

    Diagnosed Apr 29 2026: this template ("The answer to the question is X")
    produces unsupported meta-claims that NLI rejects as contradiction
    (~57% of below-1.0 positives) because "the answer to the question"
    can't be grounded against a paragraph that mentions X without saying
    "X is the answer". For real benchmark numbers prefer --llm-response,
    which generates realistic RAG-style sentences. This template is kept
    only for free / no-OpenAI smoke tests.
    """
    a = answer.rstrip(" .").strip()
    return f"The answer to the question is {a}."


def _load_llm_cache() -> dict:
    if LLM_RESPONSE_CACHE.exists():
        try:
            return json.loads(LLM_RESPONSE_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_llm_cache(cache: dict) -> None:
    LLM_RESPONSE_CACHE.write_text(json.dumps(cache, indent=2))


def _generate_llm_response(query: str, positive_chunks: List[str], cache: dict, sample_id: int) -> str:
    """Generate a realistic RAG-style response via GPT-4o-mini.

    Uses (query, top-5 positive chunks) as input — same context a real
    RAG bot would have. Caches by RGB sample_id so re-runs don't hit
    the API. Cost: ~$0.001 per sample (GPT-4o-mini, ~$0.15/1M input,
    $0.60/1M output, capped at 200 output tokens).
    """
    cache_key = str(sample_id)
    if cache_key in cache:
        return cache[cache_key]

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("install openai first: pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY env var is required for --llm-response")

    client = OpenAI()
    context = "\n\n".join(c.strip() for c in positive_chunks[:5] if c)
    user_msg = (
        "Answer the user's question using only information from the provided context. "
        "Reply with a single complete sentence (or two short ones) — no preamble, no "
        "bullet points, no apologies. If the context doesn't contain the answer, "
        "respond exactly with: NO_ANSWER\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=200,
    )
    response = resp.choices[0].message.content.strip()
    cache[cache_key] = response
    _save_llm_cache(cache)
    return response


def make_submissions(samples: List[dict], use_llm_response: bool = False) -> List[dict]:
    """Pair each RGB sample into (positive, negative) submissions.

    Positive: query → response with positive refs.   ground_truth = 1
    Negative: query → response with negative refs.   ground_truth = 0

    The same response is used for both groups — only the references change.
    With --llm-response, the response is generated by GPT-4o-mini given
    (query, positive chunks), making it a realistic RAG-style sentence.
    Without it, a synthetic template wraps the canonical RGB short answer.
    """
    cache = _load_llm_cache() if use_llm_response else {}
    plans = []
    skipped_no_answer = 0
    for s in samples:
        # Take up to 5 refs from each side. RGB positives are always 5;
        # we cap negatives at 5 for symmetry.
        pos_refs = s["positive"][:5]
        neg_refs = s["negative"][:5]
        if not pos_refs or not neg_refs:
            continue

        if use_llm_response:
            response = _generate_llm_response(s["query"], pos_refs, cache, s["id"])
            # If GPT-4o-mini decided the context didn't answer the question,
            # skip — using NO_ANSWER as the response would corrupt the test
            # (it's not the canonical answer, neither group can ground it).
            if response.strip().upper() == "NO_ANSWER":
                skipped_no_answer += 1
                continue
        else:
            response = _build_response(s["query"], s["answer"])

        plans.append({
            "rgb_id": s["id"],
            "subset": s["subset"],
            "group": "positive",
            "ground_truth": 1,
            "query": s["query"],
            "response": response,
            "references": [{"id": f"pos-{i}", "content": c, "source": "RGB-positive"}
                           for i, c in enumerate(pos_refs)],
        })
        plans.append({
            "rgb_id": s["id"],
            "subset": s["subset"],
            "group": "negative",
            "ground_truth": 0,
            "query": s["query"],
            "response": response,
            "references": [{"id": f"neg-{i}", "content": c, "source": "RGB-negative"}
                           for i, c in enumerate(neg_refs)],
        })

    if skipped_no_answer:
        print(f"  Skipped {skipped_no_answer} samples where GPT-4o-mini "
              f"could not extract an answer from the positives.")
    return plans


def submit_via_sdk(plans: List[dict]) -> List[dict]:
    """Submit each plan through the variA/Bly SDK (observe mode)."""
    try:
        from variably import VariablyClient, VariablyConfig
    except ImportError:
        raise SystemExit("install the variA/Bly SDK first: pip install variably-sdk>=2.8.0")

    config = VariablyConfig.from_env()
    client = VariablyClient(config)
    print(f"variA/Bly SDK connected to {config.base_url}")

    submitted = []
    for i, p in enumerate(plans, start=1):
        try:
            res = client.log(
                prompt=p["query"],
                response=p["response"],
                reference_materials=p["references"],
                retrieval_query=p["query"],
                tags=["benchmark", "rgb", p["subset"], f"group:{p['group']}"],
                metadata={
                    "benchmark": "rgb",
                    "rgb_id": p["rgb_id"],
                    "subset": p["subset"],
                    "group": p["group"],
                    "ground_truth": p["ground_truth"],
                },
            )
            # SDK's LogResult exposes observation_id, which is the same UUID
            # the Go side stores as prompt_evaluations.evaluation_id.
            eval_id = getattr(res, "observation_id", None) or (
                res.get("observation_id") if isinstance(res, dict) else None
            )
            submitted.append({**p, "evaluation_id": eval_id})
            if i % 10 == 0:
                print(f"  submitted {i}/{len(plans)}")
        except Exception as e:
            print(f"  FAILED rgb_id={p['rgb_id']} group={p['group']}: {e}")
    return submitted


def estimate(plans: List[dict], with_ragas: bool, with_llm_response: bool, n_samples: int) -> dict:
    """Print + return cost / runtime estimate before submitting."""
    n = len(plans)
    # Variably-only: ~5-15s per submission for grounding (NLI is slow on CPU)
    eta_min = max(5, int(n * 8 / 60))
    cost_usd = 0.0
    if with_llm_response:
        # GPT-4o-mini at ~$0.001 per sample (one call per RGB sample, not per submission)
        cost_usd += round(n_samples * 0.001, 3)
    if with_ragas:
        # ~$0.03 per RAGAS faithfulness call
        cost_usd += round(n * 0.03, 2)
        eta_min += int(n * 5 / 60)

    return {"submissions": n, "eta_min": eta_min, "openai_cost_usd": round(cost_usd, 3)}


def main() -> int:
    args = parse_args()
    print(f"Loading RGB subset={args.subset} limit={args.limit}...")
    samples = load_rgb_samples(args.subset, args.limit)
    print(f"Loaded {len(samples)} RGB samples")

    if args.dry_run:
        # For dry-run, show worst-case estimate without calling OpenAI
        worst_n = len(samples) * 2  # if no NO_ANSWER skips
        est_dry = {
            "submissions": worst_n,
            "eta_min": max(5, int(worst_n * 8 / 60)) +
                       (int(worst_n * 5 / 60) if args.ragas else 0),
            "openai_cost_usd": round(
                (len(samples) * 0.001 if args.llm_response else 0) +
                (worst_n * 0.03 if args.ragas else 0),
                3,
            ),
        }
        print(f"\nPlan: up to {est_dry['submissions']} submissions to variA/Bly")
        print(f"  ETA:  ~{est_dry['eta_min']} min")
        print(f"  Cost: ${est_dry['openai_cost_usd']:.3f}")
        print("\n--dry-run: nothing submitted. Drop --dry-run to execute.")
        return 0

    if args.llm_response:
        print(f"Generating realistic responses via GPT-4o-mini "
              f"(cached at {LLM_RESPONSE_CACHE.name})...")
    plans = make_submissions(samples, use_llm_response=args.llm_response)
    est = estimate(plans, with_ragas=args.ragas,
                   with_llm_response=args.llm_response,
                   n_samples=len(samples))
    print(f"\nPlan: {est['submissions']} submissions to variA/Bly")
    print(f"  ETA:  ~{est['eta_min']} min")
    cost_breakdown = []
    if args.llm_response:
        cost_breakdown.append(f"~${len(samples) * 0.001:.3f} for response generation")
    if args.ragas:
        cost_breakdown.append(f"~${est['submissions'] * 0.03:.2f} for RAGAS")
    print(f"  Cost: ${est['openai_cost_usd']:.3f}" +
          (f"  ({'; '.join(cost_breakdown)})" if cost_breakdown else "  (variA/Bly is free)"))

    if est["openai_cost_usd"] > 1.0:
        confirm = input(f"\nProceed with ~${est['openai_cost_usd']:.2f} OpenAI spend? [y/N] ")
        if confirm.strip().lower() != "y":
            print("aborted")
            return 1

    print("\nSubmitting to variA/Bly...")
    submitted = submit_via_sdk(plans)

    out = {
        "n_samples": len(samples),
        "n_submissions": len(submitted),
        "plans": submitted,
        "scoring_note": (
            "Submissions are async — variA/Bly scoring completes within "
            "~30-300s after each submission. Query the DB for "
            "dimensional_scores/claim_analyses by evaluation_id to compute "
            "faithfulness/hallucination_rate/misinformation per (subset, group)."
        ),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    print("\nNext step: wait ~10 min for async scoring to complete, then run the analyzer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
