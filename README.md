# variA/Bly Benchmarks

Public benchmark suite for [variA/Bly](https://www.variably.tech)'s grounding
scoring vs popular alternatives. Reproducible scripts, raw per-sample outputs,
and head-to-head comparison tables on public RAG datasets.

> **The number nobody else publishes:** RAGAS approves **38.2%** of distractor
> passages on the public RGB benchmark as "grounded." variA/Bly holds to
> **6.1%** - over 6× cleaner. Same 592 question + answer + context tuples,
> both scorers, scripts in this repo. See [`rgb/`](./rgb) for the full run.

---

## What's in this repo

| Path | What it is |
|---|---|
| [`rgb/`](./rgb) | RGB benchmark (Chen et al., 2023) - 296 samples × 2 test cases each = 592 evaluations. Includes runner + analyser + RAGAS comparison + raw outputs. |

Each benchmark directory is self-contained: scripts, raw outputs,
dataset clone target, and a per-benchmark README.

Future benchmarks will land as new top-level directories
(`fever/`, `aggrefact/`, `pubmedqa/`, ...). One dataset = one directory.

---

## Headline results (RGB, n=592, April 2026)

| Metric | variA/Bly | RAGAS | Notes |
|---|---|---|---|
| **Precision @ 0.5** | **89.9%** | 71.8% | Of cases flagged "grounded", how many actually are. |
| **FPR on distractors** | **6.1%** | 38.2% | Of cases that aren't grounded, how many were wrongly approved. **The compliance number.** |
| Recall @ 0.5 | 54.4% | **97.0%** | Of real groundings, how many were caught. |
| AUC | 0.748 | **0.864** | Threshold-independent ranking quality. |

**Read the full report:** [variably.tech/benchmarks](https://www.variably.tech/benchmarks)

---

## Verifying the numbers vs re-running the scorers

Two separate things, two different cost / friction profiles.

### 1. Verify the math (zero-cost, zero-account)

The raw per-sample outputs from our run live in
[`rgb/results/`](./rgb/results) as JSON. Use them to verify the
aggregations directly - no API key, no rescoring needed.

```bash
git clone https://github.com/varia-bly/variably-benchmark
cd variably-benchmark/rgb
python3 compare.py        # Prints AUC / precision / recall / FPR / FNR
                          # straight from the committed scores.
```

This is the strongest form of verification a third party can do
without owning the scorer: take our raw per-sample outputs and
recompute the published headlines. If our compare.py disagreed with
manual hand-calc on the same JSON, you'd catch it instantly.

### 2. Re-run the scorers (real cost, accounts required)

Re-running the scorers costs money and requires accounts.

| Scorer | What you need | Per-eval cost |
|---|---|---|
| **RAGAS** | OpenAI API key | ~$0.030 (gpt-4o-mini judge with default RAGAS pipeline) |
| **variA/Bly** | A free API key from [variably.tech](https://www.variably.tech) | $0.015 (SEU pricing, all-in) |

variA/Bly's grounding scorer runs on Variably's hosted infrastructure -
the algorithm is proprietary, so the runner submits via the public
[`variably-sdk`](https://pypi.org/project/variably-sdk/) and reads the
verdict back. To reproduce our variA/Bly column from scratch, sign up
at [variably.tech](https://www.variably.tech) and grab an API key
(free tier covers re-running this benchmark).

To reproduce the RAGAS column independently, no Variably account is
needed - just an OpenAI key. See [`rgb/README.md`](./rgb/README.md)
for the exact commands.

---

## Why these benchmarks

variA/Bly's grounding scorer is built around a specific design
tradeoff: **precision-first scoring for regulated, customer-facing AI**
(healthcare, finance, legal, insurance). LLM-as-judge tools like RAGAS
optimise for recall - catch every real grounding - which means they
also approve more wrong passages as "grounded" by default.

For low-stakes search and summarisation, that's fine. For workflows
where a wrong "grounded" verdict can produce a compliance breach, it
isn't.

The RGB benchmark in this repo measures this tradeoff explicitly. The
**FPR-on-distractors** column is the headline number every other
benchmark vendor leaves out, and the one that decides which scorer is
appropriate for which AI workflow.

---

## Reproducibility statement

- **Dataset:** RGB by Chen et al. (2023), [github.com/chen700564/RGB](https://github.com/chen700564/RGB).
  Public, labeled at the chunk level. We use the `en` subset.
- **Sample count:** 296 RGB samples × 2 test cases each = 592
  evaluations per scorer. Each sample produces a positive case
  (response paired with chunks containing the answer) and a negative
  case (paired with distractor chunks).
- **Realistic responses:** generated with gpt-4o-mini from the
  positive chunks, cached so re-runs are free. The same response is
  used in both positive and negative cases - only the references swap.
- **RAGAS judge:** gpt-4o-mini with default RAGAS faithfulness pipeline.
  Measured cost across our run: ~$17.76 (~$0.030 per eval).
- **Threshold:** 0.5 for precision/recall/FPR/FNR. AUC is
  threshold-independent.
- **Run dates:** April 28-30, 2026.

The exact run plan (per-sample inputs, evaluation_ids) is in
[`rgb/results/rgb_results.json`](./rgb/results/rgb_results.json).
The variA/Bly per-sample scores are in
[`rgb/results/rgb_labeled.json`](./rgb/results/rgb_labeled.json).
The RAGAS per-sample scores are in
[`rgb/results/rgb_ragas_comparison.json`](./rgb/results/rgb_ragas_comparison.json).

---

## License

Apache License 2.0 - see [LICENSE](./LICENSE).

The dataset itself is the property of its respective authors and is
not redistributed here. RGB clone instructions are in
[`rgb/data/README.md`](./rgb/data/README.md).

---

## Get in touch

- Email: [info@variably.tech](mailto:info@variably.tech)
- Web: [variably.tech](https://www.variably.tech)
- LinkedIn: [linkedin.com/company/variably](https://www.linkedin.com/company/variably/)

If you spot an error in the benchmark scripts or have a suggestion for
a benchmark we should add, open an issue or a pull request. We mean it
when we say "reproducible" - if you can't make our numbers reproduce
from this repo (modulo signup for the variA/Bly API key), that's a bug
we want to fix.
