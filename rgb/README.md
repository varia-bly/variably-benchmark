# RGB Benchmark

Head-to-head comparison of variA/Bly grounding scoring vs RAGAS on the
public [RGB benchmark](https://github.com/chen700564/RGB)
(Chen et al., 2023).

## Headline numbers (October 2026)

n = 592 evaluations (296 RGB samples × 2 test cases each): each answer checked against the
passages that contain it, and against distractor passages.

| Scorer | Cases scored | Grounded approved (recall) | Distractors approved | Precision | AUC |
|---|---|---|---|---|---|
| **variA/Bly 4.2.1** | **592/592** | **97.6%** | **27.7%** | **77.9%** | 0.851 |
| RAGAS faithfulness (gpt-4o-mini) | 592/592 | 97.0% | 38.2% | 71.8% | 0.852 |
| DeepEval Hallucination (gpt-4o-mini) | 592/592 | 42.9% | 4.4% | 90.7% | 0.852 |
| DeepEval Faithfulness (gpt-4o-mini) | **294/592** | 52.4% | 38.5% | 27.0% | 0.571 |

variA/Bly 4.2.1 was scored on 2026-10-24 through its full production path; the RAGAS and DeepEval
scores are from April 2026 on the identical 592 answer-and-passage pairs, joined case by case in
[`rgb/results/rgb_2026-10_four_scorers.json`](./results/rgb_2026-10_four_scorers.json).
Recompute the table with `python3 verify_2026_10.py` (standard library only).

RGB's distractor passages often contain the answer, so absolute false-approval rates overstate
every scorer's errors — compare scorers on the same cases.

For the full narrative and methodology, see
[variably.dev/benchmarks/rgb](https://www.variably.dev/benchmarks/rgb).

### What changed since April 2026

The April 2026 edition reported precision 89.9%, recall 54.4% and a 6.1% false-approval rate.
That scorer gave every case 0 or 1 and rejected 135 of 296 correct answers; its low
false-approval rate came from that strictness. Its RAGAS comparison was also joined to a
different run of the variA/Bly scorer (`rgb_ragas_comparison.json` carries that run's scores:
23.3% distractors approved). The April files stay in `results/` as the historical record; the
October file joins every scorer to one run, case by case.

---

## Layout

```
rgb/
├── README.md            (this file)
├── runner.py            ← Submits 592 evaluations through variA/Bly
├── analyze.py           ← Reads variA/Bly scores, computes per-group metrics
├── run_ragas.py         ← Submits the same 592 evaluations through RAGAS
├── compare.py           ← Joins the April 2026 score files, prints that comparison table
├── verify_2026_10.py    ← Recomputes the October 2026 four-scorer table from the per-case file
├── diagnose.py          ← Per-claim failure analysis (low-entailment, contradiction, numeric_mismatch)
├── false_negatives.py   ← Targeted look at cases where variA/Bly=0 but truth=1
├── inspect_negatives.py ← Inspect specific distractor cases by rgb_id
├── results/             (raw outputs)
│   ├── rgb_2026-10_four_scorers.json ← October 2026: one row per case, every scorer's score
│   ├── rgb_results.json              ← Submission plan: per-sample (query, response, refs, evaluation_id)
│   ├── rgb_labeled.json              ← variA/Bly per-sample faithfulness scores joined with ground_truth
│   ├── rgb_ragas_comparison.json     ← RAGAS per-sample faithfulness scores
│   └── rgb_llm_responses.json        ← Cached gpt-4o-mini responses (so re-runs are free)
└── data/                (gitignored target for the RGB dataset clone)
```

---

## How to verify the numbers (no rescoring required)

If you just want to confirm our headline numbers from our raw outputs,
no API keys needed:

```bash
cd rgb
python3 compare.py
```

`compare.py` reads `results/rgb_labeled.json` and
`results/rgb_ragas_comparison.json`, joins them by `(rgb_id, group)`,
and prints AUC / accuracy / precision / recall / FPR / FNR for each
scorer. The output reproduces the headline table from this README and
the technical brief at variably.dev/benchmarks/rgb.

---

## How to re-run from scratch

If you want to actually re-score the 592 samples, both scorers cost
money and require accounts.

### Step 1 - Clone the RGB dataset

RGB is not redistributed here.

```bash
cd rgb/data
git clone --depth 1 https://github.com/chen700564/RGB ./RGB
cd ../..
```

After cloning, `rgb/data/RGB/data/en.json` should exist.

### Step 2 - Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 - Set environment variables

```bash
# variA/Bly API key (sign up at https://www.variably.dev)
export VARIABLY_API_KEY=vb_...

# OpenAI key (for the realistic response generator and RAGAS judge)
export OPENAI_API_KEY=sk-...
```

### Step 4 - Generate responses + score with variA/Bly

```bash
python3 runner.py --limit 300 --llm-response
```

`--limit 300` runs the full English subset (296 samples after filtering,
4 are skipped because gpt-4o-mini returns NO_ANSWER on them).

`--llm-response` generates realistic 1-2 sentence responses via
gpt-4o-mini from the top-5 positive chunks. Cost: ~$0.30 for the full
run (cached - re-runs are free).

This writes:

- `results/rgb_results.json` - the submission plan with the
  variA/Bly evaluation IDs
- `results/rgb_llm_responses.json` - cached realistic responses

### Step 5 - Pull variA/Bly scores back

variA/Bly scores asynchronously - allow a few minutes for the 592
evaluations to land before running:

```bash
python3 analyze.py
```

This pulls the per-sample faithfulness scores from variA/Bly and joins
them with the ground-truth labels into `results/rgb_labeled.json`.

### Step 6 - Score the same 592 evaluations with RAGAS

```bash
python3 run_ragas.py
```

Cost: ~$17.76 (gpt-4o-mini judge × 592 evaluations × default RAGAS
pipeline). Wall time: ~28 minutes.

This writes `results/rgb_ragas_comparison.json`.

### Step 7 - Compare

```bash
python3 compare.py
```

Prints the head-to-head table.

---

## How the test cases are constructed

For each RGB sample, two test cases:

```
Positive:  query + response + 5 chunks that contain the answer
           ground_truth = 1  (scorer should say "grounded")

Negative:  query + response + 5 distractor chunks
           ground_truth = 0  (scorer should say "not grounded")
```

The same response is sent in both cases. Only the references swap.
This isolates scorer behaviour from response quality.

The "response" itself is generated by gpt-4o-mini given
`(query, top-5 positive chunks)`, asked to answer the question using
only those chunks. The LLM cannot tell which test case its output will
be used for - it just sees a question and good context.

---

## Caveats

- **RGB distractor labels are slightly leaky.** For famous events
  (Beijing Olympics, Russia-Ukraine, Activision-Microsoft acquisition)
  some "distractors" mention the answer in passing. We estimate ~5%
  label noise on the negative side. Both scorers see the same data, so
  it doesn't change the *comparison* - but it puts a ceiling on how
  clean the FPR can ever be on this dataset.
- **The variA/Bly side requires a Variably API key.** Scoring runs on
  Variably's hosted infrastructure; the algorithm is proprietary. The
  runner uses the public `variably-sdk` to submit + read back the
  verdict.
- **The RAGAS side is fully self-contained.** Just an OpenAI key.
- **AUC isn't the only metric.** RAGAS leads on AUC (0.864 vs 0.748).
  variA/Bly leads on precision and FPR. The right pick depends on which
  failure mode matters more for your AI workflow - see the technical
  brief at [variably.dev/benchmarks/rgb](https://www.variably.dev/benchmarks/rgb)
  for the discussion.
