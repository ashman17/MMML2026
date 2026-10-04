# Idea 3 Experiment Spec: Transformation Consistency on MMSI-Bench

- **Owner:** Wen-Chi (Vicky) Tsai
- **For:** 11-777 Assignment 2 (due Sun Oct 4, 8pm ET), one analysis section (~1 page)
- **Code:** `idea3_consistency/mmsi_consistency.py`, `run_babel.sbatch`
- **Compute:** CMU Babel, 1 GPU, outputs in `/data/user_data/vickytsa/idea3/`

---

## 1. Goal

Test whether a VLM's multi-image spatial answers stay the same under input
transformations that do **not** change the correct answer. Then test whether
disagreement across those transformations is a usable signal for wrong answers.

This is a multimodal analysis: every transformation edits the image side, the
text side, or both together. The two sides must stay consistent with each other.

## 2. Hypotheses

| ID | Hypothesis | Evidence that supports it |
|---|---|---|
| H1 | The model is not invariant to answer-preserving transformations | Flip rate clearly above 0 for each transform |
| H2 | Disagreement across variants predicts wrong answers | AUROC of disagreement score >= 0.6; accuracy when consistent > when inconsistent |
| H3 | Voting across transformations beats self-consistency at equal compute | Transformation-vote accuracy > self-consistency accuracy |
| H4 | Mirror failures come from left/right reasoning | Mirror flip rate higher on questions containing left/right/clockwise words |
| H5 | Option-position bias (found by Zhengfei without images) persists with images | Shuffle flip rate with images > 0; predicted letter distribution is skewed |

## 3. Relation to teammates' analyses

- **Zhengfei (language only):** found that answer wording and option position
  affect predictions without images. This analysis repeats the shuffle test
  **with images** (H5), so the two results together show whether visual input
  removes the shortcut.
- **Zhengfei's reference-frame finding** (about 86% of camera anchors refer to
  image 2 or later) motivates the **reorder** test: does the model track
  viewpoints by label ("Image 2") or by position in the input?
- **Human labeling analysis:** the validity check in Section 8 doubles as a
  small human check on transformed questions.

## 4. Data

- **Dataset:** MMSI-Bench (1,000 multiple-choice questions, 11 question types, 2+ images each)
- **Pilot:** N = 100, stratified by `question_type`, `--seed 0`
- **Main run:** N = 300, same seed (report it as a separate run from the pilot)

## 5. Model and inference

| Setting | Value |
|---|---|
| Main model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Optional second model | `Qwen/Qwen2.5-VL-3B-Instruct` (scale comparison) |
| API model (no GPU needed) | a vision model on the CMU LiteLLM gateway (e.g. `gpt-4o`), via `--backend openai` |
| Backend | vLLM, tensor parallel 1 |
| Decoding (variants) | greedy, temperature 0, max_tokens 64 |
| Image resize | long side 768 px (`--max_side 768`) |
| Prompt suffix | "Answer with the option's letter from the given choices directly. Enclose the option's letter within ``." |
| Answer extraction | MMSI-Bench official regex order: ``` ``X`` ```, `` `X` ``, `{X}`, bare letter |

## 6. Variants (answer-preserving by design)

| Variant | Image side | Text side | Why the gold letter is unchanged | Applied when |
|---|---|---|---|---|
| `orig` | images in order, labeled "Image k:" | unchanged | baseline | always |
| `mirror` | every image flipped horizontally | left/right, leftmost/rightmost, clockwise/counterclockwise swapped in question and options | whole scene mirrored consistently | question has no compass words, text/signs, "right angle", idioms |
| `reorder` | images shown in reversed order, each keeps its original label | note added: "images are not shown in numerical order" | question refers to labels, not positions | 2+ images |
| `shuffle` | unchanged | options permuted; prediction mapped back | pure relabeling | options parse cleanly |

## 7. Baseline: self-consistency

Same original input, K samples at temperature 0.7 (top_p 0.95), where K equals
the number of variants for that question. Majority vote, ties broken by the
greedy original answer. This matches the number of model calls of
transformation voting.

## 8. Procedure

### Step 0: Setup (once, on a Babel login node)
```bash
conda create -n idea3 python=3.11 -y && conda activate idea3
pip install "vllm>=0.8" pandas pyarrow pillow "huggingface_hub[cli]"
```
Make sure `run_babel.sbatch` is downloaded locally from iCloud before copying the folder to Babel.

### Step 1: Dry run (no GPU)
```bash
python mmsi_consistency.py --parquet <MMSI_Bench.parquet> --backend mock --n 20 --out results_mock
```
Check that `predictions.jsonl` and `report.md` are written.

### Step 2: Pilot, N = 100
```bash
sbatch run_babel.sbatch
```
Expected time: 15 to 30 minutes including model load.

### Step 3: Validity check (human, before trusting numbers)
Open `predictions.jsonl` and inspect:
- 20 `mirror` rows: are left/right swaps correct in both question and options? Is the gold answer still correct for the flipped images?
- 10 `reorder` rows: does each label still match the right image?

Record each row in `validity_check.csv` with columns
`id, variant, valid (y/n), issue`. Report the % valid in the paper.
If more than 10% of mirror rows are invalid, fix `MIRROR_BLOCK` / `MIRROR_MAP` and rerun.

### Step 4: Main run, N = 300
```bash
N=300 sbatch run_babel.sbatch
```

### Step 5 (optional): Second model
```bash
N=300 MODEL=Qwen/Qwen2.5-VL-3B-Instruct sbatch run_babel.sbatch
```

### Step 5b (no GPU needed): API model via the CMU LiteLLM gateway
Runs from a laptop or Babel login node. Also a fallback if Babel is busy.
```bash
pip install openai
export LITELLM_API_KEY=...                       # never hard-code or commit the key
python check_gateway.py                          # list models on the gateway
python check_gateway.py --model gpt-4o           # confirm the model accepts images
python mmsi_consistency.py --parquet <MMSI_Bench.parquet> --n 100 \
    --backend openai --model gpt-4o --workers 8 --out results_gpt4o
```
- Cost: about 2 x (number of variants) calls per question, so roughly 700 image calls for N=100. Start with `--n 20`.
- Every response is cached in `results_*/api_cache.jsonl`; rerunning the same command resumes and does not repeat paid calls.
- Reasoning models (o-series, gpt-5, etc.) need `--max_tokens 2048` or they may return nothing.
- This gives an open-source vs. proprietary comparison (Qwen2.5-VL-7B vs. GPT-4o) for the report.

### Step 6: Analysis
```bash
python mmsi_consistency.py --analyze results_<jobid>/predictions.jsonl
```
Then the extra analyses in Section 10.

## 9. Metrics (computed by `--analyze`)

| Metric | Definition |
|---|---|
| Original accuracy | greedy `orig` prediction == gold |
| Variant accuracy | per transform, prediction (mapped back for shuffle) == gold |
| Flip rate | % of questions where variant prediction != `orig` prediction (unparsable counts as a flip) |
| Mirror flip by direction words | mirror flip rate split by whether the question has left/right/clockwise words (H4) |
| Consistency rate | % of questions where all variants agree |
| Acc. when consistent / inconsistent | original accuracy within each group (H2) |
| AUROC | disagreement score = fraction of variants that differ from `orig`; label = `orig` is wrong (H2) |
| Transformation vote | majority over all variants, tie broken by `orig` (H3) |
| Self-consistency | majority over K samples, same tie break (H3) |
| Per question type | orig acc, consistency, transform vote, self-consistency for each of the 11 types |

## 10. Extra analyses to add (not yet in the script)

1. **Per-type flip rate for each transform** (type x transform table), to see which skills break under which transform.
2. **Option-position bias with images (H5):** distribution of predicted option *position* on `orig` vs `shuffle`, compared with Zhengfei's language-only numbers.
3. **Uncertainty:** bootstrap 95% CI (1,000 resamples) for accuracy, flip rate, AUROC, and the vote vs. self-consistency gap.
4. **Paired test for H3:** McNemar test on transformation vote vs. self-consistency correctness.
5. **Qualitative examples:** 2 to 3 cases (use `export_example.py`; current examples in `idea3_assets/`), at least one where the model is wrong and inconsistent.

## 11. Figures for the report

1. **Flip rate by transform** (bar chart), with mirror split into with / without left/right words.
2. **Accuracy vs. agreement:** original accuracy grouped by number of variants agreeing with `orig`, with AUROC in the caption.
3. **Vote comparison by question type:** greedy vs. self-consistency vs. transformation vote.
4. (Appendix or inline) one qualitative example: original vs. mirrored images, question text, predictions per variant.

## 12. Decision rules: what goes into the Discussion

| Result | Interpretation | Improvement direction for midterm |
|---|---|---|
| AUROC >= 0.6 | disagreement is a usable error signal | use consistency as a gate: only call geometry tools (Idea 2) or extra reasoning when variants disagree |
| Transformation vote > self-consistency | input diversity helps more than sampling diversity | test-time transformation voting as a training-free baseline |
| Mirror flips concentrated on left/right questions | systematic left/right bias | mirror-consistency training: augment with mirrored pairs, add a consistency loss between original and mirrored predictions |
| Reorder flips high | model tracks images by position, not label | explicit viewpoint/label tokens in the input; links to Zhengfei's reference-frame finding |
| Shuffle flips high with images | option-position shortcut survives visual input | always evaluate over multiple option orders (team-wide protocol) |
| Flip rates near 0 and AUROC ~0.5 | model is stable, disagreement is not informative | report as a negative result; Idea 3 becomes an evaluation protocol for the other ideas rather than a method |

## 13. Report section outline (~1 page)

1. Hypotheses (H1 to H5, short)
2. Setup: model, N, variants table (compact), baseline
3. Validity check result (% valid)
4. Results: Figure 1 + Figure 2, one sentence on H3 with Figure 3 or a small table
5. Qualitative example (1)
6. Discussion: which hypotheses held, and the improvement direction from Section 12

Also write: own AI usage disclosure and own contribution lines (appendix page).

## 14. Timeline

- [ ] **Fri:** setup on Babel, dry run, submit N=100 pilot
- [ ] **Sat AM:** validity check (Step 3), fix and rerun if needed
- [ ] **Sat:** N=300 run (+ optional 3B, or an API model via the gateway), `--analyze`, extra analyses, figures, push code + results to team GitHub
- [ ] **Sat night:** share numbers with the team
- [ ] **Sun AM:** write section in `report.tex`, Discussion paragraph, AI disclosure
- [ ] **Sun PM:** proofread, check 5-page limit, submit before 8pm ET

## 15. Known caveats

- With N = 100, differences of a few points are noise. Treat the pilot as a feasibility signal; rely on N = 300 + CIs for claims.
- Mirror is skipped for questions with compass directions, text, or signs, so mirror results cover a subset. Report n for every transform.
- Unparsable answers count as flips, which can inflate flip rates. Report the number of unparsable outputs.
- Reorder adds a short note to the prompt, so it changes the text slightly as well as the image order.
