# Language-Only Analysis of MMSI-Bench

Guiding question: *How does language specify the spatial problem, what reasoning
demands does it introduce, and what clues might let a model answer without the
intended visual evidence?*

The analysis has four parts, plus a human audit that validates the automatic tags:

| Part | Question | Feeds |
|---|---|---|
| A. Frames and operations | What spatial interpretation does the wording require? | Idea 4 extraction fields; Idea 3 transformation eligibility; Idea 2 signal choice |
| B. Entity descriptions | What visual evidence must the wording connect to? | Idea 2 grounding; Idea 4 node types |
| C. Shortcuts | Can the answer be chosen without images? | Idea 3 controls; blind baseline for all ideas |
| D. Reasoning traces | What do reference solutions use beyond the question? | Idea 1 claim types; Idea 4 pruning |

## Layout

Scripts and library modules stay in this directory and import one another by filename, which is also what the Colab bundle expects. Tests are in `tests/`. The blind-model notebook is in `colab/`.

| Path | Role |
|---|---|
| `00_build_dataset.py` … `09_audit_agreement.py` | One script per step. Each writes under `outputs/`. |
| `common.py`, `lexicon.py`, `entities.py`, `plotting.py`, `shortcuts.py`, `traces.py`, `additions.py`, `blind_vlm.py` | Shared code imported by the step scripts. |
| `splits/language_explore_confirm.json` | Frozen 500/500 explore/confirm split (seed 0). |
| `results/figures/`, `results/tables/` | Summary figures and tables kept in the repo. |
| `outputs/`, `cache/` | Local scratch: logs, raw JSON, embeddings, and per-question tables. Not committed. |
| `audit/` | Labeling packet and review notes. Kept locally; not committed. |

## Setup

```bash
conda activate 11777-project
cd MMML2026/analyses/language
pip install -r requirements.txt
python -m spacy download en_core_web_sm
mkdir -p cache
curl -sSfL -o cache/lvis_v1_categories.py \
  https://raw.githubusercontent.com/facebookresearch/detectron2/main/detectron2/data/datasets/lvis_v1_categories.py
```

The blind-model step needs a GPU environment. Its extra packages are in `requirements-blind-vlm.txt`.

**macOS 27 note.** The pip wheel of scipy 1.15.3 for Python 3.10 fails to import: its PROPACK binary is rejected by the loader with "zero-fill section type, but offset field is not zero". The fix is Anaconda's April 2026 rebuild, which leaves the pip numpy untouched:

```bash
pip uninstall -y scipy
conda install -n 11777-project --override-channels -c defaults --no-deps \
  "scipy=1.15.3=py310he2f6300_2" libgfortran=15.2.0 libgfortran5=15.2.0 \
  libopenblas=0.3.31 llvm-openmp=22.1.2 libcxx=22.1.2 "blas=1.0=openblas"
```

Data is read from `11777Project/MMSI-Bench/.analysis/`; set `MMSI_DATA_DIR`
to read it from elsewhere. Two artifacts are used:

- `MMSI_bench.tsv`: the legacy release used by the official evaluation code (questions, answers, categories, base64 images).
- `MMSI_Bench.parquet`: the Hugging Face release, which adds `thought` (the human reasoning trace), `difficulty` and `mean_normed_duration_seconds`.

Scripts write to `outputs/`. The repository keeps only the summary tables and figures under `results/`. Per-question tables, run logs, raw JSONL, and `.npy` embeddings stay in `outputs/` and are not committed.

## Committed results

Figures in `results/figures/`: `01_heatmap.png`, `01_transform_eligibility.png`, `02_entity_types.png`, `03_embeddings.png`, `language_tsne.png`, `04_shortcuts.png`, `06_traces.png`, `07_additions.png`, `language_additions.png`.

Tables in `results/tables/`: the category and summary CSVs named in each step below. Step E uses the full Colab run (`05_blind_vlm_summary.csv`, `05_blind_vlm_by_category.csv`). Token-level dumps, per-question rows, logs, and raw model JSON remain only in `outputs/`.

## Leakage rules

- Answer-shortcut rules are designed on the **explore** half and reported on the **confirm** half (`splits/language_explore_confirm.json`).
- Reasoning traces are analyzed separately and are never used as features for shortcut tests or as input to any extractor.
- The audit items (`audit/`) are drawn from the confirm half.

## Running and checks

Each script prints `[PASS]` / `[FAIL]` lines for its built-in checks and
`[EXPECTED]` / `[SURPRISE]` lines for the expectations stated below. Unit tests
for the parser, the lexicon and the entity typing:

```bash
python -m pytest tests -q
```

## Step log

Each step records the expected result **before** running and the actual result afterwards.

### Step 0: Text cache, split, manifest (`00_build_dataset.py`)

**What it does.** Reads only the text columns of the TSV and counts images
per question without decoding them. Splits each question into a stem and four
options. Joins the Parquet fields on `id`. Writes `cache/mmsi_text.parquet`, a
category-stratified 50/50 split (seed 0), and `outputs/00_manifest.json`
containing file checksums.

**Expected.**
- 1,000 rows with unique ids; all 1,000 questions parse into a stem plus A-D (the pilot parse succeeded on all 1,000).
- Category, image-count and answer-letter counts equal those in `Agentic Project Notes/README.md` section 6.2. Answers: A 265, B 250, C 255, D 230.
- Parquet join: question and answer text should match the TSV for every id. **Risk:** the maintainers announced difficulty-stratified resampling on 2025-10-23, so the Parquet release may contain different questions than the TSV. Any mismatch decides which artifact is the source of truth for each part.
- Split: 500/500, with each category's halves differing by at most one question.

**Actual (2026-10-01).** All 9 checks passed in about 20 seconds.
- The TSV and Parquet agree on every question string and answer, so the two releases contain the same 1,000 items, and every row has a non-empty trace.
- Difficulty in the Parquet is 334 easy / 333 medium / 333 hard. This is the later tertile-style revision, not the paper's 605/262/133, so the paper's difficulty counts should not be quoted alongside these labels.
- Split counts per category are in `splits/language_explore_confirm.json`; checksums are in `outputs/00_manifest.json`.
- `tests/test_common.py` (7 tests) covers: comma-separated options, commas inside an option (id 668, "First front left, then front right"), sentence options ending in periods, and malformed inputs.

### Step A: Frames, answer spaces, operations, transformation validity (`lexicon.py`, `01_frames_operations.py`)

**What it does.** Applies regex tags (`lexicon.py`) separately to the stem and the options. Tags were
written after reading only the explore half. The tag families are:

- **Answer space** (from the options): one of 10 types.
- **Frame anchor** (from the stem): one of 8 types.
- **Camera anchor index**: the image whose camera is "me/you".
- **Stated conditions**: premises, overlap, same location, sequence, image references, hedges, abstention options.
- **Operations**: 12 multi-label flags.
- **Idea 3 eligibility** for horizontal mirroring and image reordering.

`tests/test_lexicon.py` pins 21 hand-labeled phrasings.

Committed copies: `results/figures/01_heatmap.png`, `results/figures/01_transform_eligibility.png`, `results/tables/01_tag_rates_by_category.csv`, `results/tables/01_anchor_image_by_category.csv`, `results/tables/01_operation_signatures.csv`. The per-question tags stay in `outputs/01_tags.csv`.

**Expected** (from reading the explore half; each is printed as `EXPECTED` or `SURPRISE`):
- Measurement is mostly a metric answer space (at least 80%).
- Cam-Obj and Cam-Reg are mostly egocentric directions (at least 70%) and mostly camera-anchored (at least 70%).
- Cam-Cam has a 15-35% block of signed-axis questions ("+Y up, -Z forward").
- Obj-Obj, Obj-Reg and Reg-Reg use compass frames in at least 40% of questions, usually set by a premise ("the TV is on the east side").
- **Most camera-anchored Cam-Obj/Cam-Reg questions put "you" at the second or last image, not the first** (at least 50%). This would contradict Idea 4's default of anchoring at the first camera.
- Mirroring needs a stem rewrite or is unsafe for at least 40% of questions, because of compass, axes, handedness, text or traffic.
- Reordering is unsafe for at least 70% of motion questions.
- The lexicon leaves fewer than 15% of position questions without a frame.

**Actual (2026-10-01).** All 9 expectations held after two lexicon fixes, each now pinned by a test (28 tests pass):
1. The bare word "area" had marked region names ("sleeping area") as measurements.
2. Photos used as places ("the location where the second photo was taken") were not counted as camera anchors.

Findings:
- **Answer spaces.**
  - Cam-Obj 80% and Cam-Reg 82% egocentric directions.
  - Obj-Obj, Obj-Reg and Reg-Reg use compass answers in 44%, 39% and 54% of questions.
  - Measurement is 95% metric. Appearance mixes compass "which side" questions (41%) with counting (33%).
  - Motion: Obj contains signed-axis (11%) and image-order (11%) questions. Multi-step is the most mixed: 16% statements, 15% compass, 8% image order.
- **Frame anchors.**
  - Cam-Obj and Cam-Reg are 94% and 92% camera-anchored.
  - Obj-Obj, Obj-Reg and Reg-Reg are 51-58% compass, plus 14-33% hypothetical agents ("when you are cooking in the kitchen area") and 13% object-defined fronts in Obj-Reg.
  - Only 2.3% of position questions leave a relative frame unstated. Most of the 16 position questions with no frame are actually height or length comparisons filed under position categories.
- **Which camera is "you".** Of the 150 camera-anchored Cam-Obj/Cam-Reg questions, **129/150 (86%) anchor at image 2 or later** ("At the moment of the last image", "When you took the second photo"); the camera-only subset (`frame=camera`) is 126/147 (85.7%). Across all 324 anchored questions the figure is 66%. A canonical frame fixed at the first camera would make most of these questions require an extra frame conversion. *Implication for Idea 4:* anchor at the camera the question specifies.
- **Stated conditions.** 63% of questions contain a premise (a declarative sentence, a parenthetical, or an if/assume/given clause). Language often states image relations directly: overlap, "same location", "60 degrees apart", consecutive shooting. 8.7% offer an abstention option ("Cannot be determined").
- **Operations cut across categories.** 25 operation signatures appear in at least 3 official categories, covering 803 questions. For example, "direction + premise chain" appears in 8 categories.
- **Mirroring (Idea 3).** 41% of questions need a stem rewrite or cannot be mirrored.
  - Rewrites dominate the object and region categories (42-45% because of east/west).
  - Unsafe cases concentrate in Cam-Cam (30%, signed axes), Motion: Obj (44%: axes, traffic and robot handedness) and Multi-step.
- **Reordering (Idea 3).**
  - Unsafe for 97% of Motion: Cam and 86% of Motion: Obj, and for 45-50% of Cam-Obj/Cam-Reg because of "last image" or "moment".
  - 91% of Cam-Cam questions are label-dependent: "first photo" must keep meaning label 1.
  - At least 80% of object/region and attribute questions are safe.

### Step B: Entity descriptions and detector vocabulary (`entities.py`, `02_entities.py`)

**What it does.** Parses stems with spaCy `en_core_web_sm`. Options are parsed
too when they name entities (answer space "entity" or "statement"). Every noun
chunk becomes a candidate mention. Chunks are dropped when their head is a
pronoun, an image reference, a direction word or a query word ("direction",
"relation").

Each mention gets one **type**:
- object
- named room ("kitchen")
- functional area ("toothbrushing area")
- generic area
- part or surface ("the top surface")
- camera
- quantity ("the height"), which is excluded from entity counts

Each grounded mention also gets **descriptor modes**, read from the dependency parse:
- attribute ("green", "wooden")
- relational ("next to the sofa", "hanging on the wall")
- image-anchored ("from Image 1")
- in-image position ("on the right in Figure 2")
- bare, when none of the above apply

Object head nouns are matched against COCO-80 and the LVIS v1 names and synonyms (1,203 categories), using the head lemma or its compound phrase.

The script writes `outputs/02_mentions.csv` (one row per mention), `outputs/02_entity_summary_by_category.csv`, and `outputs/02_entity_types.png`. The summary table and figure are copied to `results/tables/` and `results/figures/`. `tests/test_entities.py` pins the parses of real phrasings.

**Why it matters.**
- **Idea 2.** Category names alone can drive a detector or Grounding DINO. Relational, image-anchored and positional descriptions need a referring-expression step before VGGT or LightGlue evidence can be attached to the right instance.
- **Idea 4.** The mention types are the candidate node types of the world-state graph. Regions and parts are not objects and need their own node kinds.

**Expected** (from reading the explore half):
- Regions (rooms plus areas) are at least 30% of entity mentions in Cam-Reg, Obj-Reg and Reg-Reg, and at least 50% in Reg-Reg.
- Functional "X area" names ("sleeping area", "food storage area") are at least 40% of region mentions. Such names describe what an area is used for, not what it looks like.
- Measurement descriptions are precise: at least 25% of its grounded mentions are relational, image-anchored or in-image positional, compared with at most 15% in Cam-Obj, where questions usually name one object plainly.
- Parts and surfaces are at least 10% of Measurement mentions ("the top surface of the washbasin").
- At least 75% of object head nouns are LVIS categories, but at most 60% are COCO-80. A fixed COCO detector would miss many referents, while LVIS-scale or open-vocabulary detectors mostly cover the names.

**Actual (2026-10-01).** 6 of 8 expectations held; there were 2 surprises. Runtime is about 10 seconds, and 38 tests pass.

Fixes before the final run, each found by reading mentions in context and now pinned by a test:
- The LVIS file's header comment broke parsing.
- "diagram" was not treated as an image reference.
- "with a larger volume" was being counted as a relation.
- "northwest corner" was being counted as a functional area.
- The verb head in "the camera rotating" hid the camera.
- spaCy lemmatized "vases" to "vas", so vocabulary matching now also tries the surface form.

The final run found **3,009 mentions: 2,929 entity mentions and 80 quantities**. 615 came from options, and 816 questions have at least one grounded mention. (An earlier run counted 3,018/2,938; the saved CSV reflects the final 3,009/2,929 totals.)

- **Regions dominate the region categories, as expected.**
  - Regions are 81% of mentions in Cam-Reg, 39% in Obj-Reg and 64% in Reg-Reg. 35% of all questions mention at least one region.
  - Named rooms and functional areas appear about equally often: in Cam-Reg 41% are rooms and 35% functional areas; in Reg-Reg 31% and 30%.
- **Surprise 1: functional "X area" names are 35% of region mentions, not at least 40%.** The two halves give 32% and 38%. Named rooms ("living room", "bedroom", "kitchen") are just as common. The functional names that do appear ("sleeping area", "dining area", "handwashing area", "sofa lounge area") are defined by use, not appearance, so a grounding step must infer them from the objects they contain. The proportion is lower than expected, but the implication for Idea 4 holds: regions need their own node type, defined by member objects.
- **Most mentions are plain names.**
  - 72% of grounded mentions carry no attribute, relation, image anchor or in-image position. 12% need more than a name; 25% of questions have at least one such mention.
  - As expected, Measurement is the most precise category (26%: 15% image-anchored, such as "the round stone pillar on the right in Figure 2"), and Cam-Obj is the least (10%).
  - Parts and surfaces are 11% of Measurement mentions and 13% of Appearance mentions.
- **Surprise 2: only 61% of object heads are LVIS categories, even with lenient matching.**
  - The gap is architectural "stuff". Door (105 mentions), wall (59), window (54), stairs (49, including "staircase"), plant, television, pool and building are absent from LVIS but present in ADE20K-150.
  - Head-noun coverage: COCO-80 33%, LVIS 61%, ADE20K-150 70%, LVIS or ADE20K 84%. **298 of the 1,000 questions mention at least one structural landmark that LVIS lacks.**
  - The remaining misses are mostly robot-manipulation terms (gripper, robot, hand), Appearance-specific referents (brick, model, pile) and a few non-entities that slip through, such as "sense" and "state". The audit sample (Step H) found 87/104 automatic mention heads on the sheet and 87/98 sheet heads in the automatic list; most gaps are inflectional variants or different compound heads rather than category errors.
  - Note: ADE20K-150 was added *after* this surprise, to explain it, and the LVIS matching is lenient. Both coverage figures are therefore upper bounds.

**Implications.**
- **Idea 2, grounding.** A pure thing-detector vocabulary (COCO or LVIS) misses the walls, doors, windows and stairs that MMSI questions use as landmarks. Those are also the planar structures that VGGT point maps and LightGlue matches capture well. Grounding should combine an open-vocabulary detector with a stuff-aware segmenter, or ADE-style classes. About 1 in 4 questions also needs a referring-expression step: a relation, an image anchor or an in-image position.
- **Idea 4, graph nodes.** The graph needs at least four node kinds: objects, regions (named rooms plus functional areas defined by member objects), parts or surfaces (for measurement), and cameras. Image-anchored mentions ("the lamp in photo 2") tie a node to a specific view, which is a cross-view identity constraint the graph must represent.

### Step C: Embedding geometry, category probes, templates (`03_embeddings.py`)

**What it does.**
- Embeds every stem with `all-MiniLM-L6-v2` (384 dimensions, 256-token limit) and counts truncated stems. The model is cached in `cache/hf`.
- Projects the embeddings with PCA and t-SNE.
- Computes 5-nearest-neighbour category agreement (cosine) against 1,000 label permutations.
- Trains linear category probes (5-fold stratified cross-validation) on four feature sets:
  - MiniLM embeddings.
  - The interpretable Step A lexicon tags: about 60 features.
  - TF-IDF word 1-2 grams.
  - Empath's 194 general-purpose lexical categories, as a weak baseline.
- Clusters stems with k-means (k = 11) and compares the clusters with the official category, the answer space and the frame using adjusted mutual information (AMI).
- Flags template near-duplicates (stem cosine ≥ 0.95) and checks whether paired items share their answer.

Category prediction is not an answer shortcut, so it uses all 1,000 items. Near-duplicate answer sharing is reported per split.

**Why it matters.**
- If the 60 lexicon tags recover the categories almost as well as TF-IDF or MiniLM, the official taxonomy is largely a matter of operations and answer formats. That supports Idea 4's query-conditioned design over category-specific pipelines.
- Near-duplicate templates with shared answers would be a shortcut for Step D, and a leakage risk for any learned extractor (Ideas 1 and 4).

**Expected.**
- No stem exceeds 256 tokens.
- 5-nearest-neighbour category agreement is at least 0.5, against a permutation null of about 0.10.
- Agreement is highest (at least 0.7) for Motion: Cam and Measurement, and at most 0.45 for at least two of Obj-Obj, Obj-Reg, Reg-Reg and Multi-step, which share compass and premise templates.
- Probe macro-F1: TF-IDF ≥ MiniLM. The lexicon tags come within 0.15 of the best set. Empath is at most 0.35.
- Multi-step has the lowest per-category F1 in the best probe.
- k-means clusters align at least as well with the answer space as with the official category (AMI). In other words, the wording groups by answer format.
- At least 5% of questions have a near-duplicate stem. In near-duplicate pairs the correct letter agrees at about chance (at most 35%).

**Actual (2026-10-01).** 6 of 11 printed expectations held. The clean run
took 5 minutes 14 seconds; no stem was truncated (maximum 98 tokens).

- **The categories have substantial language structure, but are not clean
  clusters.** Five-nearest-neighbour agreement is 58.2%, versus 10.4% under
  1,000 label permutations (`p = .001`). Motion: Cam (96.5%), Cam-Cam
  (93.3%) and Appearance (92.1%) are highly formulaic. Obj-Obj (29.1%),
  Multi-step (41.5%) and Obj-Reg (43.5%) overlap heavily with other
  categories. Contrary to the expectation, Measurement reaches only 55.0%.
- **Surface wording predicts the official taxonomy.** Macro-F1 is 0.824 for
  TF-IDF, 0.791 for MiniLM, 0.658 for the 64 lexicon features and 0.509 for
  Empath (majority baseline 0.030). The lexicon misses the 0.15 margin by
  0.016, so operations and frames explain much, but not all, of the taxonomy.
  Empath is stronger than expected because generic concepts such as motion,
  direction, appearance and measurement already separate several categories.
- **Multi-step is mixed, but not the hardest supervised category.** Its
  TF-IDF F1 is 0.73; Obj-Obj is lowest at 0.67. This supports treating
  Multi-step as a composition of operations rather than a wholly separate
  language domain.
- **Unsupervised clusters follow the official category more than answer
  format.** K-means AMI is 0.427 with category, 0.350 with frame and 0.252
  with answer space. Thus answer format alone does not explain the wording.
- **Templates are common but do not leak answer letters.** At cosine ≥ 0.95,
  261 questions form 727 near-duplicate pairs; 98% of pairs share a category
  and 381 cross the explore/confirm split. There are 246 exact-stem pairs.
  Correct letters agree in 25% of pairs (chance 25%), and answer text agrees
  in 15%. A template-matched learner therefore cannot simply copy a label,
  but random splitting can still overstate how well it learns category
  wording. Step D reports the requested split and will additionally report a
  template-grouped sensitivity check.

**Implications.**
- **Ideas 1 and 4.** Because broad categories are linguistically predictable
  but internally mixed, extract operations, frames and premises first; do not
  route reasoning solely by the 11-way category. Near-duplicate templates
  should be grouped when evaluating learned extractors.
- **Idea 3.** The template pairs' chance-level letter agreement is reassuring:
  answer shuffling still matters for position bias, but duplicated wording is
  not itself an answer-letter shortcut.

### Step D: Answer-choice shortcuts and text-only learner (`shortcuts.py`, `04_shortcuts.py`)

**What it does.** Rules are selected only on the 500-item **explore** half;
all headline accuracies are measured on the 500-item **confirm** half.

- Reports chance, explore-majority letter, longest option, shortest option and
  greatest stem-option token overlap.
- Tests three transparent elimination candidates on explore: `"Sometimes"`
  options, abstention options, and an option whose format differs from the
  other three. A rule is retained only with at least 10 opportunities and a
  Holm-corrected one-sided binomial test showing that its options are correct
  less often than 25%. The resulting random-after-elimination policy is then
  frozen and evaluated on confirm.
- Fits a multinomial text learner on explore. Features are stem word
  unigrams/bigrams, letter-namespaced option word unigrams/bigrams, option
  length, stem-option overlap and answer-space tags. Regularization is chosen
  by five-fold cross-validation within explore; the model is refit on all
  explore items and evaluated once on confirm.
- Repeats the learned baseline with near-duplicate stems assigned to the same
  cross-validation fold. This sensitivity check prevents the 381
  cross-split template pairs found in Step C from making model selection look
  easier than it is.
- Gives percentile bootstrap 95% confidence intervals (10,000 resamples by
  question), one-sided random-label tests against 25%, and Holm-adjusted
  p-values across the non-chance policies. The random-label test reduces to
  the exact binomial setup for deterministic policies and remains valid when
  a policy splits probability across tied options. It also reports confirm
  accuracy by category and answer space.

The script writes `outputs/04_shortcut_summary.csv`, `outputs/04_shortcuts_by_category.csv`,
`outputs/04_shortcuts_by_answer_space.csv`, `outputs/04_elimination_rules.csv`,
`outputs/04_option_content_priors.csv`, `outputs/04_text_learner_coefficients.csv`, and
`outputs/04_shortcuts.png`. The summary CSVs and the figure are in `results/`. The token prior and coefficient dumps stay local.

**Expected.**
- Explore-majority-letter accuracy is close to chance on confirm (≤ 30%).
- Longest-option accuracy exceeds 25% overall and exceeds 40% in Cam-Cam
  (the full-data pilot was 30.9% and 50.5%, respectively).
- Shortest-option and greatest-overlap accuracy are at most 30% overall.
- The `"Sometimes"` elimination candidate is retained: the full-data pilot
  found 51 such options and none was correct. At least one of the other two
  elimination rules may fail the pre-stated selection test.
- Random after the selected eliminations exceeds 25% but remains below 35%.
- The text learner exceeds the best fixed heuristic, but remains below 45%
  confirm accuracy. A substantially higher result would indicate a serious
  language-only shortcut and motivate stronger controls for all four ideas.
- Template-grouped and ordinary explore cross-validation select either the
  same regularization or adjacent grid values; a large difference would mean
  duplicated templates materially affect model selection.

**Actual (2026-10-01).** The script completed on the frozen 500/500 split.
Five of seven printed expectations held.

- **Simple positional and length shortcuts are weak.** The explore-majority
  letter is A, but reaches only 24.2% on confirm. Longest-option accuracy is
  27.3% (95% CI 25.1-29.6%) and is not significant after correcting across
  policies (`p_Holm = .082`). Cam-Cam reaches 35.3%, well below the
  pre-stated 40% and the full-data pilot's 50.5%; the pilot estimate was
  therefore optimistic. Shortest option (22.6%) and greatest stem-option
  overlap (22.3%) are below chance.
- **The wording of an option is a real shortcut.** The explore-trained
  option-token prior reaches **32.2%** on confirm (95% CI 28.2-36.3%,
  `p_Holm = .0018`), using a Beta(1, 3) prior centered at the four-choice
  chance rate (log-odds log(1/3) for unseen tokens). Its strongest positive
  tokens are from abstention answers: `"determined"`, `"be"` and `"cannot"`.
  On explore, abstention options are correct 31/53 times (58.5%), so they
  must not be eliminated. This prior is strongest in Motion: Cam (48.6%) and
  Measurement (45.3%), and weakest in Multi-step (21.4%). (An earlier implementation
  defaulted unseen tokens to zero log-odds and reached 32.9%; that figure is
  kept for reference but the corrected 32.2% is the reported result.)
- **`"Sometimes"` is the only valid elimination.** It is correct 0/29 times
  on explore (`p_Holm < .001`). Abstention and format-outlier candidates are
  correct 58.5% and 49.2%, respectively, and are rejected. Random choice
  after removing `"Sometimes"` reaches 25.45%. Its confidence interval is
  narrow because this is the expected accuracy of a randomized policy, but
  the practical gain is only 0.45 percentage points.
- **The full text learner edges out the option-token prior.** It reaches 32.8%
  (95% CI 28.6-36.8%), 0.6 percentage points above the corrected 32.2% prior.
  Accuracy is 32.9% on the 392 confirm questions without a ≥.95-cosine template
  in explore, so near-duplicate templates do not explain the result. This is the
  most important surprise: most usable language-only signal is in answer
  wording, not sophisticated stem-option reasoning.
- Ordinary and template-grouped cross-validation select adjacent
  regularization values (`C=10` and `C=1`). Grouped CV is preferable and is
  used for the reported learner.

**Implications.**
- **Idea 3.** Answer shuffling removes letter-position bias but does not remove
  answer-content priors. Transformation consistency should therefore include
  cyclic option permutations *and* an options-only baseline. Report the
  `"Sometimes"` subset separately or remove that option consistently from all
  transformed variants.
- **Ideas 1, 2 and 4.** A model can obtain about 33% without images, so raw
  end-to-end accuracy overstates visual or graph reasoning. Improvements
  should be compared against the content-prior and blind-model baselines, and
  evidence analyses should focus on the residual cases they cannot answer.

**Reproduce.**

```bash
cd MMML2026/analyses/language
conda run -n 11777-project python -m pytest tests -q
conda run -n 11777-project python 04_shortcuts.py | tee outputs/04_run.log
```

### Step E: Blind 7B model and permutation controls (`blind_vlm.py`, `05_blind_vlm.py`)

**What it does.** Runs `Qwen/Qwen2.5-VL-7B-Instruct` with text only—no image
tokens or tensors—and scores the next-token logits for A, B, C and D. If the
VL checkpoint cannot load, it falls back to `Qwen/Qwen2.5-7B-Instruct`.

Each question has five forward passes:
- the original stem and options;
- options only, with the question and images explicitly hidden;
- three cyclic rotations of the option contents.

The unrotated original also serves as cyclic rotation zero, so the analysis
has all four rotations without a redundant sixth pass. A displayed prediction
is mapped back to the original option content before scoring. The script
reports:
- original and options-only accuracy;
- accuracy pooled over all four rotations;
- consensus accuracy after averaging mapped A-D logits;
- the fraction correct under all four rotations;
- content consistency (all four mapped predictions agree);
- displayed-letter distributions and per-category accuracy.

The run is resumable. Every batch is appended to
`05_blind_vlm_raw.jsonl`; rerunning skips completed
`(model, id, condition, shift)` keys. The first eight-question smoke test can
therefore remain in the same file. The full run is 5,000 text prompts, not
24,000 separately scored answer continuations.

The script writes `outputs/05_blind_vlm_raw.jsonl`, `outputs/05_blind_vlm_predictions.csv`,
`outputs/05_circular_questions.csv`, `outputs/05_blind_vlm_summary.csv`, and
`outputs/05_blind_vlm_by_category.csv`. The published summary tables are the full Colab run, copied to `results/tables/05_blind_vlm_summary.csv` and `results/tables/05_blind_vlm_by_category.csv`. Raw JSONL and per-question predictions stay local.

**Why it matters.**
- **Idea 3.** Cyclic rotations separate option-content knowledge from
  displayed-letter bias. Options-only scoring measures the content prior that
  reached 32.2% in Step D (corrected; earlier zero-default implementation: 32.9%).
- **Ideas 1, 2 and 4.** The blind score is the language-only floor. A proposed
  reasoning or geometry module should demonstrate gains beyond that floor,
  particularly on questions where the blind model is wrong or unstable.

**Expected.**
- No rendered prompt reaches the 2,048-token limit.
- Original blind accuracy is 25-35%. This brackets the paper's reported 22.7%
  blind GPT-4o result and Step D's 32.2% option-content prior.
- Options-only accuracy is within 5 percentage points of original accuracy.
  A small gap would show that most blind signal comes from answer wording.
- Circular-consensus accuracy is within 5 points of original accuracy, but
  all-four-correct accuracy is at least 5 points lower. Individual answers
  should be sensitive to displayed position even when aggregate accuracy is
  stable.
- Content consistency across all four rotations is below 80%.
- No displayed letter accounts for more than 45% of original predictions.
- Category accuracy spans at least 15 points. Based on Step D, Motion: Cam
  and Measurement may benefit most from option-content priors.

**Actual (2026-10-02).** All 7 expectations held. The Colab T4 run loaded
`Qwen/Qwen2.5-VL-7B-Instruct` in 4-bit, resumed the 40 smoke-test prompts, and
scored the remaining 4,960. Results are in
`outputs/mmsi_blind_vlm_results/`. Every question has a complete four-rotation
set. The longest rendered prompt is 265 tokens.

- **The blind model is only slightly above chance.** Original accuracy is
  26.2% (95% CI 23.5–28.9%); the confirm half is 25.8%. This sits between the
  paper's 22.7% blind GPT-4o result and Step D's 32.2% option-content prior.
  The 7B model does not extract as much answer-wording signal as the
  explore-trained token prior.
- **Hiding the question leaves accuracy in the same range.** Options-only accuracy is 27.6%
  overall and 25.0% on confirm, within 1.4 points of the original condition.
  The two scores are close under this protocol. That closeness does not show
  that the question contributes nothing: the comparison is not a test of
  equivalence, and both intervals include values near chance.
- **Displayed position dominates option content.** The model chooses D for
  34.0% of original prompts and A for only 8.4%. With the stem hidden, D rises
  to 55.9% and A falls to 2.9%. Circular consensus stays at 26.8%, but the
  same original option is chosen under all four rotations for only **28.1%** of
  questions (content consistency), and all four rotations are correct for only
  **9.5%**. Pooled rotation accuracy across all four shifts is **27.7%**.
  Aggregate accuracy therefore hides a large letter-position bias.
  (Rotation metrics are computed from the recorded `pred_original` column;
  mean-logit consensus is computed separately and is not affected by this convention.)
- **Category differences are real, but not the ones Step D suggested.**
  Original accuracy ranges from 17.6% on Motion: Cam to 36.8% on Motion: Obj,
  a span of 19.3 points. On confirm, Obj-Reg is 40.5% and Cam-Obj is 11.6%.
  Measurement is exactly 25% in the original order, but its circular consensus
  rises to 34.4% overall and 40.6% on confirm. Averaging rotations helps that
  category more than a single displayed order does.

**Implications.**
- **Idea 3.** Report cyclic consensus, not one displayed order. A single
  shuffle can move accuracy by much more than the 1-point difference between
  the original and consensus scores. Report the options-only score beside the
  original blind score. Their closeness is not evidence that the question
  contributes nothing.
- **Ideas 1, 2, and 4.** A visual or graph module has to beat about 26–28%,
  not 25% flat. The useful residual is the 90.5% of questions that are not
  answered correctly under every option rotation. Motion: Cam and Cam-Obj are
  the weakest blind categories, so gains there are less likely to be language
  shortcuts.

**Colab bundle.** From this directory, build the upload zip:

```bash
python colab/package_colab_bundle.py
```

That writes `colab/mmsi_blind_vlm_colab.zip` (code, text cache, and
`01_tags.csv`; no Mac scoring results). In Colab, set the runtime to a T4
GPU, upload `colab/MMSI_blind_vlm.ipynb`, and follow it. The notebook asks
for the zip, loads the model in 4-bit, and writes a fresh
`outputs/05_colab_raw.jsonl`.

**Other GPU machines.** A free 16 GB T4 requires `--quantization 4bit`; an
A100/L40 can use `--quantization none`. The directory must contain
`cache/mmsi_text.parquet` and `outputs/01_tags.csv`. Start a new raw file
rather than resuming `05_blind_vlm_raw.jsonl` from the Mac.

```bash
pip install -r requirements-blind-vlm.txt

# Eight-question smoke test. These results are retained for the full run.
python 05_blind_vlm.py --limit 8 --quantization auto

# Full resumable run.
python 05_blind_vlm.py --quantization auto --batch-size 8 \
  | tee outputs/05_run.log
```

If memory is insufficient, reduce `--batch-size` to 4 or 2. To rebuild tables
without loading the model:

```bash
python 05_blind_vlm.py --analyze-only | tee outputs/05_analysis.log
```

### Step F: Reasoning traces (`traces.py`, `06_traces.py`)

**What it does.** Describes the human reference trace in every question. Traces
are not used as features for any answer test. Three views are reported:

- **Complexity proxies:** trace length against difficulty, the released
  normalized duration against difficulty, and stem length against trace length.
  Duration is `mean_normed_duration_seconds`, which is already centered near
  zero; it is not a raw time in seconds.
- **Bridging nouns:** content nouns in the trace whose lemma is absent from
  the stem and options. Image words and direction words are excluded.
- **Step cues:** numbered steps, sequence words, conclusion words, numbered-view
  references, perspective phrases, comparisons, and motion verbs.

The cache already establishes the marginal facts: all 1,000 traces are
non-empty, difficulty is 334/333/333, and trace length has median 45 words
and maximum 219 words. The expectations below concern relations that have not
yet been measured.

The script writes `outputs/06_trace_questions.csv`, `outputs/06_trace_summary_by_category.csv`,
`outputs/06_trace_correlations.csv`, `outputs/06_bridging_heads.csv`, and `outputs/06_traces.png`.
The summary tables and figure are in `results/`. Per-question trace rows stay local.

**Why it matters.**
- **Idea 1.** Conclusion, comparison, perspective, and motion cues are
  candidate claim types in a dependency graph.
- **Idea 4.** A noun that occurs only in the reference trace is a landmark a
  question-only graph would prune. A high bridging rate says pruning cannot
  keep only entities named in the question.

**Expected.**
- Trace length rises with difficulty (Spearman at least 0.20).
- Normalized duration rises with difficulty (Spearman at least 0.15).
- Stem length and trace length are weakly related (Spearman at most 0.30).
- At least 40% of traces introduce one or more nouns absent from the question.
- The Multi-step bridging rate is at least as high as the overall rate.
- Explore and confirm bridging rates differ by at most 8 points.
- At least 25% of traces contain a conclusion cue, and at least 40% name a
  numbered view such as "Figure 2".
- Numbered step lists are at least 10 points more common in Multi-step than
  in Cam-Obj.

**Actual (2026-10-02).** Five of nine expectations held.

- **Difficulty tracks time more than trace length.** Normalized duration has
  Spearman 0.54 with the easy/medium/hard order. Mean duration is −0.63,
  0.28, and 0.35. Median trace length is 39, 52, and 47 words, so medium
  traces are longer than hard ones and the length correlation is only 0.12.
- **Longer questions have moderately longer traces.** Stem length and trace
  length have Spearman 0.35, just above the 0.30 cutoff. Trace length and
  duration have Spearman 0.28.
- **Most reference solutions name landmarks the question does not.** At least
  one bridging noun appears in 84.2% of traces. The two halves agree: 83.8%
  and 84.6%. Removing the discourse nouns "reference" and "analysis" leaves
  the rate at 83.4%, so the result is not an annotation-boilerplate artifact.
  Frequent real landmarks are table (79 questions), wall (72), chair (72),
  door (67), corner (54), camera (51), and sofa (47).
- **Multi-step is not the category that adds the most entities.** Its bridging
  rate is 75.8%, below the overall rate. Camera-anchored position questions
  are higher, at 88–99%. Appearance is the exception: only 40.9% of its traces
  introduce a new noun, and its traces are short (mean 29 words).
- **Step format is shared across categories.** A conclusion cue appears in
  52.2% of traces, and 48.3% name a numbered view. Numbered lists occur in
  49.5% of Multi-step traces and 40.7% of Cam-Obj traces, a gap of 8.8 points.
  Obj-Obj is actually the most list-like category, at 79.8%.

**Implications.**
- **Idea 1.** The useful claim types are conclusion, comparison, cross-view
  reference, and motion. Numbered lists are an annotation style, not evidence
  that only Multi-step questions have multiple steps.
- **Idea 4.** A graph built only from the question misses a landmark in about
  five out of six reference solutions. Pruning should retain visually salient
  objects and structural surfaces even when the question does not name them.
  Difficulty is better read from the released duration label than from trace
  length.

**Run.**

```bash
cd MMML2026/analyses/language
conda run -n 11777-project python -m pytest tests/test_traces.py -q
conda run -n 11777-project python 06_traces.py | tee outputs/06_run.log
```

### Step G: Addition profiles (`additions.py`, `07_additions.py`)

**What it does.** Diffs each human trace against its stem and options, then
summarizes the added claims inside the official categories and inside MiniLM
wording groups. The four added-claim types are an intermediate spatial
relation, a viewpoint change beyond the Step A anchor, a comparison whose
word is new, and a reasoning step (a cross-view identity or a route
consequence). Bridging nouns are copied from `06_trace_questions.csv`. Step F
presence cues stay unchanged. This is a local CPU run: spaCy plus the saved
MiniLM matrix. No Colab notebook and no new blind-model run.

The script writes `outputs/07_addition_questions.csv`, `outputs/07_addition_by_category.csv`,
`outputs/07_addition_groups.csv`, `outputs/07_addition_neighbors.csv`, and `outputs/07_additions.png`.
`results/` keeps the by-category table, the neighbor table, and the figure. Per-question rows and template-group listings stay local.

**Unit tests.**
- Expected: `tests/test_additions.py` passes. A failure is a detector bug.
  "Behind the chair" counts only when the question does not already attach
  "behind" to "chair". A bare option "Directly behind" does not block it.
  "Behind you" is not a landmark. "Between A and B" keeps both nouns.
  Repeating "closer" is not an addition. "Between" is not a comparison.
  "Figure 2" alone is not a viewpoint change. Comparing with another photo,
  or facing a new object, is. Repeating both cameras the question already
  names is not. A numbered "so / therefore" list is not a reasoning step.
  "Not the same" with a content noun is. "Walking area" is not a route.
- **Actual (2026-10-03).** 17 passed in 3.54s.

**Expected (measurement, before the run).**
- The copied bridging rate matches Step F: 842/1,000 questions.
- Rebuilding cosine ≥ 0.95 components reproduces Step C: 727 pairs, 64
  components, 4 of them mixed-category.
- Each of the four addition rates differs by at most 8 points between the
  explore and confirm halves.
- On the confirm half, cross-category 5-nearest-neighbor agreement beats a
  1,000-draw permutation of the addition labels (one-sided p < 0.05) for at
  least one of the four types. This is one combined check. If only one type
  barely passes, treat it as suggestive. If none passes, that is a finding:
  similar wording does not carry the addition across official categories.
- The rates themselves are what this run measures. No rate floor is
  pre-registered.

**Actual (2026-10-03).** Six of seven expectations held. The cross-category neighbor check is the surprise.

- **About half of traces add one of these claims.** At least one flag is present in 488/1,000 traces: spatial relation 23.0%, comparison 17.0%, reasoning step 14.3%, viewpoint change 13.9%. 328 traces have exactly one type and 160 have two or more. Explore and confirm differ by at most 1.4 points. These rates are lower than Step F's presence cues because a word already in the stem or options is no longer counted.
- **The official categories tilt which claim is modal, and the tilt is mild.** Viewpoint is modal in Cam-Obj (37.2%) and is at 0% in Measurement. Comparison is modal in Reg-Reg (29.6%), Measurement (32.8%), and Motion: Cam (33.8%). Reasoning is modal in Obj-Reg (35.3%). Spatial relations are modal in Cam-Cam, Cam-Reg, Obj-Obj, Motion: Obj, and Multi-step, at 13–33%. Appearance stays nearly empty (highest rate 6.1%). No modal rate exceeds 37%.
- **A shared template shares an absence more often than a specific claim.** Of the 64 cosine ≥ 0.95 components (261 questions), 25 contain no added claim (67 questions), 8 tie between types, and 21 have a single type in at least half their members (72 questions). The four mixed-category components are small. The size-10 Cam-Cam / Multi-step component is 40% spatial. One size-2 Cam-Obj / Reg-Reg pair is reasoning in both members. One size-2 Motion: Obj / Multi-step pair has no addition.
- **Similar wording across categories stays at the permutation baseline.** On confirm, cross-category 5-NN agreement is spatial 0.655 (null mean 0.646, p = .35), comparison 0.716 (null 0.718, p = .56), reasoning 0.736 (null 0.755, p = .86), and viewpoint 0.726 (null 0.762, p = .98). The bars in `results/figures/07_additions.png` sit near 0.7–0.9 because a shared absence counts as agreement; the null means are 0.65–0.76, and the cross-category bars land on them. Viewpoint is the low side of that null. Same-category agreement is higher on every flag (confirm: spatial 0.709, viewpoint 0.810, comparison 0.738, reasoning 0.875). That gap matches the category tilt above. It was not a separate pre-registered test.
- **An added spatial relation often uses a noun the question never names.** 230 traces add a spatial relation, covering 298 landmarks. 181 of those landmarks (60.7%) are in the copied Step F bridging list, and 153 of the 230 traces have at least one such landmark. The other 117 landmarks are nouns the question already names, tied to a relation it does not state. Motion: Cam's spatial landmarks are all bridging nouns. Obj-Reg's share is 0.29, in line with reasoning being its modal addition.

**Implications.**
- **Idea 1.** The added claim types are a spatial relation, a viewpoint change, a comparison, and an identity or route step. Category shifts which one is most common. A near-duplicate stem is a weak guide to which claim the reference trace adds, and we found no above-null agreement for neighbors in another official category under this test. A claim graph can use the category as a mild prior and still has to read the trace. 512 traces contain none of these four claims.
- **Idea 4.** A question-only graph misses a spatial landmark relation in 23% of reference solutions. In 181/298 of those relations the landmark is a noun Step F already found to be absent from the question. The remaining relations reuse a named noun in a configuration the question does not state. Both are dropped by a graph that keeps only the relations written in the question.

**Run.** From `MMML2026/analyses/language`:

```bash
conda run -n 11777-project python 07_additions.py | tee outputs/07_run.log
```

### Step H: Human audit sample (`08_make_audit.py`)

**What it does.** Draws 44 confirm-half questions, four from each official
category, for a wording audit. Seed 1. Within a category the draw skips a
stem that is a cosine ≥ 0.95 near-duplicate of a stem already chosen.
`audit/packet_questions.md` is labeled first. `audit/packet_traces.md` is
opened second. Automatic tags stay in `audit/key.csv` and
`audit/key_mentions.csv`, which the labeler leaves closed. The response
sheet is `audit/response_sheet.csv`. The label definitions and the patterns
in this draw are in `audit/GUIDE.md`.

**Expected.**
- 44 items, four from each category, every item on the confirm half.
- No two items share a cosine ≥ 0.95 template.

**Actual (2026-10-03).** Both expectations held. The 44 ids are in
`audit/manifest.json`. Template collisions: none.

**Agreement with the automatic tags (`09_audit_agreement.py`).** The filled
sheet is compared with `audit/key.csv` and `audit/key_mentions.csv`. The
sheet was labeled from the guide with model assistance, then revised against
that same guide. These rates say how closely the automatic tags match the
model-assisted sheet. They are **not** an agreement between two independent
raters: the guide contains category-specific hints and expected patterns, the
sheet was filled and corrected with model help, and the agreement run itself
compares the tagger against that assisted sheet. No agreement floor was
pre-registered. A small independent check by a teammate not involved in
guide construction would be needed before treating these rates as validated
accuracy estimates.

**Actual (2026-10-04).**

- **Closed question labels.** Frame 41/44, anchor 39/44, reorder 41/44, mirror 35/44.
  The three frame misses are object-front questions (18, 20, 30) that the tagger files as camera, object-unspecified, or viewer-implicit. Anchor misses are image numbers the tagger attaches where the sheet treats the number as locating an object or a rotation endpoint rather than the camera called "you" (18, 19, 37, 43); item 10 is the reverse, with the sheet anchoring the agent at Figure 2. Seven of the nine mirror misses are east/west words that sit only in the options: the sheet marks `swap_options`, and the tagger marks `rewrite_stem`. Item 15 is `unsafe` in the tagger because "clock" matches a text pattern, and item 25 because "writing" does. Reorder misses are item 28 ("respectively") and items 38 and 40, which the sheet treats as order-sensitive and the tagger leaves at `label_dependent` or `safe`.
- **Added-claim flags.** Viewpoint 40/44, comparison 38/44, reasoning 37/44, spatial 29/44. Spatial is the weak flag. The sheet marks intermediate relations the extractor does not build, including "left of the exit," "opposite," "connected to," and "west of the mural." The tagger marks a few pairs the sheet rejected, including a relation to the camera or the viewpoint itself. The other three flags miss new facing directions, ratios, "approaching," "expanded," and explicit same-object or overlap sentences. The tagger also counts "lower" inside "lower left edge" as a comparison and "cannot see" as a route.
- **Mentions and bridge nouns.** Of 98 mention heads on the sheet, 87 occur among the 104 automatic mention heads, and 87 of those automatic heads are on the sheet. Several remaining gaps are the same noun in another form (`chairs`/`chair`, `plants`/`plant`) or a different head inside a compound ("camera coordinate system" headed `system`). Of 76 sheet lemmas marked landmark or synonym, 68 appear in the automatic bridging list. Of 107 automatic bridging lemmas, 68 are on the sheet. The extra automatic lemmas are mostly parser nouns the sheet did not keep.

The script writes `outputs/09_audit_agreement.csv` and `outputs/09_audit_mention_bridge.csv`. Both are copied to `results/tables/`.

**Run.** From `MMML2026/analyses/language`:

```bash
conda run -n 11777-project python 08_make_audit.py
conda run -n 11777-project python 09_audit_agreement.py
```
