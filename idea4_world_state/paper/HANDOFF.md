# Paper-drafting handoff: world-state evidence experiment

## Purpose and current status

Zhengfei Li is drafting a standalone contribution to a four-person 11-777
midterm report. The experiment and quantitative analysis are complete. The
20-item audit has been reviewed by the user, and its finalized CSVs have been
re-imported. A paragraph-level outline, artifact plan, and LaTeX results table
are prepared. The full LaTeX section and qualitative figure are not yet written.

Proposed title: **Structured World-State Evidence for Multi-Image Spatial
Reasoning**. Target approximately 2.25 pages, including one table and one figure.
Keep this standalone; the user did not authorize assigning it to the team's
"Research Idea 1" placeholder.

The strongest narrative is that supported local descriptions can still omit
task-relevant objects, correspondences and relationships. The current baseline
tests sparse evidence presentation; it does not establish a benefit from rich
spatial-relation graphs.

## Experiment and verified results

- Dataset: MMSI-Bench, 1,000 four-option questions with 2–10 images.
- Model: Qwen3-VL-8B-Instruct, bfloat16, vLLM 0.32.0, checkpoint revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; PSC L40S inference, local runner.
- Development: 22 questions from the existing explore partition. Paired
  evaluation: a frozen, approximately category-balanced sample of 200 two-image
  questions from confirm, across 11 categories. Exact-image groups are separated
  from development. The underlying explore/confirm partition is 500/500.
- Raw receives the images, question and options. Extraction receives images and
  the question stem, without options or gold labels. Flat and Graph serialize the
  same extracted facts differently. Both answer stages retain the original images.
- Raw on all 1,000: **291/1,000 = 29.1%**; eight truncated answers count as wrong.

| Frozen paired sample | Correct / 200 | Accuracy |
| --- | ---: | ---: |
| Raw | 57 | 28.5% |
| Flat | 53 | 26.5% |
| Graph | 61 | 30.5% |

Primary Graph-minus-Flat difference: **+4.0 percentage points**, 95% CI
**−0.0125 to +8.5 points** (report rounded as −0.01 to +8.50). Graph corrects
15 Flat errors and introduces seven errors. Graph-minus-Raw is +2.0 points
(CI −5.0 to +9.0); Flat-minus-Raw is −2.0 (CI −8.5 to +4.5). All intervals
include zero. These are 2,000 paired bootstrap resamples within category,
seed 0. Do not claim a significant improvement or equivalence.

Extraction succeeds on **183/200** items. Accepted inventories contain 511
entities and 622 observations (means 2.79 and 3.40), **zero relation/attribute
claims**, and zero identity links. Sixty-five have cross-view annotations;
nine have known query frames. Seventeen extraction failures count as incorrect
for both evidence conditions; no answer fallback was applied.

Importantly, **eight rejected raw responses attempted claims**. Seven fail
first on missing claim fields; one fails on an invalid identity link. Across
all 17 failures, the first errors are seven missing-claim-field errors, five
overlong unknown strings, one invalid identity-link list, two camera-count
errors and two cross-view errors. Say "no validated inventory contained
claims," not "the model never generated claims." These are first failures;
other defects and unsupported claims may coexist in a rejected response.

Secondary, post-hoc diagnostic on the same 183 successful-extraction items:
Raw 52/183, Flat 53/183, Graph 61/183. Raw gets 5/17 correct on extraction
failures. Keep the 200-item denominator primary; do not present the conditional
subset as a causal estimate of fixing extraction.

Strict parsing accepts single or double backticks. Paired strict accuracies are
3.5%, 0.5%, 1.0%, mainly reflecting output-format noncompliance: most outputs are
unambiguous bare letters. The predefined lenient score remains primary.
Evidence methods each use roughly **7× Raw's summed request latency**, including
extraction. This is not billed GPU time; extraction is shared between the two
evidence conditions and must not be double-counted in physical experiment cost.

## Development history worth summarizing briefly

The initial prompt's only complete JSON example was empty; 21/22 accepted
development inventories repeated an empty statement. Revised instructions elicited
objects and observations. Reusing one entity ID across views with null cross-view
metadata is normalized to `same`, uncertainty `medium`, with the uncertainty
explicitly recorded as `parser_default`, not model-estimated confidence.

The active prompt is v3. A final v4 development check on six items added a
supported within-image relation example but still produced zero claims. Prompt
iteration stopped, and the sparse v3 baseline was frozen. Detailed debugging
history belongs in the appendix rather than dominating the main section.

## Human-reviewed audit and main qualitative example

The audit samples 20 items from the 183 accepted inventories, using the existing
category-balanced, correctness-independent selection. It contains 238 statement
rows and 20 question-level rows. Codex proposed the first-pass labels. Zhengfei
explicitly confirmed reviewing them, endorsing or revising their wording.
Some revisions occurred after outcome exposure. Describe this as **AI-assisted,
human-reviewed qualitative annotation**, not independent human annotation or a
fully blinded final audit. CSV reviewer fields still record the original Codex
attribution; document the subsequent human review separately.

Current observation judgments: **69/77 supported**, four contradicted, two
unsupported, two cannot_judge. All **20/20** inventories are judged to omit
critical evidence. Camera aliases are consistently treated as structural
bookkeeping and excluded from substantive support summaries. Bare unknown
metadata is treated as abstention, not an incorrect spatial assertion.

Do not interpret 20/20 as a dataset-level recall estimate, or as 20 cases of
missing objects specifically. Omissions include entities, measurements,
correspondences, spatial relations and frame grounding. Absence of an explicitly
encoded relation alone does not prove the image-aware answer stage cannot infer it.

**Main figure: item 919.** Its two photographs show a bathroom. The inventory
contains door, bathtub and shower curtain but omits a clearly visible toilet.
It also leaves the entry-direction relationship unresolved. The user highlighted
this as an important example of missing items despite accurate extracted claims.
All three conditions answer this item incorrectly. Use it to illustrate coverage
failure, not graph-induced harm or proof that adding the toilet fixes the answer.
No need to identify the omitted object as the gold answer without further checking.

## Paper structure and relation to teammates

Use the assignment's headings:
1. **Initial Implementation:** motivation; shared extraction and Flat/Graph
   control; representation and normalization; actual model/split/metric settings.
2. **Results and Discussion:** paired table; uncertain positive point estimate;
   schema failures; accuracy-versus-coverage audit finding; item 919; cost/limits.
3. **Updated Research Idea:** staged grounded relational graph—image-local
   evidence, explicit cross-view matching, then supported relations with named
   frames. Compare against the frozen sparse baseline and preserve Flat/Graph
   controls. VGGT integration is future collaboration, not completed work.

Teammates' draft: Idea 2 uses Qwen2.5-VL-7B with VGGT camera geometry and a judge;
Idea 3 uses Qwen2.5-VL-7B and InternVL3-8B with transformations, consistency and
aggregation. Position this section around evidence construction/representation.
Do not rank their accuracy numbers against ours: models, prompts and populations
differ. Flag shared-methodology issues rather than silently rewriting their work:
the draft's purportedly disjoint 100+300+700 split totals 1,100; its universal
Qwen2.5 description and 1,000-bootstrap convention do not describe this section.

Assignment essentials: four-person team, 10 main-text pages plus references;
optional appendix up to five pages is not graded; AI-use/contribution disclosure
has its own excluded page. An initial baseline and substantive failure analysis
are sufficient; an accuracy gain is not required. Disclose Codex/Grok assistance,
Qwen inference and AI-assisted audit/drafting honestly.

## Resources: exact locations

**Repository root**
`/Users/zhengfeili/2026_Fall/11777/11777Project/MMML2026`

Paths below are relative to that root unless absolute:

### Start here for drafting
- `idea4_world_state/paper/OUTLINE.md`: paragraph-level section outline, equations,
  proposed wording and integration caveats.
- `idea4_world_state/paper/ARTIFACT_PLAN.md`: main/appendix priorities and item 919
  figure design/caption. The figure itself is not yet assembled.
- `idea4_world_state/paper/tables/main_results.tex`: prepared booktabs table;
  not yet compiled in the team's ICML document.
- `runs/idea4_world_state/result_review/RESULTS_REVIEW.md`: independently checked
  interpretation, failure details and limitations.
- `runs/idea4_world_state/result_review/verify_results.py` and `verification.json`:
  reproducible checks against saved records and dataset gold labels.
- `runs/idea4_world_state/result_review/rejected_claims.json`: eight rejected
  responses containing attempted claims.

### Numerical and protocol sources
- `runs/idea4_world_state/raw_all/analysis/`: report, summary, category scores,
  per-item outcomes and costs for all 1,000 questions.
- `runs/idea4_world_state/eval200/analysis/`: authoritative paired scores,
  confidence intervals, failures, coverage, costs and item outcomes.
- `idea4_world_state/artifacts/protocol_freeze.json`: frozen protocol.
- `idea4_world_state/artifacts/prepare_v1/selection.json`: sample IDs and provenance.
- `runs/idea4_world_state/{raw_all,eval200}/run_config.json`: run settings and hashes.
- `idea4_world_state/IMPLEMENTATION_PLAN.md`: detailed method and audit conventions.
- `idea4_world_state/prompts/extract_v3.txt`: active extraction prompt.
- `runs/idea4_world_state/debug_extract_v4/summary.json`: final six-item prompt check.

### Finalized audit and figure sources
- `runs/idea4_world_state/eval200/audit/atom_labels.csv`: statement judgments.
- `runs/idea4_world_state/eval200/audit/item_labels.csv`: omissions and item notes.
- `runs/idea4_world_state/eval200/audit/imported.json`: matches finalized label
  hashes as checked in this conversation; check again if the CSVs are edited.
- `runs/idea4_world_state/eval200/audit/index.html`: images, stems and inventory atoms.
- `runs/idea4_world_state/eval200/audit/images/919_1.jpg` and `919_2.jpg`: main figure.
- `runs/idea4_world_state/eval200/inventories/919.json`: actual extracted inventory.
- `runs/idea4_world_state/eval200/repairs/`: per-entity normalization provenance.
- `runs/idea4_world_state/eval200/audit_outcomes/`: outcome-aware material exists,
  but summaries may predate final label edits. Recompute from current CSVs before
  quoting; do not assume importing labels refreshed all downstream summaries.

### Team draft and assignment
- Team's pasted LaTeX draft:
  `/Users/zhengfeili/.codex/attachments/016ddff6-e396-4916-95ae-d50b671010ba/Pasted text.txt`
- Assignment requirements:
  `/Users/zhengfeili/2026_Fall/11777/11777Project/11777HWInstructions/MidtermAssignment-Instructions-Report.pdf`
- Verified literature starting points: Qwen3-VL report
  https://arxiv.org/abs/2511.21631 ; scene graph generation
  https://openaccess.thecvf.com/content_cvpr_2017/html/Xu_Scene_Graph_Generation_CVPR_2017_paper.html ;
  Neural State Machine
  https://papers.neurips.cc/paper_files/paper/2019/hash/c20a7ce2a627ba838cfbff082db35197-Abstract.html .

## Minimal continuation

Assemble the item 919 figure from the original photographs with clearly labeled
human annotations, then draft the standalone LaTeX section from the outline and
prepared table. Keep frozen results unchanged. No new model experiments are
needed for the first draft. Use the project `11777-project` Python environment
for any calculations; preserve unrelated uncommitted work.
