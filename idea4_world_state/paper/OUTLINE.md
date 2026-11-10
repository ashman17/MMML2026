# Structured World-State Evidence for Multi-Image Spatial Reasoning

Standalone section outline; no research-idea number assigned. Target approximately
2.25 pages in the team's ICML-format main report, including one results table
and one qualitative figure. No additional model runs are needed for this draft.

## Central argument

We compare direct VLM answering with two presentations of the same extracted
visual evidence. Graph formatting has a positive but uncertain effect relative
to a flat list. The more actionable finding is a construction bottleneck:
accepted inventories mostly contain supported local descriptions but omit
task-relevant objects, attributes, correspondences, or relations. The next
method should separate and ground these stages rather than rely on one-shot
generation of a complex world-state schema.

Use "sparse entity/observation graph" for the implemented baseline. Do not imply
that it contains successfully extracted spatial-relation edges.

## 1. Initial Implementation (about 0.8 page)

### Paragraph 1: Motivation and controlled question

Opening: "Multi-image spatial reasoning requires identifying relevant entities
and relating their observations across views. We investigate whether exposing
an explicit intermediate state, and organizing that state as a graph, helps a
fixed VLM answer spatial questions."

Define the primary comparison as Graph versus Flat with identical evidence.
Raw provides a direct-answer reference. Position this as evidence construction
and representation, complementing the teammates' external geometry and
transformation-consistency investigations.

### Paragraph 2: Pipeline and notation

Use one compact equation instead of a separate pipeline figure:

\[
z=V(E_\theta(I_1,I_2,q)),\qquad
\hat y_s=\operatorname{Parse}(D_\theta(I_1,I_2,q,\mathcal O,S_s(z))),
\quad s\in\{\mathrm{Flat},\mathrm{Graph}\}.
\]

Explain extraction, validation/normalization, and the deterministic serializers.
Extraction sees the question stem and images, but not options or gold labels.
Both evidence-answer conditions retain the original images and options.
Flat and Graph share one inventory; graph organization groups observations by
entity. Invalid extraction counts as wrong in both evidence conditions.

### Paragraph 3: Representation and development

Describe entities, per-image observations, cross-view identities, frames,
claims and unknowns. State that the accepted evaluation inventories contain no
relation/attribute claims, so the implemented graph is sparse.

Summarize development in two or three sentences: the initial empty example
encouraged empty output; revised prompts elicited objects; shared-ID
normalization supplied explicit cross-view metadata with parser-default
uncertainty. A six-item v4 development check still produced no claims, so v3
was frozen. Treat example imitation as an interpretation, not proven causality.
Keep detailed prompt history and full schema outside the main section.

### Paragraph 4: Setup and evaluation

Specify Qwen3-VL-8B-Instruct, greedy generation, 768-pixel maximum image side,
and the frozen 22-item development / 200-item evaluation protocol. The paired
sample comprises two-image questions across 11 categories, sampled from the
confirm partition, with exact-image groups separated from development.
Raw additionally covers all 1,000 questions with their original image counts.

Use full-denominator accuracy and the predefined primary Graph-minus-Flat
comparison with 2,000 within-category paired bootstrap resamples. Explain that
the approximately category-balanced 200-item sample is not representative of
all 1,000 questions. Full checkpoint/version details belong in the appendix.

## 2. Results and Discussion (about 1 page)

### Paragraph 1: Quantitative result — Table 1

Insert `tables/main_results.tex`. State Raw 57/200 (28.5%), Flat 53/200 (26.5%)
and Graph 61/200 (30.5%). Graph-minus-Flat is +4.0 percentage points, 95% CI
[-0.01, +8.50], with 15 corrections and seven introduced errors. All paired
intervals include zero. Mention full-dataset Raw 291/1,000 (29.1%) separately;
do not place it in the paired table as another directly comparable row.

### Paragraph 2: Construction and schema failures

Report accepted inventories for 183/200 items (91.5%), averaging 2.79 entities
and 3.40 observations. No accepted inventory contains a relation or attribute
claim. Eight rejected responses attempted claims; seven fail first on missing
claim fields and one on an invalid identity link. Zero accepted claims therefore
does not mean that no raw response attempted relational or attribute evidence.

State that 17 extraction failures remain in the primary denominator. If space
permits, add the diagnostic: on the same 183 accepted-inventory items, Raw,
Flat and Graph score 52, 53 and 61 correct. Mark this as post hoc and conditional,
not a replacement result or a causal estimate of fixing extraction.

### Paragraph 3: Supported observations can still omit critical evidence

Describe the 20-item category-balanced audit of accepted inventories as
AI-assisted annotation, subsequently reviewed/revised by Zhengfei. The packet
initially hid options, gold answers and predictions, but revisions after outcome
exposure mean the final audit is not fully blinded or independent human labeling.

Use the finalized CSVs as the authority: 69/77 observation statements are labeled
supported; four contradicted, two unsupported, two cannot_judge. All 20 inspected
inventories are labeled missing critical evidence. Camera aliases are excluded
from substantive support summaries. Report these as descriptive reviewer
judgments, not scene-graph recall or population estimates.

Explain two different coverage failures: missing entities and missing properties/
relations/correspondences. Do not equate absence of an explicit reasoning step
with proof that the image-aware answer model cannot infer it.

### Paragraph 4: Concrete omission — Figure 1

Use item 919. Show both original bathroom images and the extracted entity list:
door, bathtub, shower curtain. Highlight the visible toilet as absent from the
inventory and note that the entry-direction relationship is unresolved.
All three conditions answered incorrectly on this item. It illustrates an
omission accompanying a failure; it does not show graph-induced harm or prove
that adding the toilet would fix the answer.

### Paragraph 5: Costs and limitations

Including extraction, Flat and Graph each use about seven times Raw's summed
request latency on the paired sample. This is not allocation cost or job wall
time. Discuss the small sample, one model/checkpoint, sparse accepted evidence,
and the descriptive, partly outcome-aware audit. Strict parsing is a formatting
diagnostic; it accepts single or double backticks and is not a second estimate
of reasoning accuracy. Detailed parsing/category tables go in the appendix.

## 3. Updated Research Idea (about 0.45 page)

Propose a staged grounded relational graph:

\[
O_i=E_{\mathrm{local}}(I_i,q),\quad
M=\operatorname{Match}(O_1,O_2),\quad
G=\operatorname{Fuse}(O_1,O_2,M).
\]

1. Extract question-relevant local entities, attributes and relations with
   explicit visual support, retaining candidate entities rather than prematurely
   dropping them when cross-view geometry is uncertain.
2. Establish cross-view correspondences explicitly, distinguishing uncertain
   links from confirmed merges and recording the source of uncertainty.
3. Construct supported relations in named reference frames, validating each
   stage separately to distinguish observation, matching and schema failures.

Evaluate the new method against this frozen sparse baseline. Retain Raw/Flat/
Graph controls to separate evidence improvements from presentation changes.
Measure question-relevant coverage and grounded relation quality as well as
end-to-end accuracy. A controlled, manually supplemented evidence subset can
test whether omissions limit performance; label it as an oracle diagnostic.

VGGT-derived camera relations are future collaboration with the geometry idea,
not part of the completed method. Transformation consistency is a complementary
diagnostic from Idea 3, not proof that a graph or answer is correct.

## Team integration and drafting rules

- Leave section numbering unset. Do not silently claim the Idea 1 placeholder.
- Flag the team's 100 + 300 + 700 "disjoint" split description for correction.
- Make model, split, prompt and bootstrap differences explicit per idea; do not
  merge Qwen2.5 and Qwen3 baseline scores or rank methods across different samples.
- Add related work on structured scene representations/graph-based reasoning
  and cite Qwen3-VL for this model. Avoid an unsupported novelty claim.
- Disclose Codex/Grok assistance and Qwen inference; record Zhengfei's design,
  execution, artifact review, audit verification and final writing decisions.
- The CSV reviewer field still says Codex. Preserve that initial attribution;
  document subsequent human review separately rather than implying independent
  human annotation or two independent raters.
- The completed audit supports drafting now. No additional model inference,
  prompt tuning, held-out repair or rescoring is part of this writing task.
