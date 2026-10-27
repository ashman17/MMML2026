# Paper artifacts

Use one results table and one qualitative figure in the main section. These
serve different purposes: paired performance and a concrete coverage failure.
Avoid spending space on an accuracy bar chart duplicating the table.

## Main text

### T1. Frozen paired results — prepared

File: `tables/main_results.tex`. A one-column, two-panel booktabs table:
condition scores followed by paired differences and confidence intervals.
The caption names the 200-item two-image population and failure policy.
Raw's all-1,000 result belongs in nearby prose, not this paired comparison.
The table uses the team's existing booktabs package. LaTeX source is prepared;
final layout must be checked in the team's compiled ICML document.

Sources: `runs/idea4_world_state/eval200/analysis/summary.json` and independently
verified calculations in `runs/idea4_world_state/result_review/verification.json`.

### F1. Accurate local descriptions with an omitted entity — to prepare

Use original images from
`runs/idea4_world_state/eval200/audit/images/919_1.jpg` and `919_2.jpg`.
Show both views at sufficient resolution, retaining the door, bathtub, curtain
and toilet. Add a simple outline around the toilet, visibly identified as a
human annotation; do not alter scene contents or generate replacement imagery.

Include a compact inset:

    Extracted entities: door; bathtub; shower curtain
    Omitted visible entity: toilet
    Unresolved relation: entry direction / fixture layout

Use labeled outlines rather than relying on color alone. Avoid adding an
unsupported 3D arrow or asserting that a 2D image direction is the entry axis.
Export PDF for LaTeX and PNG for preview; check readability at final column size.
Use a short float that can span two columns if the photographs are illegible
at one-column width, subject to the team's total ten-page budget.

Suggested caption:

> Evidence omission in a bathroom question (item 919). The inventory names the
> door, bathtub and shower curtain, but omits the clearly visible toilet
> (annotated), while leaving the question's entry-direction frame unresolved.
> Raw, Flat and Graph all answer this item incorrectly. This case illustrates
> incomplete evidence coverage, without establishing that the omission caused
> the error or that supplying the missing entity would correct it.

The annotation summarizes a human-reviewed omission; it is not a generated
bounding box or an additional model output. The current inventory and audit note
are the source for the omission. The frozen per-item records verify that all
three answers are wrong.

## Appendix or space-dependent additions

### T2. Audit and extraction diagnostic table

Keep denominators separate:

- Evaluation inventory acceptance: 183/200.
- Accepted inventories containing claims: 0/183.
- Rejected raw outputs containing attempted claims: 8/17.
- Supported observation statements in the audit: 69/77.
- Audited inventories judged to omit critical evidence: 20/20.

Include an explicit note that the audit is AI-assisted and human-reviewed,
with some post-outcome revisions. These rows are not independent performance
measures, and 20/20 is not a dataset-level object-recall estimate. If T2 does
not fit in the main text, retain its main findings in prose there.

### T3. Failure taxonomy and resource accounting

Use the existing 17 first-validation-error records and cost CSV. Report one
first error per failed item; other errors may also exist. Separate per-method
extraction-inclusive latency from unique physical calls. Do not sum Flat and
Graph method costs as if shared extraction was performed twice.

### F2. Additional qualitative cases

Optional: one Graph-over-Flat correction and one introduced error, selected
after inspecting the images. These are outcome-selected illustrations, not a
random sample or proof of causal mechanisms. Do not delay the main draft for
these if the mechanism remains ambiguous. Item 919 is the committed main case.

### A1. Reproducibility details

Checkpoint revision, vLLM/dtype, input resizing, token budgets, frozen sample,
prompts, parser-default policy, and selected inventory examples. Keep main
design decisions and main failure findings in graded main text.

## Deliberately omitted from the main section

- A separate pipeline diagram: the method equation and prose are sufficient.
- A zero-edge network visualization: it could suggest a richer representation
  than the implemented baseline and adds little beyond the inventory inset.
- A strict-versus-lenient accuracy bar chart: format compliance is secondary.
- Full category tables: many cells are small; reserve them for the appendix.

## Checks before insertion

1. Reconcile all table values with frozen reports and imported audit CSV hashes.
2. Keep full-dataset Raw and paired-sample Raw clearly distinguished.
3. Ensure the figure caption does not claim an intervention or causal effect.
4. Preserve original photographs and mark human annotations explicitly.
5. Compile in the team ICML document and inspect float placement, font size,
   margins and overall page count. Current deliverables have not been compiled.
