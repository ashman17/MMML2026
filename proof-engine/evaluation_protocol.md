# F-VPG evaluation protocol

## 1. Freeze before annotation

- Preserve the current 100-pair sample order and raw tool outputs.
- Do not expose pair type, question IDs, rationales, labels, or tool scores to
  annotators.
- Record the code commit, model weights, resize, keypoint budget, and runtime.

## 2. Overlap annotation

Two team members label every pair independently:

- `overlap`: the same physical surface or object region is visibly present in
  both images;
- `same_scene_no_overlap`: the images depict the same place but share no
  identifiable visible region;
- `different_scene`: the locations differ;
- `uncertain`: the evidence is inadequate.

For `overlap`, also label approximate shared image area as `<25%`, `25–50%`, or
`>50%`. Labels concern visible geometric overlap, not merely semantic similarity.

Export both JSON files separately. Compute four-way and binary-overlap Cohen's
κ. Adjudicate every disagreement without viewing tool scores. Report κ before
adjudication and retain the original labels.

## 3. Split and calibration

- Keep connected question IDs in one partition so the same image cannot appear
  in both calibration and evaluation.
- Select each tool threshold only on the calibration partition, targeting at
  most 5% empirical false-positive rate.
- Freeze thresholds before opening evaluation results.
- Exclude `uncertain` from the binary primary analysis; report its frequency.

## 4. Correspondence endpoints

Primary:

- evaluation false-positive rate at the frozen threshold;
- evaluation true-positive rate/coverage at that threshold.

Secondary:

- ROC AUC;
- precision;
- performance by overlap fraction and MMSI category;
- latency and peak memory;
- DINO, LightGlue, and agreement-gated comparison.

The verifier is eligible to create `supported` proof evidence only if it has a
low evaluation false-positive rate and an explicit `unknown` outcome. Otherwise
it remains a scheduling or representation signal.

## 5. Full proof-system evaluation

Compare, at matched compute:

1. direct VLM answer;
2. chain-of-thought/self-consistency;
3. single constrained tool path;
4. multiple proof paths without counterproof search;
5. F-VPG with counterproof search and calibrated evidence.

Report answer accuracy, claim validity, reference-frame validity, Brier score,
risk–coverage, tool calls, latency, and GPU time. Test causal faithfulness by
removing or corrupting cited evidence and verifying that the conclusion changes
or becomes unknown.

## 6. Statistical reporting

- Use paired bootstrap confidence intervals for accuracy differences.
- Use McNemar's exact test for paired answer changes.
- Report all attempted prompts, thresholds, and ablations.
- Treat Camera–Object as development; freeze the method before Camera–Camera
  and Object–Object transfer evaluation.
- Do not use MMSI reference rationales as solver inputs.
