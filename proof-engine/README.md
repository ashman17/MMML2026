# F-VPG proof engine

Current measured pilot and its limitations: [`artifacts/pilot_status.md`](artifacts/pilot_status.md).

Prototype for the typed proof layer described in the research memo. It generates
structured candidate graphs with a local VLM, rejects claims that violate tool
contracts, and checks whether calibrated visual evidence and spatial inferences
form a structurally valid certificate.

## Run

```bash
python3 -m unittest discover -s tests -v
python3 examples/chair_curtain.py
python3 scripts/compile_mmsi.py ../mmsi-explorer/dist/records.json artifacts/mmsi_proof_skeletons.json
python3 scripts/audit_rationales.py ../mmsi-explorer/dist/records.json artifacts/rationale_audit.csv
python3 scripts/measure_dino_overlap.py ../mmsi-explorer/dist/records.json artifacts/dino_overlap_50.json \
  --repo /Users/ashman/.cache/torch/hub/facebookresearch_dinov2_main \
  --weights /Users/ashman/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
.venv/bin/python scripts/measure_lightglue_overlap.py ../mmsi-explorer/dist/records.json \
  artifacts/lightglue_overlap_50.json --lightglue-repo third_party/LightGlue
python3 scripts/compare_overlap_probes.py artifacts/dino_overlap_50.json \
  artifacts/lightglue_overlap_50.json artifacts/correspondence_comparison.json
python3 scripts/prepare_overlap_annotation.py ../mmsi-explorer/dist/records.json \
  artifacts/dino_overlap_50.json artifacts/lightglue_overlap_50.json \
  ../mmsi-explorer/dist/overlap_annotation_sample.json
python3 scripts/merge_overlap_annotations.py annotator_a.json annotator_b.json \
  --output artifacts/overlap_consensus.json
python3 scripts/evaluate_overlap_verifiers.py artifacts/overlap_consensus.json \
  artifacts/overlap_evaluation.json
python3 scripts/run_ollama_mmsi.py ../mmsi-explorer/dist/records.json \
  artifacts/qwen3vl_pilot.jsonl --model qwen3-vl:4b-instruct --condition both --limit 20
python3 scripts/summarize_vlm_run.py artifacts/qwen3vl_pilot.jsonl \
  artifacts/qwen3vl_pilot_summary.json

# Re-audit saved proof outputs after a verifier contract changes.
python3 scripts/audit_vlm_proofs.py artifacts/qwen3vl_pilot.jsonl \
  artifacts/qwen3vl_pilot_audited.jsonl
```

The example intentionally rejects the inference “chair position implies chair
orientation.” This should remain `unknown` until an orientation verifier supplies
independent evidence.

## Current boundary

- implemented: typed claims, frames, provenance, three-valued evidence,
  evidence contracts, proof alternatives, deterministic rule validation, a
  conservative MMSI compiler, local VLM direct/proof harness, DINO and
  ALIKED+LightGlue correspondence probes, and blinded calibration workflow;
- next: adapters for grounding and VGGT outputs, then active counterproof search;
- deliberately excluded: free-form VLM reasoning as proof evidence.

`audit_rationales.py` creates a conservative candidate list for manual review.
It only flags cases where the final rationale clause contains one unambiguous
direction that disagrees with the labeled option; it is not an automatic claim
that the benchmark label is wrong.

`artifacts/reviewed_rationale_inconsistencies.csv` contains the manually checked
subset where the answer text and rationale conclusion explicitly disagree. This
still does not establish which one matches the images.

The DINOv2 overlap script is a cheap feasibility probe using locally cached
weights. Its `supported` field is explicitly provisional: it measures patch
match coverage and affine consistency, not calibrated geometric correctness.
The measured negative controls show that this provisional rule is unsuitable as
proof evidence; see `artifacts/overlap_probe_report.md`.
