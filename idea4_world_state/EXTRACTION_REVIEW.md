# Extraction pilot artifact review

Reviewed the saved original, v2, debug-v3 and final v3 artifacts against the
current code. No inference, scoring against gold answers, protocol freeze or
experiment-code changes were performed during this review.

## Verified alignment

- The active extraction instruction is `prompts/extract_v3.txt`.
- Original development: 21 accepted empty inventories; the two repeated unknown
  strings occur 15 and 6 times. Item 583's raw unknown is 178 characters long.
- Original and final development requests both contain matching image hashes
  between Raw and extraction for all 22 items. The referenced input files were
  hashed to verify their bytes, not merely their recorded hash strings.
- v2: 14 accepted inventories, with 8 rejected extractions. The saved v3 debug
  summary records 2 successful and 6 rejected extractions among those 8 items.
- Final `dev_v3`: 22 accepted, nonempty inventories and 66 successful answer
  records. Its 88 calls, plus 6 `timing_v3` calls and 1 `synthetic10_v3` call,
  all have attempt 1 and outcome `ok`.
- The 22 final inventories contain 57 entities, 67 observations, 0 claims,
  0 identity links and 22 unknowns. Items 188 and 524 have unknown strings of
  161 and 180 characters, respectively.
- All 10 non-null cross-view objects were inserted by the parser. They occur
  in 6 items: 14, 33, 236, 319, 531 and 912. No final raw extraction response
  supplied a non-null cross-view object itself.
- Reparsing every final raw extraction response with current code exactly
  reproduces its saved inventory.
- The final development, timing and synthetic runs all match current prompt,
  JSON schema, serializer and aggregate run-code hashes. They use the same
  Qwen checkpoint revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- The current protocol-compatibility check and all existing development gates
  pass. Flat and Graph have equivalent fact atoms on all 22 development items.
  The synthetic prompt has 5,876 tokens. No protocol freeze artifact was found.
- Final recorded request latency totals: development 233.53 seconds, timing
  15.33 seconds, synthetic10 0.96 seconds. These are sums of request latencies,
  not whole-job duration or GPU allocation charges.

## Findings before freeze

### Parser behavior is not protected by the freeze

`runner.py` records an aggregate `code_sha256` with each run, including
`inventory.py`. However, `_require_tested_artifacts`, `_freeze_document` and
`_enforce_freeze` protect prompt hashes, the JSON schema and `serialize.py`,
not parser/normalization code. A later change to the cross-view repair rule can
therefore change what evidence means without invalidating a freeze.

Reproduction: simulate a different hash for `inventory.py` in memory. The
aggregate code hash differs from the development run, but
`_require_same_protocol(dev_v3, timing_v3, synthetic10_v3, None)` still passes.
No files were changed in this reproduction. Today's final runs do match the
current parser; the problem is missing enforcement of the protocol lock.

Before freezing, bind parser/normalization behavior into the frozen fingerprint
and validate it both against development provenance and before evaluation.
The existing matching aggregate run-code hashes provide evidence of which
parser these runs used; do not silently backfill an unverified parser version.

### Repair success must be distinguished from model compliance

The implementation and updated plan agree on filling shared-id cross-view
objects with `status: same, uncertainty: medium`. Reusing an entity ID already
encodes an identity assertion under this representation, so making that
assertion explicit is a defensible normalization. The confidence label
`medium`, however, is supplied by code rather than estimated by the model.

Both evidence formats present that uncertainty without identifying its source.
The analysis also counts these as cross-view matches. Report them as
parser-normalized identity assertions, not independently established matches
or model-reported uncertainty. Raw responses remain available for auditing;
a per-item repair record would make the distinction easier to inspect.

Use these development counts when describing compliance:

- 16/22 satisfy the original cross-view semantic requirement without repair.
- 6/22 need repair, affecting 10 entities.
- 22/22 are accepted after normalization.

Inspect those six items against their images before treating the identity
assertions as reliable. This review verifies provenance and consistency; it
does not establish the visual correctness of every extracted entity or match.

### The current representation is sparse, not a relational world-state result

There are no explicit relation or attribute claims and no identity-link edges.
The graph still groups observations under entity nodes and shared identities,
so saying it has no structure at all would be too strong. It can test sparse
entity/observation evidence and its presentation, but cannot establish the
benefit of extracted spatial-relation edges that never appeared in development.

Both v3 JSON examples still contain empty claims arrays. Continued example
imitation is one plausible reason for zero claims, but the artifacts alone do
not establish causality. If the final allowed prompt revision is used, a small
development-only test with an explicitly supported within-image claim example
would directly address this limitation. Do not demand unsupported cross-view
relations or select prompts using evaluation answers.

## Validation and verdict

Focused inventory, serialization, prompt and debugging tests: **81 passed**.
Read-only checks also confirmed the final run fingerprints, regenerated
inventories and current development gates.

The supplied account substantially matches the code and artifacts. Before a
protocol lock, fix parser fingerprint enforcement and explicitly document
normalization/uncertainty provenance. Whether to accept sparse inventories or
use the remaining prompt revision is a scientific scope decision. Neither
choice should be presented as evidence that graphs improve answer accuracy.
