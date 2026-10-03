# Falsifiable Visual Proof Graphs: From Tool Use to Verifiable Multimodal Reasoning

*Research memo — 12 September 2026*

## Bottom line

Your idea has a credible research opening, but the novelty cannot simply be “generate several reasoning trees and verify them with tools.” Every component of that sentence already exists separately:

- **Tree of Thoughts** and VReST explore multiple reasoning paths.[^1][^2]
- **VisProg, ViperGPT, and VADAR** translate visual questions into executable programs over vision tools.[^3][^4][^5]
- **CoRGI, ViSA, and binary visual verification** ground or verify intermediate/candidate claims.[^6][^7][^8]
- **GCA, S-Agent, SpatialClaw, and Think3D** use explicit geometry or reconstructed spatial representations.[^9][^10][^11][^12]
- **PCRLLM and GroundedPRM** provide proof-like step structure or tool-validated process supervision, but outside this multi-image spatial setting.[^13][^14]

What I did **not** find is a system combining all of the following at inference time:

1. multiple competing, multi-image spatial proof graphs;
2. typed claims with explicit entities, views, coordinate frames, and tolerances;
3. independent tools returning **support, contradiction, or unknown** with provenance;
4. machine-checked propagation from observations to derived spatial claims;
5. active selection of the next test by its value for distinguishing answers;
6. explicit search for a **counterproof** to the currently leading answer; and
7. evaluation of whether the cited evidence actually caused the answer.

That combination is plausible research whitespace. I would call it **Falsifiable Visual Proof Graphs (F-VPG)**. Use “proof” operationally: perception is uncertain, so this is an auditable probabilistic certificate, not a mathematical proof that the pixels are true.

## 1. Where the nearest literature stops

| Research line | What it establishes | What remains missing |
|---|---|---|
| ToT / VReST | Branch, score, and backtrack across reasoning paths | The scorer is largely the model itself; nodes are not geometrically testable claims |
| VisProg / ViperGPT / VADAR | An LLM can compose detectors, depth models, and code | Successful execution is not evidence that each premise is correct; usually one answer-producing program is pursued |
| Visual Program Distillation | Sample several executable programs and retain useful traces | Candidate programs are filtered using the known final label during data generation, not independently verified at test time[^15] |
| CoRGI / VG-CoT | Link textual steps to image regions or detected evidence | Grounding a step does not establish its geometric validity; largely post-hoc, not competing proof search |
| ViSA | Convert reasoning into microclaims and label them entailed, contradicted, or insufficient | Verification relies primarily on a VLM and generated visual evidence, rather than independent multi-view geometric measurements |
| GCA | Formalize the query's reference frame and constrain deterministic computation | Commits to a task formulation; does not compare alternative proofs and counterproofs |
| S-Agent / SpatialClaw | Iterative spatial planning, tools, state, and cross-checks | Verification is an agent behavior, not a typed certificate with calibrated evidence dependencies |
| PCRLLM / SatLM | Structured premises and formal symbolic checking | The formal system can validate deductions from parsed premises, but cannot guarantee that visual premises were perceived correctly[^13][^16] |

Therefore, your method should not be pitched as a better toolbox. The scientific question is:

> **Can a VLM answer more accurately and more faithfully when it must construct falsifiable, frame-aware proof graphs and actively try to disprove its current spatial hypothesis?**

## 2. The proposed representation

A free-form chain is too ambiguous to verify. Represent reasoning in three linked layers.

### A. Evidence layer

Immutable observations tied to inputs and tool runs:

```text
E17:
  source: GroundingModel(image_2, "green chair")
  output: mask_17
  confidence: 0.91
  provenance: image_2, model_version, parameters
```

Evidence can be pixels, boxes, masks, correspondences, depth distributions, camera poses, tracks, or VLM judgments. A derived relation is never allowed to masquerade as independent evidence for its own premise.

### B. Claim layer

Atomic, typed, falsifiable predicates:

```text
C8:
  predicate: facing
  arguments: [green_chair, green_curtain]
  frame: room_global
  views: [image_1, image_2]
  tolerance: 30_degrees
  admissible_verifiers: [chair_orientation, reconstructed_pose]
  status: unknown
```

Every claim contains its coordinate frame, scope, admissible evidence, tolerance, provenance, and status in `{supported, contradicted, unknown}`. This blocks common errors such as treating image-right as room-right or interpreting a failed feature match as a contradiction.

Claims should be separated into:

- **perceptual:** `is_green(x)`, `visible(x, view_2)`, `same_entity(x@v1, x@v2)`;
- **geometric:** `relative_pose(v1,v2)`, `left_of(x,y,frame)`, `facing(x,y)`;
- **semantic bridge:** “sitting on this chair implies the observer faces along the chair's forward axis”; and
- **derived:** conclusions obtained from checked rules and supported premises.

### C. Answer-proof layer

Construct a proof DAG backward from every candidate answer. AND nodes require all premises; OR nodes encode alternative ways to establish the same claim. Shared claims are deduplicated, so four answer choices do not count one camera estimate four times.

For each option, the system records:

```text
conclusion -> required spatial relation
           -> alternative derivations
           -> atomic verifiable claims
           -> evidence and residuals
```

A deterministic checker enforces type compatibility and only a small library of valid rules. For example, left/right inversion is safe under a declared frame; qualitative spatial transitivity is generally **not** safe without geometric bounds.

## 3. How the chair–curtain example should run

Suppose the question asks: “When sitting on the green chair, where is the green curtain relative to you?”

### Step 1: Compile the question

The model identifies:

- observer origin: the green chair's sitting position;
- observer forward axis: the direction a normally seated person would face;
- target: green curtain;
- answer predicates: front-left, front-right, back-left, back-right.

The forward axis is an unresolved claim—not an assumption.

### Step 2: Generate genuinely different proof routes

**Route A — reconstructed geometry**

1. Ground chair and curtain across views.
2. Verify view overlap and camera pose using correspondences/VGGT.
3. estimate chair position and forward direction from seat/backrest geometry.
4. transform the curtain into the chair-centered coordinate frame.
5. compute the target bearing and map it to an answer region.

**Route B — landmark topology**

1. Verify the same dustbin across the relevant views.
2. establish bounded spatial relations among chair, dustbin, and curtain.
3. combine only relations whose frames agree and whose uncertainty bounds imply the requested quadrant.

**Route C — viewpoint transformation**

1. infer the motion between the entrance view and the later view;
2. locate chair and curtain in a shared room frame;
3. transform the target into the seated observer's frame.

These routes may share evidence, but they should differ in their critical inference. Merely paraphrasing one chain three times is not multiple verification.

### Step 3: Try to falsify the leader

If Route B initially suggests “front-left,” the scheduler asks for the cheapest claim that could overturn it. Here that is likely `facing(chair, curtain)`, not another left/right check.

Critically, “the chair is behind the dustbin, and the dustbin is left of the curtain” does **not** prove that the chair faces the curtain. Position does not determine orientation. Unless the backrest/seat geometry or another independent cue establishes the chair's forward direction, that claim remains `unknown`; Route B cannot complete its proof.

This is the proposed system's main advantage: it exposes an invalid but linguistically plausible bridge that an ordinary VLM may glide over.

### Step 4: Return a certificate

```json
{
  "answer": "front-left",
  "confidence": 0.73,
  "reference_frame": "chair-centered, forward from verified chair orientation",
  "surviving_proofs": ["geometry_route"],
  "rejected_proofs": [
    {"route": "landmark_route", "reason": "chair orientation unsupported"}
  ],
  "critical_evidence": ["chair mask", "curtain mask", "relative camera pose"],
  "unresolved_claims": ["dustbin identity in image 1"]
}
```

This trace is illustrative; it does not claim that those tools have actually verified this dataset example.

## 4. Verification engine

Each tool advertises an **evidence contract**:

```text
inputs | preconditions | output type | uncertainty | known failure modes
```

Useful verifier families are:

- **2D perception:** open-vocabulary detection, segmentation, OCR, attributes;
- **cross-view identity:** LightGlue/RANSAC, track consistency, embedding or mask agreement;
- **3D geometry:** VGGT/depth, camera pose, ray intersection, reprojection residuals;
- **motion:** optical flow, tracking, rigidity and background compensation;
- **symbolic checker:** coordinate transformations and typed spatial rules;
- **VLM verifier:** semantic or commonsense bridges that other tools cannot test.

The VLM verifier should be labeled weak evidence. A model must not independently “verify” a claim it proposed merely by rereading it. Prefer a different crop, representation, model, or tool, and record dependence between evidence sources.

Every verifier should return measurements rather than a bare yes/no:

```text
support probability | contradiction probability | unknown probability
residual/confidence interval | evidence provenance | reliability group
```

Calibrate these outputs on held-out data. Aggregate them using a factor graph or a small calibrated model, grouping dependent results so that a VGGT depth map and a relation derived from that map are not counted twice. This is more defensible than evidence-weighted majority voting.

The next tool call should maximize expected reduction in answer entropy per unit cost. A simple one-step policy estimated from development data is sufficient for a course project; reinforcement learning is unnecessary initially.

## 5. Generalizing across tasks

Do not attempt a universal prompt. Generalization should come from a stable **claim language** plus replaceable task adapters.

### Stable core ontology

- entities: object, person, region, camera;
- indices: image/view, time, scene;
- frames: image, camera/egocentric, object-centered, room/global, geographic;
- predicates: identity, visibility, attribute, containment, overlap, relative position, orientation, pose, distance, size, motion;
- operators: transform frame, compare intervals, compose poses, check consistency.

### MMSI task adapters

| Task family | Critical leaves and verifiers |
|---|---|
| Camera–Camera | relative pose, view overlap, camera motion; correspondences and pose consistency |
| Camera–Object | object grounding, camera pose, object position, observer frame |
| Object–Object | grounding/re-identification and relative 3D position |
| Region relations | region grounding, containment/topology, scene layout |
| Measurement | scale calibration, depth uncertainty, known reference size |
| Appearance | crop/attribute consistency across illumination and view |
| Motion | tracking/flow, temporal order, camera-motion compensation |
| Multi-step questions | composition of the same typed subgraphs |

Beyond MMSI, a new domain supplies new predicates or verifiers while retaining the search, provenance, three-valued logic, and certificate format. A chart task may use OCR and arithmetic; a medical-image task may use anatomy localization and clinical constraints; a video task adds temporal predicates.

This is **open-world extensibility**, not proof for every possible image. Some claims will remain unverifiable. A credible system must abstain internally, lower confidence, or choose the least contradicted option rather than fabricate support.

## 6. What makes the work publishable

Answer accuracy alone will not demonstrate verification. Create a 150–250-example proof annotation set over three MMSI categories, ideally Camera–Object, Camera–Camera, and Object–Object. Annotate atomic claims, coordinate frames, whether each inference rule is valid, and what evidence would settle it.

Evaluate:

- answer accuracy and matched-compute accuracy;
- claim-level support/contradiction/unknown accuracy;
- proof validity and completeness;
- calibration: Brier score, ECE, selective risk–coverage;
- causal faithfulness: remove or corrupt cited evidence and check whether the conclusion changes;
- counterfactual consistency: permute options, swap permissible image order, or alter a verified relation;
- verifier robustness: inject incorrect pose/correspondence outputs;
- cost: tool calls, latency, tokens, and GPU time.

The strongest baselines are direct prompting, CoT/self-consistency, Tree-of-Thought-style self-verification, one executable visual program, CoRGI-style post-hoc grounding, candidate-wise binary verification, a GCA-style single constrained plan, and a one-path spatial tool agent.

Essential ablations:

1. one proof versus competing proofs;
2. proof search with versus without counterproofs;
3. binary versus three-valued verification;
4. VLM-only versus tool-backed verifiers;
5. fixed versus information-gain tool selection;
6. heuristic voting versus calibrated aggregation;
7. no provenance/dependence constraints; and
8. free-form chains versus the typed checker.

A particularly novel test is **verification under pressure**: as the model is allowed to sample more proofs, does its apparent proof score rise while true accuracy stays flat or falls? If so, it is exploiting verifier weaknesses. Proof-Carrying Cognition, a very recent preprint, makes verifier soundness under optimization a central concern; treat it as an emerging parallel idea rather than established evidence.[^17]

## 7. Feasible scope for four people and ten weeks

Build a narrow MVP:

- three MMSI categories, with Camera–Object used for development;
- four predicate families: entity identity, position, orientation, and reference-frame transformation;
- one grounding model, LightGlue/RANSAC, VGGT, and a deterministic rule checker;
- one primary VLM and one transfer VLM;
- backward proof generation, shared DAG representation, and counterproof selection;
- no RL in the first paper.

Suggested ownership:

1. proof DSL and symbolic checker;
2. perception/geometry verifiers;
3. planner, search, and evidence aggregation;
4. annotations, evaluation, and statistical analysis.

Three early go/no-go tests:

- Can the tools settle at least roughly 60% of the critical atomic claims on a 50-example audit?
- Does proof validity predict answer correctness beyond the VLM's own confidence?
- At matched compute, do competing proofs plus counterproof search outperform a single constrained plan?

If the first test fails, pivot to a verifier-failure benchmark. If the second fails, the certificates are decorative. If the third fails, the publishable result may be the typed verification dataset and analysis rather than a stronger solver.

## Recommended claim

> Existing visual agents optimize trajectories that produce answers. We instead search for falsifiable proof graphs whose critical claims must be settled by independent, calibrated evidence contracts, and actively seek counterevidence before answering.

That is narrow enough to test, meaningfully different from generic tool use, and generalizable through a typed predicate/verifier interface.

## Sources

[^1]: Yao et al., [“Tree of Thoughts: Deliberate Problem Solving with Large Language Models”](https://proceedings.neurips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract.html), NeurIPS 2023.
[^2]: Cheng et al., [“VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism”](https://arxiv.org/abs/2506.08691), 2025.
[^3]: Gupta and Kembhavi, [“Visual Programming: Compositional Visual Reasoning Without Training”](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.html), CVPR 2023.
[^4]: Surís et al., [“ViperGPT: Visual Inference via Python Execution for Reasoning”](https://openaccess.thecvf.com/content/ICCV2023/html/Suris_ViperGPT_Visual_Inference_via_Python_Execution_for_Reasoning_ICCV_2023_paper.html), ICCV 2023.
[^5]: Marsili et al., [“Visual Agentic AI for Spatial Reasoning with a Dynamic API”](https://openaccess.thecvf.com/content/CVPR2025/html/Marsili_Visual_Agentic_AI_for_Spatial_Reasoning_with_a_Dynamic_API_CVPR_2025_paper.html), CVPR 2025.
[^6]: Yi and Shang, [“CoRGI: Verified Chain-of-Thought Reasoning with Visual Grounding”](https://arxiv.org/abs/2508.00378), 2025.
[^7]: Chandar et al., [“Probing Multimodal Spatial Reasoning through World Models”](https://arxiv.org/abs/2512.05809), 2025.
[^8]: Hu et al., [“Binary Verification for Zero-Shot Vision”](https://openaccess.thecvf.com/content/CVPR2026W/VAR/html/Hu_Binary_Verification_for_Zero-Shot_Vision_CVPRW_2026_paper.html), CVPR Workshops 2026.
[^9]: Chen et al., [“Geometrically-Constrained Agent for Spatial Reasoning”](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Geometrically-Constrained_Agent_for_Spatial_Reasoning_CVPR_2026_paper.html), CVPR 2026.
[^10]: Dai et al., [“S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence”](https://arxiv.org/abs/2606.20515), 2026.
[^11]: Cho et al., [“SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning”](https://arxiv.org/abs/2606.13673), 2026.
[^12]: Zhang et al., [“Think3D: Thinking with Space for Spatial Reasoning”](https://arxiv.org/abs/2601.13029), 2026.
[^13]: Wang et al., [“Proof-Carrying Reasoning with Large Language Models”](https://arxiv.org/abs/2511.08392), 2025.
[^14]: [“GroundedPRM: Tree Search with External-Tool Process Verification”](https://arxiv.org/abs/2510.14942), 2025.
[^15]: Hu et al., [“Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models”](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_Visual_Program_Distillation_Distilling_Tools_and_Programmatic_Reasoning_into_Vision-Language_CVPR_2024_paper.html), CVPR 2024.
[^16]: Ye et al., [“SatLM: Satisfiability-Aided Language Models Using Declarative Prompting”](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8e9c7d4a48bdac81a58f983a64aaf42b-Abstract-Conference.html), NeurIPS 2023.
[^17]: [“Proof-Carrying Cognition”](https://arxiv.org/abs/2609.09776), preprint, 9 September 2026.
