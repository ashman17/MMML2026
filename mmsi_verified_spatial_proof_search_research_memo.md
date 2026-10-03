# Falsify Before You Answer: A Research Plan for Verified Multi-Image Spatial Reasoning

*Research memo — 11 September 2026*

## Executive verdict

The project is technically feasible for four people in 10–12 weeks, but the current proposal is **not sufficiently novel as written**. Recent systems already combine VLM planning, 2D/3D tools, iterative evidence accumulation, memory, and revision. The closest are **GCA**, **S-Agent**, **SpatialClaw**, and **ViSA**.[^2][^3][^4][^5]

The strongest publishable pivot is:

> **Option-conditioned spatial proof search:** construct a falsifiable proof graph for every answer choice, use calibrated geometric tests to support, contradict, or leave each claim unknown, and choose the next test by how much it is expected to discriminate among the surviving choices.

This differs from asking an agent to gather more evidence for one evolving answer. Its central hypothesis is: **geometry is more useful for falsifying candidate proofs than as extra context for a VLM.**

My assessment:

- **Course-project success:** high probability, if the scope is restricted to two geometry tools, one or two VLMs, and Camera–Object as the development category.
- **Workshop/arXiv-quality result:** realistic if the method improves both accuracy and faithfulness/calibration, and transfers beyond Camera–Object.
- **Main-conference paper:** possible but not assured; it likely requires cross-category or cross-benchmark generalization, strong matched-compute baselines, and a small claim-level evaluation set.

## 1. What MMSI-Bench supports—and what it does not

[MMSI-Bench](https://arxiv.org/abs/2505.23764) contains 1,000 human-authored multiple-choice questions, 1,990 unique images, an average of 2.55 images per question, and 11 spatial categories.[^1] Six categories are positional: Camera–Camera, Camera–Object, Camera–Region, Object–Object, Object–Region, and Region–Region. Images come from varied indoor, outdoor, egocentric, autonomous-driving, robotic, and reconstruction datasets.

The benchmark is well matched to your motivation. Its authors identify four major failure modes: grounding, overlap matching/scene reconstruction, reference-frame or situation transformation, and spatial logic. Overlap/reconstruction is the largest reported source of errors. Moreover, answer accuracy exceeds reasoning accuracy, so a correct multiple-choice answer does not guarantee a sound derivation.[^1]

The official baseline already tested:

- direct prompting;
- zero-shot chain-of-thought; and
- visual prompting using sparse cross-image correspondences.

Chain-of-thought was inconsistent, and correspondence overlays yielded only small, model-dependent gains.[^1] This is useful evidence for your representation study: geometry must be **selected, transformed, and uncertainty-qualified**, not merely drawn on the input.

Two evaluation cautions matter:

1. The public dataset exposes answers and reference rationales. Treat the full benchmark as a test set, freeze the method before the final evaluation, and never place reference rationales in the solver context.
2. Camera–Object is only a small slice of 1,000 questions—approximately 86 examples by inference from reported per-category score increments. Around 40% accuracy, a binomial 95% interval on 86 items is roughly ±10 percentage points. It is adequate for development, but too small to support a strong paper claim by itself.

The current official repository also lists materially stronger models than the paper's original baselines, with the leading listed result still far below human accuracy.[^15] Use the repository's current evaluator and report the date of the leaderboard snapshot.

## 2. Nearest prior work and the novelty boundary

| Work | Core idea | Overlap with your plan | Remaining opening |
|---|---|---|---|
| [MMSI-Bench](https://arxiv.org/abs/2505.23764) | Multi-image spatial benchmark; correspondence overlays and error taxonomy | Same benchmark and motivation | Verification and formal reasoning are not solved |
| [VADAR](https://openaccess.thecvf.com/content/CVPR2025/html/Marsili_Visual_Agentic_AI_for_Spatial_Reasoning_with_a_Dynamic_API_CVPR_2025_paper.html)[^12] | VLM writes programs against visual/spatial APIs | Tool-using visual agent | Not multi-view proof verification |
| [GCA](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Geometrically-Constrained_Agent_for_Spatial_Reasoning_CVPR_2026_paper.html) | VLM converts the query into a formal reference-frame/objective constraint, then uses deterministic tools | Constrained planning, VGGT, verifiable execution | It commits to one task formulation rather than comparing counterfactual answer proofs |
| [S-Agent](https://arxiv.org/abs/2606.20515) | Semantic planner, hierarchical 2D/3D tools, scene memory, agent memory, iterative evidence requests | Extremely close to the proposed tool hierarchy and revision loop | No explicit competing proof graphs, calibrated contradiction semantics, or information-gain test selection |
| [SpatialClaw](https://arxiv.org/abs/2606.13673) | Stateful Python kernel; inspect, execute, and revise using perception/geometry primitives | Iterative tool composition and cross-checking | Verification remains an agent behavior rather than a typed, machine-checkable proof object |
| [ViSA](https://arxiv.org/abs/2512.05809) | Question-conditioned microclaims verified as entailed, contradicted, or insufficient | Very close to node-level verification | It verifies primarily against generated visual evidence and a VLM, not calibrated multi-view geometric residuals |
| [VReST](https://arxiv.org/abs/2506.08691) / [Tree of Thoughts](https://proceedings.neurips.cc/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract.html)[^13] | Branching search with self-evaluation | Multiple hypotheses and pruning | Generic self-evaluation is not reliable geometric verification |
| [ViewFusion](https://arxiv.org/abs/2603.06024) / [Think3D](https://arxiv.org/abs/2601.13029) | Cross-view alignment or 3D-grounded reasoning-time representations | Supports the representation hypothesis | Does not establish option-conditioned, falsification-first search |

Therefore, avoid claiming novelty for any of the following alone:

- “an agent with spatial tools”;
- using VGGT or depth/pose estimates;
- iterative inspect-and-revise;
- multiple chains plus voting;
- decomposing reasoning into claims; or
- showing BEV/canonical views to a VLM.

The defensible novelty is the **combination of competing option-conditioned proof graphs, explicit three-valued geometric verification, active test selection, and calibrated evidence aggregation**.

## 3. Recommended method: Counterfactual Spatial Proof Search

### 3.1 Build proofs backward from answer choices

Instead of sampling four free-form chains of thought, parse each answer option into a typed proof graph. For a Camera–Object question, leaves may include:

```text
same_scene(I1, I2)
same_entity(object_x@I1, object_x@I2)
relative_pose(T_I2<-I1)
object_position(object_x, frame=I2)
bearing(object_x, frame=I2) ∈ front-left
```

Each claim must contain:

- operands and coordinate frame;
- the evidence types that can test it;
- provenance;
- a continuous residual or confidence interval; and
- status in `{supported, contradicted, unknown}`.

`Unknown` is essential: a failed match, weak texture, occlusion, or unstable reconstruction must not be interpreted as a contradiction.

### 3.2 Separate language, perception, and geometry

Assign the VLM tasks for which it is useful: resolve the query, identify candidate entities, detect reference-frame ambiguity, and create candidate proof structures. Use deterministic code for coordinate transforms, sign conventions, consistency checks, and option elimination.

Use a minimal tool set:

- **LightGlue + robust estimation:** sparse correspondence, overlap confidence, relative-pose hypotheses, inlier ratio, and reprojection residual. [LightGlue](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html) is fast and practical, but textureless surfaces, repeated patterns, appearance change, large baselines, and dynamic objects remain failure cases.[^6]
- **VGGT:** camera parameters, depth, point maps, and tracks from one or more views. [VGGT](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html) is feed-forward and practical for a two-view benchmark, but its predictions remain estimates rather than ground truth.[^7]
- **One grounding model:** boxes or masks for the object named in the question. Do not add a large general-purpose toolbox initially.

When possible, independently estimate a relation through LightGlue/RANSAC and VGGT. Agreement increases confidence; disagreement creates an `unknown` node or triggers a targeted fallback. This is stronger than letting the VLM judge two textual tool outputs.

### 3.3 Select tests by expected discrimination

At each step, choose the unresolved claim whose test is expected to reduce uncertainty over the answer choices most per unit cost:

$$
t^*=\arg\max_t \frac{\mathbb{E}[H(A\\mid E)-H(A\\mid E,o_t)]}{\mathrm{cost}(t)}.
$$

For the MVP, estimate test reliability and outcome frequencies from a held-out development set and use one-step lookahead. A learned planning policy is unnecessary.

Example: if three choices require “object left of camera 2” and only one requires “right,” verifying that bearing is more useful than rechecking a claim shared by every choice.

### 3.4 Aggregate calibrated evidence, not votes

Use the base VLM distribution as a prior and update it with calibrated evidence likelihoods:

$$
S(a)=\log p_0(a)+\sum_{c\in C_a}\log\frac{p(e_c\mid c)}{p(e_c\mid \neg c)}-\lambda N_{\mathrm{contradictions}}(a).
$$

Fit a small logistic or isotonic calibrator on held-out data. Group dependent measurements by evidence source so that, for example, a VGGT pose and a relation derived from the same VGGT output are not counted twice. Compare this against majority voting and manually chosen weights.

This produces both a forced multiple-choice answer and an inspectable proof certificate. It also enables risk–coverage analysis, even though official MMSI accuracy requires answering every item.

## 4. Representation study: make it a gate, not the headline

Before implementing full search, compare evidence interfaces under the same frozen VLM, geometry, token budget, and option order randomization:

1. original images only;
2. raw numeric geometry;
3. typed symbolic relation table;
4. visual overlay with matches, axes, boxes, and confidence;
5. canonical egocentric or top-down render; and
6. a **hybrid evidence packet**: one compact relation table, one annotated canonical view, and uncertainty/provenance.

The hypothesis—not yet a result—is that the hybrid packet will outperform both raw numbers and dense overlays. More importantly, test a representation × tool-reliability interaction: a useful interface should help the VLM distinguish reliable geometry from a failed reconstruction.

This phase is a go/no-go test and an ablation inside the final paper. Representation alone is unlikely to be a sufficient novelty claim because ViewFusion, Think3D, GCA, and related geometry-grounded systems already explore aligned or canonical spatial evidence.[^2][^8][^9]

## 5. Evaluation that could sustain a paper

### Baselines

Use matched VLMs and, where possible, matched tool budgets:

1. direct answer;
2. zero-shot CoT;
3. self-consistency at the same inference-token budget;[^14]
4. LightGlue/PATS-style overlay;
5. single fixed tool pass;
6. one-path iterative agent;
7. GCA-style formal constraint baseline;
8. proposed counterfactual proof search.

If reproducing S-Agent or SpatialClaw exactly is blocked by unavailable code, license, or model access, reproduce the closest interface faithfully and label it as a reimplementation—not the named system.

### Primary metrics

- forced-choice accuracy with paired bootstrap confidence intervals;
- paired McNemar tests against the strongest baseline;
- Brier score, negative log-likelihood, and expected calibration error;
- latency, token use, GPU time, and number of tool calls;
- claim support/contradiction accuracy on a manually annotated subset; and
- selective accuracy or risk–coverage curves as a secondary diagnostic.

### Essential ablations

- one proof versus all answer-conditioned proofs;
- fixed test order versus random versus information-gain selection;
- VLM-only, geometry-only, and hybrid verification;
- binary verification versus `{support, contradict, unknown}`;
- heuristic voting versus calibrated aggregation;
- remove the independent LightGlue/VGGT consistency check;
- each evidence representation; and
- matched compute curves, rather than only one expensive operating point.

### Robustness and faithfulness tests

- permute answer options;
- paraphrase the question without changing its geometry;
- swap image order when the semantics permit it;
- remove one view;
- inject controlled corruption into pose, depth, or correspondences;
- replace a tool output with an incompatible one; and
- test whether a contradiction changes the predicted option in the expected direction.

These tests measure whether the agent uses its evidence causally rather than producing a plausible post-hoc trace. Related generic work on visual claim grounding and external geometric trust estimation reinforces the need to validate the verifier rather than trust VLM self-confidence.[^10][^11]

### Generalization

Develop on Camera–Object, but freeze the design and test on at least two additional MMSI categories—Camera–Camera and Object–Object are natural choices—and one external benchmark such as ViewSpatial or MindCube. A Camera–Object-only result should be presented as a pilot, not general spatial intelligence.

A strong optional contribution is a 150–250-example, scene-disjoint **proof annotation set** containing atomic claims, reference frames, valid tests, verifier outcomes, and evidence quality. It would let you distinguish answer accuracy from proof faithfulness. Check the licenses of the original images before redistributing derived annotations or renders; MMSI's aggregate license does not automatically erase source-dataset restrictions.[^1]

## 6. Feasibility and principal risks

| Risk | Likelihood | Mitigation |
|---|---:|---|
| Geometry fails on low-overlap or dynamic views | High | Treat failure as unknown; cache diagnostics; add a semantic fallback only after measuring failure modes |
| VLM branches are correlated paraphrases | High | Derive proofs from mutually exclusive answer choices, not repeated stochastic CoT |
| Agent consumes compute without improving decisions | Medium–high | Information-gain stopping rule; matched-compute curves; cache geometry |
| Public-test overfitting | High | Fixed development subset from separate/scenedisjoint data; full MMSI once after freezing |
| Grounding errors dominate | High | Explicit masks/boxes, entity-identity claim, and grounding-specific metrics |
| Evidence double-counting creates false confidence | Medium | Source-grouped factors and held-out calibration |
| Contribution is overtaken by new agent papers | Medium–high | Anchor the paper on falsification, calibration, and causal proof tests—not the tool inventory |

[S-Agent](https://arxiv.org/abs/2606.20515) reports that tool augmentation can even hurt a smaller planner, illustrating that more tools are not automatically beneficial.[^3] [SpatialClaw](https://arxiv.org/abs/2606.13673) shows that a flexible stateful interface can yield strong gains, but it also raises the baseline your system must beat.[^4] These findings favor a small, explicit proof engine over a broad autonomous agent.

## 7. Ten-week execution plan for four people

| Weeks | Deliverable | Owner emphasis |
|---|---|---|
| 1 | Reproduce official evaluation; lock prompts, VLMs, budget, and data protocol | Evaluation/statistics |
| 2 | Manually audit 50 Camera–Object cases for overlap, grounding, reference frame, and tool applicability | All; one person curates taxonomy |
| 3–4 | LightGlue/VGGT cache; residuals and failure labels; six-condition representation pilot | Geometry + VLM interface |
| 4–6 | Typed option-conditioned proof graph and deterministic constraint propagation | Agent/reasoning |
| 6–7 | Active test selection and evidence calibration | Agent + statistics |
| 8 | Camera–Object experiments and error analysis; freeze system | All |
| 9 | Other MMSI categories, external benchmark, robustness, ablations | Evaluation |
| 10 | Paper, figures, reproducibility package, code cleanup | All |

Suggested division: (1) data/evaluation/statistics, (2) geometry/tools, (3) proof engine/search, and (4) VLM interface/experiments. Everyone should review the same error-analysis sample so that component boundaries do not conceal systematic failures.

### Go/no-go gates

- **End of week 2:** on 50 cases, at least one geometric route produces discriminative evidence on roughly 60–70% of relevant items. Otherwise pivot toward a rigorous representation/tool-failure study.
- **End of week 4:** the best evidence packet improves either accuracy or claim verification over original images at a matched budget. Otherwise do not build a large agent around it.
- **End of week 7:** proof search beats the strongest one-path baseline at matched compute and improves calibration. Otherwise simplify to deterministic formal constraints and publish the diagnostic result only if the analysis itself is novel.

## 8. Recommended paper claim

Working title:

> **Falsify Before You Answer: Counterfactual Spatial Proof Search for Multi-Image VLMs**

Keep the paper to three contributions:

1. an option-conditioned spatial proof representation with explicit reference frames;
2. active, three-valued geometric verification with calibrated evidence aggregation; and
3. an evaluation demonstrating when verification improves accuracy, faithfulness, and compute efficiency across multi-image spatial tasks.

Success should not be defined only as a few points on MMSI. A convincing result is one where the method (a) improves paired accuracy on more than one category or benchmark, (b) reduces confident wrong answers, (c) produces verifiably better claim-level proofs, and (d) reaches a better accuracy–cost frontier than a one-path tool agent.

## Sources

[^1]: Yang et al., [“MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence”](https://arxiv.org/html/2505.23764), ICLR 2026; [official repository and leaderboard](https://github.com/InternRobotics/MMSI-Bench); [dataset card](https://huggingface.co/datasets/RunsenXu/MMSI-Bench).
[^2]: Chen et al., [“Geometrically-Constrained Agent for Spatial Reasoning”](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Geometrically-Constrained_Agent_for_Spatial_Reasoning_CVPR_2026_paper.html), CVPR 2026.
[^3]: Dai et al., [“S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence”](https://arxiv.org/html/2606.20515), 2026; [official repository](https://github.com/Ropedia/S-Agent).
[^4]: Cho et al., [“SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning”](https://arxiv.org/html/2606.13673), 2026; [official repository](https://github.com/NVlabs/SpatialClaw).
[^5]: Chandar et al., [“Probing Multimodal Spatial Reasoning through World Models”](https://arxiv.org/html/2512.05809), 2025.
[^6]: Lindenberger et al., [“LightGlue: Local Feature Matching at Light Speed”](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html), ICCV 2023; [official repository](https://github.com/cvg/LightGlue).
[^7]: Wang et al., [“VGGT: Visual Geometry Grounded Transformer”](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html), CVPR 2025; [official repository](https://github.com/facebookresearch/vggt).
[^8]: Tao et al., [“ViewFusion: Structured Spatial Thinking Chains for Multi-View Reasoning”](https://arxiv.org/abs/2603.06024), 2026.
[^9]: Zhang et al., [“Think3D: Thinking with Space for Spatial Reasoning”](https://arxiv.org/abs/2601.13029), 2026.
[^10]: Yi and Shang, [“CoRGI: Verified Chain-of-Thought Reasoning with Visual Grounding”](https://arxiv.org/abs/2508.00378), 2025.
[^11]: Imran and Lee, [“Predicting When to Trust Vision-Language Models for Spatial Reasoning”](https://arxiv.org/abs/2601.11644), 2026.
[^12]: Marsili et al., [“Visual Agentic AI for Spatial Reasoning with a Dynamic API”](https://openaccess.thecvf.com/content/CVPR2025/html/Marsili_Visual_Agentic_AI_for_Spatial_Reasoning_with_a_Dynamic_API_CVPR_2025_paper.html), CVPR 2025.
[^13]: Yao et al., [“Tree of Thoughts: Deliberate Problem Solving with Large Language Models”](https://proceedings.neurips.cc/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract.html), NeurIPS 2023.
[^14]: Wang et al., [“Self-Consistency Improves Chain of Thought Reasoning in Language Models”](https://arxiv.org/abs/2203.11171), ICLR 2023.
[^15]: [MMSI-Bench official results table](https://github.com/InternRobotics/MMSI-Bench#results), accessed 11 September 2026.
