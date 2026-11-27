# Poster presentation notes

## 30-second version

Here are two pictures of the same cable. It shifts left in the pictures, even though it stays still in the scene. That distinction is hard for vision-language models. We tested whether camera geometry and named-object measurements help a model answer in the right frame. Geometry improved benchmark accuracy overall. On controlled scenes, making the coordinate transform and “still” rule explicit helped the reader, while small real displacements in the simulation often remained difficult to measure.

## Four-minute version

- **0:00–0:40 | Two images.** Point to the cable markers and ask for the two answers. The scene direction is measured horizontally using the first camera’s axes. Explain why camera motion can change the image without moving the target in the scene.
- **0:40–1:25 | One idea, two implementations.** The midterm gave a frozen model camera-pose hints. The adapted benchmark model learns from external geometry-derived labels and also receives continuous pose text. The diagnostic grounder instead tracks the object, combines depth and pose, and computes the requested displacement before a reader answers.
- **1:25–2:05 | Benchmark result.** Point to the overall and multi-image panels. On the same 1,000 items, 30.2% raw became 35.3% or 34.6% after adaptation plus hints. The gain over raw was supported in each seed. The smaller full-benchmark gain over hints alone remained uncertain. On the 198 multi-image reasoning questions, accuracy fell.
- **2:05–2:55 | What the controlled reader shows.** The 24 pairs have an image-motion and a scene-motion label. Both must be correct. The routed, thresholded reader gets 24/24 with verified evidence and 17/24 with predicted v2 evidence, matching deterministic readout on the latter. Applying the tolerance is already part of the geometry module, so this is evidence about the interface.
- **2:55–3:45 | What the grounder misses.** On 48 fresh scenes, the audit gives the frozen grounder a deterministic reader. It gets 46 image answers and 29 scene answers right. Small scene movers are the most frequent false still calls in this cohort. A lower threshold recovers seven directions but loses three previously correct still cases.
- **3:45–4:00 | Takeaway.** Pose is useful for camera questions. Object questions also require the target’s identity and reliable coordinates. A better readout can expose the remaining measurement error.

## Questions to rehearse

- **What does “verified” mean?** MOVi provides camera pose and the named object's simulator center of mass. Predicted 3D points instead come from its visible surface. Verified coordinates are an offline control.
- **How does the “still” rule work?** The deployed diagnostics use 0.02 image widths for image motion and 0.05 canonical scene units for scene motion. Simulator labels use the declared scene tolerance in simulator units. Canonical scene units come from setting median first-view depth to 5.
- **Did the explicit object grounder improve MMSI?** The full-benchmark gain is for the adapted VLM with pose text. Evidence for the explicit grounder comes from separately designed cohorts.
- **How general is the 48-scene result?** It is a visibility-filtered, deliberately motion-balanced subset of MOVi with one target per scene. The false-still rate describes these selected examples.
- **What did learned matching contribute?** In Ref-DAVIS17 the fitted head tied independent localization and trailed fixed matching by one correct primary reply. Numeric evidence still improved overall answers compared with raw, while unchanged cases exposed a threshold-use failure.
- **How much did pose versus center quality explain?** The offline center substitution recovered more labels than a pose substitution. A center substitution also supplies identity and depth, and either swap can compensate for another component’s error.

Have each teammate rehearse the short version and handle questions on the chart or method panel they present. Rotate during the session.
