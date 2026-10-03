# Official repository setup audit

Pinned official clone: `../official-mmsi-bench`, commit
`13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac`.

The clone itself is modest (about 24 MB before Git metadata), but installation
is comparatively cumbersome for an Ollama-only experiment:

- 41 top-level requirement lines;
- 415 Python files in the bundled VLMEvalKit tree;
- heavyweight Torch, TorchVision, Transformers, OpenCV, and video decoders;
- unrelated Gradio, spreadsheet, plotting, and multiple cloud-provider SDKs;
- platform-specific `eva-decord` on ARM Macs;
- model-specific imports can make environment failures unrelated to MMSI.

The full repository remains valuable for final leaderboard-compatible runs,
circular evaluation, and comparison with supported Hugging Face/API models.
For rapid Ollama experiments, the standalone runner is lower-risk and easier to
audit. It depends only on Python's standard library and the already-extracted
local dataset.
