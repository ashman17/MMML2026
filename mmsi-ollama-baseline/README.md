# MMSI-Bench one-pass baseline through Ollama

This is a small, dependency-free reproduction of the benchmark's **vanilla,
single-response evaluation path**. It sends every question's images and text to
an Ollama vision model once, then applies the official MMSI answer prompt and
regular-expression extractor.

It intentionally does not install the full bundled VLMEvalKit repository.

## Run a five-question smoke test

```bash
cd "/Users/ashman/Documents/ChatGPT/Multimodal ML/mmsi-ollama-baseline"
python3 preflight.py
python3 run.py --limit 5 --sampling stratified
```

Results are written incrementally to `outputs/qwen3vl_4b.jsonl`; a summary is
written beside it. Re-running the same command resumes completed examples.

## Useful filters

```bash
python3 run.py --limit 10 --sampling stratified \
  --category "Positional Relationship (Cam.–Obj.)"

python3 run.py --id 66 --output outputs/question_66.jsonl

python3 run.py --difficulty hard --limit 10 --sampling stratified \
  --output outputs/hard_10.jsonl
```

For a full vanilla run, omit `--limit`. Keep a separate output file for every
model and experimental condition.

## Inspect a streamed claim trace

```bash
python3 inspect_example.py --id 66
```

This asks the model for concise numbered claims before its answer and streams
them token-by-token. It must first finish direct observations for Image 1, then
Image 2 (and any later images), then identify common aspects/correspondences,
and only afterward make spatial inferences. Claims cite one-based image numbers.
Rationale inspection defaults to Qwen's thinking-model temperature of `0.6`;
override it with `--temperature`. To wait for the full response, add
`--no-stream`.

To audit each claim using the images, correct answer, and human reference
reasoning in a separate post-hoc call:

```bash
python3 inspect_example.py --id 66 --verify-claims
```

The solver never sees the correct answer or reference reasoning. The audit is
model-generated and therefore useful for diagnosis, not ground-truth proof.

For the exact official direct-answer prompt instead, run:

```bash
python3 inspect_example.py --id 66 --official-direct
```

The optional machine-readable reasoning mode remains available with
`--structured`.

## Iteratively correct the model

```bash
python3 chat_example.py --id 3 --debug-inputs
```

The chat defaults to Qwen's thinking-model sampling recommendation:
`temperature=0.6`, `top_p=0.95`, and `top_k=20`. Override the temperature with
`--temperature` when needed. This does not change the official scored baseline,
which remains at temperature zero.

The model first produces the full observation/commonality/inference trace. At
the `feedback>` prompt, enter a correction such as `Claim 4 assumes a camera
direction that is not established.` The model then regenerates the entire trace
with the prior response and correction in context. Use `/quit` to save and exit.

The transcript is saved after every response and correction. Resume it with:

```bash
python3 chat_example.py --id 3 --resume
```

Ollama is stateless, so every generation resends the complete conversation,
including the original ordered image array. The transcript stores paths and
text rather than duplicate base64 image data.

To compare it with the dataset's human rationale after inference:

```bash
python3 inspect_example.py --id 66 --show-reference-rationale
```

Multiple `--id` flags are supported. Rationale modes show a prompted,
model-generated explanation—not access to private hidden reasoning—and are not
part of the official baseline score.

## Fidelity to the official benchmark

- official repository pinned at commit
  `13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac`;
- exact `MMSIBenchDataset.build_prompt` post-prompt;
- exact primary regular-expression answer extraction;
- temperature 0 and maximum output length 2,048, as stated in the paper;
- explicit 8,192-token Ollama context, needed because an eight-image MMSI item
  exceeded Ollama's 4,096-token runtime default;
- images remain in dataset order;
- explicit text-separated `[img]` placeholders prevent Ollama's Qwen-VL
  renderer from merging same-sized images into one temporal visual chunk;
- human reasoning annotations are never sent to the model.

Ollama tokenization, image preprocessing, and model weights differ from the
authors' original serving stacks. Therefore this reproduces the **evaluation
protocol**, not their exact published model numbers.

This runner covers vanilla accuracy. Circular testing and LLM-based fallback
answer extraction are deliberately excluded from the first smoke-test path.
