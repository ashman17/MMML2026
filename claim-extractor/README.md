# Atomic claim extractor

Small terminal prototype for extracting claims and reasoning dependencies using
`deepseek-ai/DeepSeek-V4.1-Flash` on a Modal OpenAI-compatible endpoint.

## Setup

```bash
cd "/Users/ashman/Documents/ChatGPT/Multimodal ML/claim-extractor"
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Python 3.11 or newer is required. Avoid macOS's bundled Python 3.9 because it is
past end-of-life and uses an old LibreSSL build.

Add the Modal proxy credentials to `.env`:

```text
MODAL_PROXY_TOKEN_ID=your_token_id
MODAL_PROXY_TOKEN_SECRET=your_token_secret
MODAL_ENDPOINT=https://ashmanm--ep-deepseek-v4-1-flash-server.us-west.modal.direct/v1/chat/completions
```

`.env` is ignored by Git.

## Run

Interactive multiline input (finish with `Ctrl-D`):

```bash
python extract_claims.py
```

Or pass text directly:

```bash
python extract_claims.py --text 'The white chair is in-front of the window in Image 1. The table appears to the right of the window in image 2.'
```

Expected shape:

```json
{
  "claims": [
    {
      "id": "c1",
      "entity": "white chair",
      "image": "Image 1",
      "claim": "white chair in Image 1",
      "source_span": "The white chair is in-front of the window in Image 1."
    }
  ]
}
```

To inspect the exact prompt without making an API call:

```bash
python extract_claims.py --text 'your reasoning' --show-prompt
```

## Two-stage reasoning graph

`reasoning_graph.py` first extracts atomic claims, prints them as a table, then
sends the original trace and validated claim list through the dependency-graph
prompt and prints the resulting joint parent sets.

```bash
python reasoning_graph.py
```

Paste a multiline reasoning trace and finish with `Ctrl-D`. You can instead use
`--text '...'` or `--file trace.txt`. Add `--raw-json` to print both validated
JSON objects as well as the formatted terminal representation.

Temporary Modal server failures are retried three times. Override this with
`--attempts N`. Both calls use strict JSON Schema, SSE streaming,
`reasoning_effort: none`, and local Pydantic validation.

Prompt 1 is automatically saved to `artifacts/prompt1_checkpoint.json` before
Prompt 2 starts. If Prompt 2 fails, resume without rerunning Prompt 1:

```bash
python reasoning_graph.py --resume
```

Use `--checkpoint path/to/file.json` to choose another checkpoint. `--resume`
loads both the original reasoning trace and its validated atomic claims from it.
