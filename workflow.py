#!/usr/bin/env python3
"""Reproducible MMSI-Bench preparation, image model runs and blinded review."""

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
OFFICIAL_URL = "https://huggingface.co/datasets/RunsenXu/MMSI-Bench/resolve/main/MMSI_Bench.parquet"
OFFICIAL_SHA256 = "72a9c94699ca88f3eba79f330ab28965ee537bccd7cbf563265982a6ff8b74b4"
CMU_GATEWAY_URL = "https://ai-gateway.andrew.cmu.edu"
FAILURES = ("grounding", "cross_view", "reference_frame_or_motion", "spatial_logic", "other", "none", "unclear")
PROMPT_VERSION = "images-only-v3"
DIFFICULTIES = ("easy", "medium", "hard")


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_path(provider, model, condition):
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    return RUNS / f"{provider}_{slug}_{condition}.jsonl"


def local_secret(name):
    if os.environ.get(name):
        return os.environ[name]
    path = ROOT / ".env.local"
    if not path.exists():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
    return ""


def download(args):
    dest = DATA / "MMSI_Bench.parquet"
    DATA.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256(dest) == OFFICIAL_SHA256:
            print("Official dataset already present and checksum verified")
            return
        raise SystemExit("Dataset file exists but checksum differs; inspect it before retrying")
    part = DATA / "MMSI_Bench.parquet.part"
    req = urllib.request.Request(OFFICIAL_URL, headers={"User-Agent": "MMML-evaluation/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response, part.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    actual = sha256(part)
    if actual != OFFICIAL_SHA256:
        raise SystemExit(f"Dataset checksum mismatch: {actual}")
    part.rename(dest)
    print(f"Downloaded and verified {dest}")


def prepare(args):
    import pyarrow.parquet as pq

    src = DATA / "MMSI_Bench.parquet"
    if not src.exists():
        raise SystemExit(f"Missing {src}")
    if (DATA / "items.jsonl").exists() and not args.force:
        raise SystemExit("Already prepared; use --force to regenerate")
    pf = pq.ParquetFile(src)
    print("Schema:", pf.schema_arrow.names)
    needed = {"id", "images", "question_type", "question", "answer", "thought"}
    missing = needed - set(pf.schema_arrow.names)
    if missing:
        raise SystemExit(f"Missing required fields: {sorted(missing)}")
    fixed_sample = ROOT / "protocol" / "sample.json"
    if args.sample_only and not fixed_sample.exists():
        raise SystemExit("protocol/sample.json is required for --sample-only")
    selected = set(json.loads(fixed_sample.read_text(encoding="utf-8"))["ids"]) if args.sample_only else None
    items, keys = [], []
    image_dir = DATA / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for batch in pf.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            item_id = str(row["id"])
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", item_id)
            image_paths = []
            for n, blob in enumerate(row["images"] or [] if selected is None or item_id in selected else []):
                if isinstance(blob, dict):
                    blob = blob.get("bytes")
                if not isinstance(blob, bytes):
                    raise ValueError(f"Unexpected image format for {item_id}: {type(blob)}")
                suffix = ".png" if blob.startswith(b"\x89PNG") else ".jpg"
                rel = f"images/{safe_id}_{n}{suffix}"
                (DATA / rel).write_bytes(blob)
                image_paths.append(rel)
            items.append({"id": item_id, "question_type": row["question_type"],
                          "question": row["question"], "images": image_paths,
                          "difficulty": row.get("difficulty"),
                          "mean_normed_duration_seconds": row.get("mean_normed_duration_seconds")})
            keys.append({"id": item_id, "answer": row["answer"], "thought": row["thought"]})
    if len({x["id"] for x in items}) != len(items):
        raise ValueError("Duplicate IDs in dataset")
    write_jsonl(DATA / "items.jsonl", items)
    write_jsonl(DATA / "keys.jsonl", keys)
    if args.sample_only:
        (DATA / "sample.json").write_bytes(fixed_sample.read_bytes())
    (DATA / "provenance.json").write_text(json.dumps({"source": "RunsenXu/MMSI-Bench",
        "file": src.name, "sha256": sha256(src), "rows": len(items),
        "categories": dict(Counter(x["question_type"] for x in items))}, indent=2), encoding="utf-8")
    print(f"Prepared {len(items)} questions and {sum(len(x['images']) for x in items)} images")


def sample(args):
    items = read_jsonl(DATA / "items.jsonl")
    if not items:
        raise SystemExit("Run prepare first")
    reviews_db = DATA / "reviews.sqlite3"
    if reviews_db.exists():
        with sqlite3.connect(reviews_db) as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='answers'").fetchone():
                if conn.execute("SELECT 1 FROM answers LIMIT 1").fetchone():
                    raise SystemExit("Human answers exist; preserve this sample and its assignments")
    if (DATA / "sample.json").exists() and not args.force:
        raise SystemExit("Sample already exists; use --force to replace")
    if len(args.reviewers) != 4 or len(set(args.reviewers)) != 4:
        raise SystemExit("Give four distinct reviewer names")
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for item in items:
        difficulty = str(item.get("difficulty") or "").lower()
        if difficulty in DIFFICULTIES:
            groups[(item["question_type"], difficulty)].append(item["id"])
    chosen = []
    assignment = {}
    sets = {"A": [], "B": []}
    categories = sorted({item["question_type"] for item in items})
    for cat in categories:
        for difficulty in DIFFICULTIES:
            ids = sorted(groups[(cat, difficulty)])
            if len(ids) < 2:
                raise SystemExit(f"Need two {difficulty} examples in {cat}; found {len(ids)}")
            selected = rng.sample(ids, 2)
            for set_name, item_id in zip(("A", "B"), selected):
                chosen.append(item_id)
                sets[set_name].append(item_id)
                assignment[item_id] = args.reviewers[:2] if set_name == "A" else args.reviewers[2:]
    payload = {"seed": args.seed, "per_category": 6, "per_difficulty": 2,
               "reviewers": args.reviewers, "sets": sets, "ids": chosen, "assignments": assignment}
    serialized = json.dumps(payload, indent=2) + "\n"
    (DATA / "sample.json").write_text(serialized, encoding="utf-8")
    protocol = ROOT / "protocol" / "sample.json"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(serialized, encoding="utf-8")
    print(f"Selected {len(chosen)} questions; workloads: {dict(Counter(r for v in assignment.values() for r in v))}")


def parse_answer(raw):
    text = str(raw or "")
    matches = re.findall(r"(?:final\s+answer|answer)\s*[:：]\s*\(?([A-D])\)?\b", text, re.I)
    if matches:
        return matches[-1].upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        match = re.fullmatch(r"(?:\*\*)?\(?([A-D])\)?[.)]?(?:\*\*)?", lines[-1], re.I)
        if match:
            return match.group(1).upper()
    return None


def gateway_models(args):
    key = local_secret("MMML_LITELLM_API_KEY")
    if not key:
        raise SystemExit("Set MMML_LITELLM_API_KEY in .env.local first")
    base_url = args.base_url or local_secret("MMML_LITELLM_BASE_URL") or CMU_GATEWAY_URL
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Gateway model listing failed: HTTP {exc.code}") from exc
    for item in body.get("data", []):
        if item.get("id"):
            print(item["id"])


def model_run(args):
    sample_data = json.loads((DATA / "sample.json").read_text(encoding="utf-8"))
    items = {x["id"]: x for x in read_jsonl(DATA / "items.jsonl")}
    if args.provider == "litellm" and not args.model:
        args.model = local_secret("MMML_LITELLM_MODEL")
    if not args.model:
        raise SystemExit("Set --model (or MMML_LITELLM_MODEL for LiteLLM)")
    base_url = args.base_url or (local_secret("MMML_LITELLM_BASE_URL") or CMU_GATEWAY_URL if args.provider == "litellm" else None) or ("https://api.openai.com/v1" if args.provider == "openai" else
                                "https://generativelanguage.googleapis.com/v1beta" if args.provider == "gemini" else "")
    if not base_url.startswith(("http://", "https://")):
        raise SystemExit("--base-url must be an HTTP(S) URL for compatible providers")
    out = run_path(args.provider, args.model, args.condition)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {x["id"] for x in read_jsonl(out) if x.get("error") is None and x.get("prompt_version") == PROMPT_VERSION}
    key_name = ("MMML_OPENAI_API_KEY" if args.provider == "openai" else
                "MMML_GEMINI_API_KEY" if args.provider == "gemini" else
                "MMML_LITELLM_API_KEY" if args.provider == "litellm" else "MMML_API_KEY")
    key = local_secret(key_name)
    if args.provider in ("openai", "gemini", "litellm") and not key:
        raise SystemExit(f"Set {key_name} in .env.local first")
    endpoint = (base_url.rstrip("/") + ("/responses" if args.provider == "openai" else
                f"/models/{args.model}:generateContent" if args.provider == "gemini" else "/chat/completions"))
    selected = sample_data["ids"][:args.limit] if args.limit else sample_data["ids"]
    for pos, item_id in enumerate(selected, 1):
        if item_id in done and not args.force:
            continue
        item = items[item_id]
        prompt = ("Answer this spatial reasoning multiple-choice question using the supplied images. "
                  "Briefly explain the visual evidence you used. Always end with 'Final answer: X', "
                  "where X is A, B, C, or D.\n\n"
                  + str(item["question"]))
        content = [{"type": "text", "text": prompt}]
        gemini_parts = [{"text": prompt}]
        if not item["images"]:
            raise SystemExit(f"Selected item {item_id} has no images")
        for image_number, rel in enumerate(item["images"], 1):
            path = DATA / rel
            mime = "image/png" if path.suffix == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "text", "text": f"Image {image_number}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            gemini_parts.append({"text": f"Image {image_number}:"})
            gemini_parts.append({"inline_data": {"mime_type": mime, "data": encoded}})
        if args.provider == "openai":
            input_content = [({"type": "input_text", "text": part["text"]} if part["type"] == "text" else
                              {"type": "input_image", "image_url": part["image_url"]["url"]})
                             for part in content]
            payload = {"model": args.model, "input": [{"role": "user", "content": input_content}],
                       "max_output_tokens": args.max_tokens, "store": False}
        elif args.provider == "gemini":
            payload = {"contents": [{"role": "user", "parts": gemini_parts}],
                       "generationConfig": {"maxOutputTokens": args.max_tokens,
                                            "temperature": 0 if args.temperature is None else args.temperature}}
        else:
            payload = {"model": args.model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": args.max_tokens}
            if args.provider == "compatible" or args.temperature is not None:
                payload["temperature"] = 0 if args.temperature is None else args.temperature
        headers = {"Content-Type": "application/json"}
        if args.provider == "gemini":
            headers["x-goog-api-key"] = key
        elif key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
        record = {"id": item_id, "condition": args.condition, "model": args.model,
                  "provider": args.provider, "base_url": base_url,
                  "temperature": payload.get("temperature", payload.get("generationConfig", {}).get("temperature")),
                  "max_tokens": args.max_tokens, "prompt": prompt,
                  "prompt_version": PROMPT_VERSION,
                  "dataset_sha256": json.loads((DATA / "provenance.json").read_text())["sha256"]}
        try:
            for attempt in range(args.retries + 1):
                try:
                    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                        body = json.load(resp)
                    break
                except urllib.error.HTTPError as exc:
                    try:
                        detail = json.loads(exc.read(4096)).get("error", {})
                        code = detail.get("code")
                        reason = str((code if code and not str(code).isdigit() else None)
                                     or detail.get("message") or detail.get("type") or exc.reason)
                    except (ValueError, UnicodeDecodeError):
                        reason = str(exc.reason)
                    reason = reason.replace(key, "[REDACTED]")[:600] if key else reason[:600]
                    if reason in ("insufficient_quota", "billing_hard_limit_reached", "credit_balance_exhausted"):
                        raise ValueError(f"HTTP {exc.code}: {reason}") from exc
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == args.retries:
                        raise ValueError(f"HTTP {exc.code}: {reason}") from exc
                    wait = min(30, 2 ** attempt * 2)
                    print(f"HTTP {exc.code}; retrying in {wait}s", flush=True)
                    time.sleep(wait)
            if args.provider == "openai":
                raw = body.get("output_text") or "\n".join(
                    part.get("text", "") for output in body.get("output", [])
                    for part in output.get("content", []) if part.get("type") == "output_text")
            elif args.provider == "gemini":
                raw = "\n".join(part.get("text", "") for part in
                    body["candidates"][0]["content"].get("parts", []) if "text" in part and not part.get("thought"))
            else:
                raw = body["choices"][0]["message"].get("content", "")
                if isinstance(raw, list):
                    raw = "\n".join(part.get("text", "") for part in raw if isinstance(part, dict))
            record.update(raw_output=raw, parsed_answer=parse_answer(raw),
                          usage=body.get("usage") or body.get("usageMetadata"),
                          response_id=body.get("id") or body.get("responseId"), error=None)
        except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as exc:
            record["error"] = str(exc)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{pos}/{len(selected)} {item_id}: {record.get('parsed_answer') or record.get('error')}", flush=True)
        if record["error"]:
            raise SystemExit("Model request failed; inspect the last run record and resume after fixing the endpoint")
        if args.delay:
            time.sleep(args.delay)


def db_connect():
    conn = sqlite3.connect(DATA / "reviews.sqlite3")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS answers (reviewer TEXT, item_id TEXT, answer TEXT, confidence INTEGER, seconds REAL, evidence_images TEXT, explanation TEXT, created_at REAL, PRIMARY KEY(reviewer,item_id))")
    conn.execute("CREATE TABLE IF NOT EXISTS traces (reviewer TEXT, item_id TEXT, first_bad_claim TEXT, claim_status TEXT, failure_type TEXT, evidence_images TEXT, notes TEXT, created_at REAL, PRIMARY KEY(reviewer,item_id))")
    conn.execute("CREATE TABLE IF NOT EXISTS trace_access (reviewer TEXT PRIMARY KEY, started_at REAL)")
    return conn


def export_review(args):
    sample_data = json.loads((DATA / "sample.json").read_text(encoding="utf-8"))
    if args.reviewer not in sample_data["reviewers"]:
        raise SystemExit("Unknown reviewer profile")
    with db_connect() as conn:
        answers = [dict(zip(("reviewer", "item_id", "answer", "confidence", "seconds", "evidence_images", "explanation", "created_at"), row))
                   for row in conn.execute("SELECT * FROM answers WHERE reviewer=? ORDER BY item_id", (args.reviewer,))]
        traces = [dict(zip(("reviewer", "item_id", "first_bad_claim", "claim_status", "failure_type", "evidence_images", "notes", "created_at"), row))
                  for row in conn.execute("SELECT * FROM traces WHERE reviewer=? ORDER BY item_id", (args.reviewer,))]
    payload = {"format": "mmml-review-v1", "reviewer": args.reviewer,
               "sample_sha256": sha256(DATA / "sample.json"), "answers": answers, "traces": traces}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        raise SystemExit("Export already exists; use --force to replace")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported {len(answers)} answers and {len(traces)} trace reviews to {out}")


def import_review(args):
    sample_path = DATA / "sample.json"
    sample_data = json.loads(sample_path.read_text(encoding="utf-8"))
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    reviewer = payload.get("reviewer")
    if payload.get("format") != "mmml-review-v1" or payload.get("sample_sha256") != sha256(sample_path):
        raise SystemExit("Submission format or sample checksum does not match this workspace")
    if reviewer not in sample_data["reviewers"]:
        raise SystemExit("Unknown reviewer profile")
    schemas = (("answers", ("reviewer", "item_id", "answer", "confidence", "seconds", "evidence_images", "explanation", "created_at")),
               ("traces", ("reviewer", "item_id", "first_bad_claim", "claim_status", "failure_type", "evidence_images", "notes", "created_at")))
    with db_connect() as conn:
        for table, columns in schemas:
            for row in payload.get(table, []):
                item_id = str(row.get("item_id"))
                if row.get("reviewer") != reviewer or reviewer not in sample_data["assignments"].get(item_id, []):
                    raise ValueError(f"Invalid assignment in {table}: {item_id}")
                values = tuple(row[c] for c in columns)
                existing = conn.execute(f"SELECT * FROM {table} WHERE reviewer=? AND item_id=?", (reviewer, item_id)).fetchone()
                if existing and tuple(existing) != values:
                    raise ValueError(f"Conflicting {table} record for {reviewer}, {item_id}")
                if not existing:
                    conn.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in columns)})", values)
    print(f"Imported {len(payload.get('answers', []))} answers and {len(payload.get('traces', []))} trace reviews from {reviewer}")


def load_model_outputs(provider, model):
    outputs = {}
    if not provider or not model:
        return outputs
    for row in read_jsonl(run_path(provider, model, "images")):
        if row.get("error") is None and row.get("prompt_version") == PROMPT_VERSION:
            outputs[row["id"]] = row
    return outputs


PAGE = """<!doctype html><html><head><meta charset='utf-8'><title>MMSI human review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:25px auto;padding:0 16px;color:#17202a}button,input,select,textarea{font:inherit}button{padding:8px 14px;cursor:pointer}nav{display:flex;gap:10px;margin-bottom:20px}.images{display:flex;gap:12px;overflow-x:auto}.images figure{margin:0;min-width:240px}.images img{max-height:360px;max-width:440px;object-fit:contain;border:1px solid #ccc}.question{white-space:pre-wrap;background:#f4f6f7;padding:16px;border-radius:8px}.row{margin:12px 0}textarea{width:100%;min-height:75px}label{margin-right:14px}.muted{color:#667}#message{color:#046}</style></head>
<body><h1>MMSI-Bench review</h1><p id='who'></p><nav><button onclick="stage='answer';currentId=null;load()">Independent answer</button><button onclick="stage='trace';currentId=null;load()">Trace review</button><button onclick="load(true)">Previous</button></nav><div id='app'></div><p id='message'></p>
<script>
let reviewer=new URLSearchParams(location.search).get('reviewer')||prompt('Reviewer name');let stage='answer',started=Date.now(),currentId=null;
document.getElementById('who').textContent='Reviewer: '+reviewer;
function esc(s){let d=document.createElement('div');d.textContent=s??'';return d.innerHTML}
async function load(previous=false){let old=document.getElementById('form');if(previous&&old?.dataset.dirty==='yes'&&!confirm('Discard unsaved changes?'))return;document.getElementById('message').textContent='';let url=(previous?'/api/previous?':'/api/next?')+'reviewer='+encodeURIComponent(reviewer)+'&stage='+stage;if(previous&&currentId)url+='&before='+encodeURIComponent(currentId);let r=await fetch(url);let d=await r.json();let a=document.getElementById('app');if(d.error){document.getElementById('message').textContent=d.error;return}if(d.done){currentId=null;a.textContent='All assigned '+stage+' reviews complete ('+d.completed+'/'+d.total+').';return}started=Date.now();let x=d.item;currentId=x.id;
let images=x.images.map((p,i)=>`<figure><img src="/image/${encodeURIComponent(p)}"><figcaption>Image ${i+1}</figcaption></figure>`).join('');
let common=`<p class="muted">${d.completed}/${d.total} completed · Category: ${esc(x.question_type)} · ID: ${esc(x.id)}</p><div class="images">${images}</div><div class="question">${esc(x.question)}</div>`;
if(stage==='answer'){a.innerHTML=common+`<form id="form"><div class="row">Answer: ${['A','B','C','D'].map(v=>`<label><input type="radio" name="answer" value="${v}" required>${v}</label>`).join('')}</div><div class="row">Confidence: <select name="confidence"><option value="1">1 very low</option><option value="2">2 low</option><option value="3" selected>3 medium</option><option value="4">4 high</option><option value="5">5 very high</option></select></div><div class="row">Brief reasoning or ambiguity note:<textarea name="explanation"></textarea></div><button>Save independent answer</button></form>`}
else{a.innerHTML=common+`<p><b>Your independent answer:</b> ${esc(d.human_answer)}<br><b>Benchmark correct answer:</b> ${esc(d.gold_answer)}</p><div class="question"><b>Model answer:</b> ${esc(d.model.parsed_answer)}<br><b>Model trace:</b><br>${esc(d.model.raw_output)}</div><form id="form"><div class="row">First incorrect or unsupported claim (leave blank if correct):<textarea name="first_bad_claim"></textarea></div><div class="row">Claim status: <select name="claim_status"><option value="correct" selected>Correct</option><option value="incorrect">Incorrect</option><option value="unsupported">Unsupported by images</option><option value="unclear">Cannot determine</option></select></div><div class="row">Failure type: <select name="failure_type">${['none','grounding','cross_view','reference_frame_or_motion','spatial_logic','other','unclear'].map(v=>`<option>${v}</option>`).join('')}</select></div><div class="row">Notes:<textarea name="notes"></textarea></div><button>Save trace review</button></form>`}
let form=document.getElementById('form');if(d.saved){for(let [name,value] of Object.entries(d.saved)){let field=form.elements.namedItem(name);if(field)field.value=value??''}form.querySelector('button').textContent='Save revised review'}form.addEventListener('input',()=>form.dataset.dirty='yes');
form.onsubmit=async e=>{e.preventDefault();let f=new FormData(e.target);let body=Object.fromEntries(f.entries());body.item_id=x.id;body.reviewer=reviewer;body.seconds=(Date.now()-started)/1000;let r=await fetch('/api/'+stage,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let z=await r.json();if(z.error){document.getElementById('message').textContent=z.error}else load()};}
load();
</script></body></html>"""


def serve(args):
    if args.provider == "litellm" and not args.model:
        args.model = local_secret("MMML_LITELLM_MODEL")
    sample_data = json.loads((DATA / "sample.json").read_text(encoding="utf-8"))
    items = {x["id"]: x for x in read_jsonl(DATA / "items.jsonl")}
    keys = {x["id"]: str(x["answer"]).strip().upper() for x in read_jsonl(DATA / "keys.jsonl")}
    assignments = sample_data["assignments"]
    db_connect().close()

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, mime="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def json(self, code, obj):
            self.send(code, json.dumps(obj), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                return self.send(200, PAGE, "text/html; charset=utf-8")
            if url.path.startswith("/image/"):
                from urllib.parse import unquote
                rel = unquote(url.path[len("/image/"):])
                if not re.fullmatch(r"images/[A-Za-z0-9_.-]+\.(jpg|png)", rel):
                    return self.json(404, {"error": "Not found"})
                path = DATA / rel
                if not path.is_file():
                    return self.json(404, {"error": "Not found"})
                return self.send(200, path.read_bytes(), "image/png" if path.suffix == ".png" else "image/jpeg")
            if url.path not in ("/api/next", "/api/previous"):
                return self.json(404, {"error": "Not found"})
            previous = url.path == "/api/previous"
            q = parse_qs(url.query)
            reviewer = q.get("reviewer", [""])[0]
            stage = q.get("stage", ["answer"])[0]
            if reviewer not in sample_data["reviewers"] or stage not in ("answer", "trace"):
                return self.json(400, {"error": "Unknown reviewer or stage"})
            assigned = [i for i in sample_data["ids"] if reviewer in assignments[i]]
            outputs = load_model_outputs(args.provider, args.model) if stage == "trace" else {}
            if stage == "trace" and not outputs:
                return self.json(200, {"error": "No full-image model output yet. Run the model first."})
            with db_connect() as conn:
                table = "answers" if stage == "answer" else "traces"
                completed = {r[0] for r in conn.execute(f"SELECT item_id FROM {table} WHERE reviewer=?", (reviewer,))}
                answered = {r[0]: r[1] for r in conn.execute("SELECT item_id,answer FROM answers WHERE reviewer=?", (reviewer,))}
                if stage == "trace":
                    if not set(assigned) <= set(answered):
                        return self.json(200, {"error": "Trace review unlocks after all of your independent answers are submitted."})
                    conn.execute("INSERT OR IGNORE INTO trace_access VALUES (?,?)", (reviewer, time.time()))
                if previous:
                    if stage == "answer" and (conn.execute("SELECT 1 FROM trace_access WHERE reviewer=?", (reviewer,)).fetchone()
                                              or conn.execute("SELECT 1 FROM traces WHERE reviewer=?", (reviewer,)).fetchone()):
                        return self.json(200, {"error": "Independent answers are locked after Trace review begins."})
                    before = q.get("before", [""])[0]
                    if before and before not in assigned:
                        return self.json(400, {"error": "Unknown current question"})
                    earlier = assigned[:assigned.index(before)] if before else assigned
                    pending = next((i for i in reversed(earlier) if i in completed), None)
                    if pending is None:
                        return self.json(200, {"error": "No earlier saved review."})
                    columns = (("answer", "confidence", "explanation") if stage == "answer" else
                               ("first_bad_claim", "claim_status", "failure_type", "notes"))
                    saved = conn.execute(f"SELECT {','.join(columns)} FROM {table} WHERE reviewer=? AND item_id=?",
                                         (reviewer, pending)).fetchone()
                    result = {"item": items[pending], "completed": len(completed), "total": len(assigned),
                              "saved": dict(zip(columns, saved))}
                    if stage == "trace":
                        result["human_answer"] = answered[pending]
                        result["gold_answer"] = keys[pending]
                        result["model"] = {k: outputs[pending].get(k) for k in ("parsed_answer", "raw_output", "model")}
                    return self.json(200, result)
            eligible = [i for i in assigned if stage == "answer" or (i in answered and i in outputs)]
            pending = next((i for i in eligible if i not in completed), None)
            if pending is None:
                return self.json(200, {"done": True, "completed": len(completed), "total": len(eligible)})
            result = {"item": items[pending], "completed": len(completed), "total": len(eligible)}
            if stage == "trace":
                result["human_answer"] = answered[pending]
                result["gold_answer"] = keys[pending]
                result["model"] = {k: outputs[pending].get(k) for k in ("parsed_answer", "raw_output", "model")}
            return self.json(200, result)

        def do_POST(self):
            stage = self.path.removeprefix("/api/")
            if stage not in ("answer", "trace"):
                return self.json(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 100000:
                    raise ValueError("Too large")
                payload = json.loads(self.rfile.read(length))
                reviewer, item_id = payload.get("reviewer"), str(payload.get("item_id"))
                if reviewer not in sample_data["reviewers"] or reviewer not in assignments.get(item_id, []):
                    raise ValueError("Reviewer is not assigned this item")
                evidence_images = ",".join(str(n) for n in range(1, len(items[item_id]["images"]) + 1))
                with db_connect() as conn:
                    if stage == "answer":
                        if (conn.execute("SELECT 1 FROM trace_access WHERE reviewer=?", (reviewer,)).fetchone()
                                or conn.execute("SELECT 1 FROM traces WHERE reviewer=?", (reviewer,)).fetchone()):
                            raise ValueError("Independent answers are locked after Trace review begins")
                        answer = str(payload.get("answer", "")).upper()
                        if answer not in "ABCD" or len(answer) != 1:
                            raise ValueError("Select A, B, C, or D")
                        conf = int(payload.get("confidence", 0))
                        if conf not in range(1, 6):
                            raise ValueError("Confidence must be 1–5")
                        conn.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(reviewer,item_id) DO UPDATE SET "
                                     "answer=excluded.answer, confidence=excluded.confidence, "
                                     "seconds=answers.seconds+excluded.seconds, evidence_images=excluded.evidence_images, "
                                     "explanation=excluded.explanation",
                            (reviewer, item_id, answer, conf, float(payload.get("seconds", 0)),
                             evidence_images, str(payload.get("explanation", "")), time.time()))
                    else:
                        if not conn.execute("SELECT 1 FROM answers WHERE reviewer=? AND item_id=?", (reviewer, item_id)).fetchone():
                            raise ValueError("Independent answer required first")
                        if item_id not in load_model_outputs(args.provider, args.model):
                            raise ValueError("Model output missing")
                        assigned = {i for i, reviewers in assignments.items() if reviewer in reviewers}
                        answered = {i for (i,) in conn.execute("SELECT item_id FROM answers WHERE reviewer=?", (reviewer,))}
                        if not assigned <= answered:
                            raise ValueError("All of your independent answers must be submitted first")
                        status = payload.get("claim_status")
                        failure = payload.get("failure_type")
                        if status not in ("correct", "incorrect", "unsupported", "none", "unclear") or failure not in FAILURES:
                            raise ValueError("Invalid trace label")
                        conn.execute("INSERT OR IGNORE INTO trace_access VALUES (?,?)", (reviewer, time.time()))
                        conn.execute("INSERT INTO traces VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(reviewer,item_id) DO UPDATE SET "
                                     "first_bad_claim=excluded.first_bad_claim, claim_status=excluded.claim_status, "
                                     "failure_type=excluded.failure_type, evidence_images=excluded.evidence_images, "
                                     "notes=excluded.notes",
                            (reviewer, item_id, str(payload.get("first_bad_claim", "")), status, failure,
                             evidence_images, str(payload.get("notes", "")), time.time()))
            except (ValueError, TypeError, sqlite3.IntegrityError) as exc:
                return self.json(400, {"error": str(exc)})
            return self.json(200, {"saved": True})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Review UI: http://{args.host}:{args.port}/?reviewer={sample_data['reviewers'][0]}")
    server.serve_forever()


def report(args):
    if args.provider == "litellm" and not args.model:
        args.model = local_secret("MMML_LITELLM_MODEL")
    if not args.model:
        raise SystemExit("Set --model (or MMML_LITELLM_MODEL for LiteLLM)")
    sample_data = json.loads((DATA / "sample.json").read_text(encoding="utf-8"))
    ids = sample_data["ids"]
    items = {x["id"]: x for x in read_jsonl(DATA / "items.jsonl")}
    keys = {x["id"]: x for x in read_jsonl(DATA / "keys.jsonl")}
    with db_connect() as conn:
        human = defaultdict(list)
        for reviewer, item_id, answer in conn.execute("SELECT reviewer,item_id,answer FROM answers"):
            human[item_id].append((reviewer, answer))
        traces = defaultdict(list)
        for reviewer, item_id, status, failure in conn.execute("SELECT reviewer,item_id,claim_status,failure_type FROM traces"):
            traces[item_id].append((reviewer, status, failure))
    model = load_model_outputs(args.provider, args.model)
    set_by_id = {item_id: name for name, members in sample_data["sets"].items() for item_id in members}
    rows = []
    for item_id in ids:
        key = str(keys[item_id]["answer"]).strip().upper()
        if key not in "ABCD" or len(key) != 1:
            raise ValueError(f"Unexpected answer key for {item_id}: {key}")
        people = sorted(human[item_id])
        row = {"id": item_id, "set": set_by_id[item_id], "question_type": items[item_id]["question_type"],
               "difficulty": items[item_id]["difficulty"], "gold": key,
               "human_n": len(people), "human_answers": ";".join(f"{r}:{a}" for r, a in people),
               "human_correct_n": sum(a == key for _, a in people),
               "human_agree": len(people) == 2 and people[0][1] == people[1][1],
               "model_images": model.get(item_id, {}).get("parsed_answer"),
               "trace_labels": ";".join(f"{r}:{s}:{f}" for r, s, f in traces[item_id])}
        rows.append(row)
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / "items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    disagreements = [r for r in rows if r["human_n"] == 2 and not r["human_agree"]]
    with (REPORTS / "disagreements.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question_type", "question", "human_answers", "gold", "image_paths"])
        w.writeheader()
        for r in disagreements:
            item = items[r["id"]]
            w.writerow({"id": r["id"], "question_type": r["question_type"],
                        "question": item["question"], "human_answers": r["human_answers"],
                        "gold": r["gold"], "image_paths": ";".join(item["images"])})
    summary = {"sample_n": len(rows), "complete_human_pairs": sum(r["human_n"] == 2 for r in rows),
               "human_agreement_on_pairs": None, "human_disagreements_n": len(disagreements),
               "overall": {}, "categories": {}}
    pairs = [r for r in rows if r["human_n"] == 2]
    if pairs:
        summary["human_agreement_on_pairs"] = sum(r["human_agree"] for r in pairs) / len(pairs)
    all_judgments = [(r, a) for r in rows for _, a in human[r["id"]]]
    summary["overall"]["human_judgments_n"] = len(all_judgments)
    summary["overall"]["human_accuracy"] = (sum(a == r["gold"] for r, a in all_judgments) / len(all_judgments)
                                               if all_judgments else None)
    scored = [r for r in rows if r["model_images"] is not None]
    summary["overall"]["model_images_n"] = len(scored)
    summary["overall"]["model_images_accuracy"] = (
        sum(r["model_images"] == r["gold"] for r in scored) / len(scored) if scored else None)
    summary["overall"]["both_humans_correct_model_wrong_n"] = sum(
        r["human_correct_n"] == 2 and r["model_images"] is not None and r["model_images"] != r["gold"] for r in rows)
    for cat in sorted({r["question_type"] for r in rows}):
        subset = [r for r in rows if r["question_type"] == cat]
        stats = {"n": len(subset)}
        scored = [r for r in subset if r["model_images"] is not None]
        stats["model_images_n"] = len(scored)
        stats["model_images_accuracy"] = sum(r["model_images"] == r["gold"] for r in scored) / len(scored) if scored else None
        judgments = [(r, a) for r in subset for _, a in human[r["id"]]]
        stats["human_judgments_n"] = len(judgments)
        stats["human_accuracy"] = sum(a == r["gold"] for r, a in judgments) / len(judgments) if judgments else None
        stats["human_correct_model_wrong_n"] = sum(
            r["human_correct_n"] == 2 and r["model_images"] is not None and r["model_images"] != r["gold"]
            for r in subset)
        summary["categories"][cat] = stats
    summary["sets"] = {name: {"n": len(members), "completed_human_pairs": sum(
        r["human_n"] == 2 for r in rows if r["set"] == name), "model_images_n": sum(
        r["model_images"] is not None for r in rows if r["set"] == name)}
        for name, members in sample_data["sets"].items()}
    summary["difficulties"] = {difficulty: {"n": sum(r["difficulty"] == difficulty for r in rows),
        "model_images_n": sum(r["difficulty"] == difficulty and r["model_images"] is not None for r in rows)}
        for difficulty in DIFFICULTIES}
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("download"); x.set_defaults(func=download)
    x = sub.add_parser("prepare"); x.add_argument("--force", action="store_true"); x.add_argument("--sample-only", action="store_true"); x.set_defaults(func=prepare)
    x = sub.add_parser("sample"); x.add_argument("--seed", type=int, default=777); x.add_argument("--reviewers", nargs=4, default=["R1", "R2", "R3", "R4"]); x.add_argument("--force", action="store_true"); x.set_defaults(func=sample)
    x = sub.add_parser("gateway-models"); x.add_argument("--base-url"); x.set_defaults(func=gateway_models)
    x = sub.add_parser("model"); x.add_argument("--provider", choices=["openai", "gemini", "litellm", "compatible"], required=True); x.add_argument("--model"); x.add_argument("--base-url"); x.add_argument("--condition", choices=["images"], default="images"); x.add_argument("--temperature", type=float); x.add_argument("--max-tokens", type=int, default=4096); x.add_argument("--timeout", type=int, default=180); x.add_argument("--retries", type=int, default=3); x.add_argument("--delay", type=float, default=0); x.add_argument("--limit", type=int); x.add_argument("--force", action="store_true"); x.set_defaults(func=model_run)
    x = sub.add_parser("serve"); x.add_argument("--host", default="127.0.0.1"); x.add_argument("--port", type=int, default=8766); x.add_argument("--provider", choices=["openai", "gemini", "litellm", "compatible"]); x.add_argument("--model"); x.set_defaults(func=serve)
    x = sub.add_parser("export-review"); x.add_argument("--reviewer", required=True); x.add_argument("--out", required=True); x.add_argument("--force", action="store_true"); x.set_defaults(func=export_review)
    x = sub.add_parser("import-review"); x.add_argument("--path", required=True); x.set_defaults(func=import_review)
    x = sub.add_parser("report"); x.add_argument("--provider", choices=["openai", "gemini", "litellm", "compatible"], required=True); x.add_argument("--model"); x.set_defaults(func=report)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
