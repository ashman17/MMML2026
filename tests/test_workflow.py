import contextlib
import json
import math
import os
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import workflow


class MockHandler(BaseHTTPRequestHandler):
    received = []

    def do_GET(self):
        data = json.dumps({"data": [{"id": "vision-model"}, {"id": "text-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.received.append((self.path, body))
        if self.path.endswith("/responses"):
            result = {"id": "openai-test", "output": [{"content": [{"type": "output_text", "text": "Evidence. Final answer: B"}]}]}
        elif ":generateContent" in self.path:
            result = {"responseId": "gemini-test", "candidates": [{"content": {"parts": [{"text": "Evidence. Final answer: B"}]}}]}
        elif "Reasoning trace:" in json.dumps(body):
            result = {"id": "judge-test", "choices": [{"message": {"content": "The trace says left; geometry says right.\nInconsistent: True"}}]}
        else:
            result = {"id": "qwen-test", "choices": [{"message": {"content": "Evidence. Final answer: B"}}]}
        data = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class WorkflowTests(unittest.TestCase):
    def test_answer_parser_does_not_guess_from_rationale(self):
        self.assertEqual(workflow.parse_answer("A is tempting.\nFinal answer: C"), "C")
        self.assertIsNone(workflow.parse_answer("A is tempting, but I am uncertain."))

    def test_provider_payloads_and_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, runs = Path(tmp) / "data", Path(tmp) / "runs"
            (data / "images").mkdir(parents=True)
            (data / "images/1_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            workflow.write_jsonl(data / "items.jsonl", [{"id": "1", "question": "Which? Options: A: no, B: yes, C: no, D: no", "images": ["images/1_0.jpg"]}])
            (data / "sample.json").write_text(json.dumps({"ids": ["1"]}))
            (data / "provenance.json").write_text(json.dumps({"sha256": "test"}))
            server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.object(workflow, "DATA", data), patch.object(workflow, "RUNS", runs), patch.dict(os.environ, {"MMML_OPENAI_API_KEY": "test", "MMML_GEMINI_API_KEY": "test", "MMML_LITELLM_API_KEY": "test"}):
                    base = f"http://127.0.0.1:{server.server_port}/v1"
                    workflow.gateway_models(Namespace(base_url=base))
                    for provider in ("openai", "gemini", "litellm", "compatible"):
                        for condition in ("images",):
                            args = Namespace(provider=provider, model="test-model", base_url=base,
                                condition=condition, ids="sample", temperature=None, max_tokens=128, timeout=5,
                                retries=0, delay=0, limit=None, force=False)
                            workflow.model_run(args)
                            record = workflow.read_jsonl(workflow.run_path(provider, "test-model", condition))[0]
                            self.assertEqual(record["parsed_answer"], "B")
                            self.assertEqual(record["prompt_version"], workflow.PROMPT_VERSION)
                            self.assertIn("using the supplied images", record["prompt"])
                            sent = MockHandler.received[-1][1]
                            if provider == "gemini":
                                content = sent["contents"][0]["parts"]
                                image_count = sum("inline_data" in x for x in content)
                            elif provider == "openai":
                                content = sent["input"][0]["content"]
                                image_count = sum(x["type"] == "input_image" for x in content)
                            else:
                                content = sent["messages"][0]["content"]
                                image_count = sum(x["type"] == "image_url" for x in content)
                                if provider == "litellm":
                                    self.assertNotIn("temperature", sent)
                            self.assertEqual(image_count, 1)
                            self.assertTrue(any(x.get("text") == "Image 1:" for x in content))
            finally:
                server.shutdown()
                server.server_close()

    def test_camera_hints_use_image1_frame(self):
        # Camera 2 sits 1 right and 2 forward of camera 1, looking along camera 1's +x (turned right 90°).
        R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        t = [2, 0, -1]
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        line = workflow.camera_hint_lines([identity, [R[k] + [t[k]] for k in range(3)]])[0]
        self.assertIn("1.00 right", line)
        self.assertIn("2.00 forward", line)
        self.assertIn("turned 90° right", line)
        self.assertIn("tilted 0° up", line)
        # Camera looking slightly up (forward has negative y in OpenCV) and left.
        pitch = math.radians(20)
        R_up = [[1, 0, 0], [0, math.cos(pitch), math.sin(pitch)], [0, -math.sin(pitch), math.cos(pitch)]]
        line = workflow.camera_hint_lines([identity, [row + [0] for row in R_up]])[0]
        self.assertIn("tilted 20° up", line)

    def test_vggt_condition_adds_hints_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, runs, reports = (Path(tmp) / p for p in ("data", "runs", "reports"))
            (data / "images").mkdir(parents=True)
            (data / "vggt").mkdir()
            for n in range(2):
                (data / f"images/1_{n}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            workflow.write_jsonl(data / "items.jsonl", [{"id": "1", "question_type": "cat", "question": "Which?",
                                                        "images": ["images/1_0.jpg", "images/1_1.jpg"]}])
            workflow.write_jsonl(data / "keys.jsonl", [{"id": "1", "answer": "B", "thought": ""}])
            (data / "provenance.json").write_text(json.dumps({"sha256": "test"}))
            identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
            (data / "vggt/1.json").write_text(json.dumps({"extrinsic": [identity, [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]]]}))
            server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                with patch.object(workflow, "DATA", data), patch.object(workflow, "RUNS", runs), patch.object(workflow, "REPORTS", reports):
                    for condition in workflow.CONDITIONS:
                        workflow.model_run(Namespace(provider="compatible", model="test-model",
                            base_url=f"http://127.0.0.1:{server.server_port}/v1", condition=condition, ids="all",
                            temperature=None, max_tokens=128, timeout=5, retries=0, delay=0, limit=None, force=False))
                    record = workflow.read_jsonl(workflow.run_path("compatible", "test-model", "images_vggt"))[0]
                    self.assertIn("Image 2 camera relative to Image 1 camera: moved 1.00 right", record["prompt"])
                    self.assertEqual(record["prompt_version"], workflow.prompt_version("images_vggt"))
                    baseline = workflow.read_jsonl(workflow.run_path("compatible", "test-model", "images"))[0]
                    self.assertNotIn("camera geometry", baseline["prompt"])
                    workflow.score(Namespace(provider="compatible", model="test-model", ids="all",
                                             conditions=list(workflow.CONDITIONS)))
                    summary = json.loads((reports / "score_test-model_all.json").read_text())
                    self.assertEqual(summary["accuracy"]["overall"]["images_vggt"], 1)
                    self.assertEqual(summary["accuracy"]["cat"]["n"], 1)
            finally:
                server.shutdown()
                server.server_close()

    def test_vggt_writes_batches_and_hints_read_them(self):
        class FakeTensor:
            def __init__(self, data):
                self.data = data
                self.shape = (len(data), 3, 518, 518)
            def __getitem__(self, i):
                return FakeTensor(self.data[i])
            def to(self, *args):
                return self
            float = cpu = to
            def flatten(self, dim):
                return self
            def median(self, dim):
                return Namespace(values=FakeTensor([0.5] * len(self.data)))
            def tolist(self):
                return self.data

        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        fake_model = Namespace(to=lambda device: Namespace(eval=lambda: lambda images: {
            "pose_enc": images, "depth_conf": FakeTensor([[1] * len(images.data)])}))
        modules = {
            "torch": Namespace(cuda=Namespace(is_available=lambda: False), no_grad=contextlib.nullcontext,
                               autocast=lambda **kw: contextlib.nullcontext(), bfloat16="bf16", float16="fp16"),
            "vggt": Namespace(), "vggt.models": Namespace(), "vggt.utils": Namespace(),
            "vggt.models.vggt": Namespace(VGGT=Namespace(from_pretrained=lambda name: fake_model)),
            "vggt.utils.load_fn": Namespace(load_and_preprocess_images=lambda paths: FakeTensor([0] * len(paths))),
            "vggt.utils.pose_enc": Namespace(pose_encoding_to_extri_intri=lambda enc, hw: (
                FakeTensor([[identity] * len(enc.data)]), FakeTensor([[identity] * len(enc.data)]))),
        }
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            workflow.write_jsonl(data / "items.jsonl", [{"id": str(i), "images": [f"images/{i}_0.jpg", f"images/{i}_1.jpg"]}
                                                        for i in range(3)])
            with patch.object(workflow, "DATA", data), patch.dict(sys.modules, modules):
                args = Namespace(ids="all", model="fake", limit=None, batch_size=2, force=False)
                workflow.vggt_run(args)
                bundles = sorted(p.name for p in (data / "vggt/batches").glob("*.jsonl"))
                self.assertEqual(bundles, ["all_batch_001.jsonl", "all_batch_002.jsonl"])
                self.assertEqual([r["id"] for r in workflow.read_jsonl(data / "vggt/batches/all_batch_001.jsonl")], ["0", "1"])
                # A downloaded bundle alone is enough to build hints.
                for p in (data / "vggt").glob("*.json"):
                    p.unlink()
                self.assertIn("Image 2 camera relative to Image 1 camera: moved 0.00 right", workflow.vggt_hint_block("2"))
                workflow.vggt_run(args)  # completed batches are skipped, not recomputed
                self.assertFalse(list((data / "vggt").glob("*.json")))

    def test_verdict_parser(self):
        self.assertTrue(workflow.parse_verdict("Turned left vs right.\nInconsistent: True"))
        self.assertFalse(workflow.parse_verdict("Consistent.\n**Inconsistent:** False"))
        self.assertFalse(workflow.parse_verdict("Inconsistent: True? No.\nInconsistent: false"))
        self.assertIsNone(workflow.parse_verdict("I cannot tell."))

    def test_judge_sees_only_trace_and_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, runs, reports = (Path(tmp) / p for p in ("data", "runs", "reports"))
            (data / "vggt").mkdir(parents=True)
            workflow.write_jsonl(data / "items.jsonl", [
                {"id": "1", "question_type": "cat", "question": "SECRET QUESTION? Options: A: x, B: y", "images": ["images/1_0.jpg", "images/1_1.jpg"]},
                {"id": "2", "question_type": "cat", "question": "No VGGT for this one", "images": ["images/2_0.jpg"]}])
            workflow.write_jsonl(data / "keys.jsonl", [{"id": "1", "answer": "B", "thought": ""}, {"id": "2", "answer": "A", "thought": ""}])
            identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
            (data / "vggt/1.json").write_text(json.dumps({"extrinsic": [identity, [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]]]}))
            server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                with patch.object(workflow, "DATA", data), patch.object(workflow, "RUNS", runs), patch.object(workflow, "REPORTS", reports):
                    workflow.write_jsonl(workflow.run_path("compatible", "test-model", "images"), [
                        {"id": i, "prompt_version": workflow.PROMPT_VERSION, "error": None, "parsed_answer": "B",
                         "raw_output": "The camera turned left. Final answer: B"} for i in ("1", "2")])
                    workflow.judge(Namespace(provider="compatible", model="test-model", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                             source_provider=None, source_model=None, source_condition="images", ids="all",
                                             max_tokens=64, timeout=5, retries=0, limit=None, force=False))
                    records = workflow.read_jsonl(workflow.run_path("compatible", "test-model", "judge_images"))
                    self.assertEqual([r["id"] for r in records], ["1"])  # item without VGGT output is skipped
                    self.assertTrue(records[0]["inconsistent"])
                    prompt = MockHandler.received[-1][1]["messages"][0]["content"]
                    self.assertIn("The camera turned left.", prompt)
                    self.assertIn("moved 1.00 right", prompt)
                    self.assertNotIn("SECRET QUESTION", prompt)
                    self.assertNotIn("image_url", json.dumps(MockHandler.received[-1][1]))
                    summary = json.loads((reports / "judge_test-model_images_all.json").read_text())
                    self.assertEqual(summary["by_category"]["overall"]["inconsistent_rate"], 1)
                    self.assertEqual(summary["by_category"]["overall"]["accuracy_when_inconsistent"], 1)
                    self.assertEqual(summary["by_category"]["overall"]["confusion"],
                                     {"inconsistent": {"correct": 1, "wrong": 0}, "consistent": {"correct": 0, "wrong": 0},
                                      "unparsed": {"correct": 0, "wrong": 0}})
                    self.assertEqual(summary["flag_as_error_detector"], {"precision": 0.0, "recall": None})
            finally:
                server.shutdown()
                server.server_close()

    def test_report_with_humans_before_model_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, runs, reports = (Path(tmp) / p for p in ("data", "runs", "reports"))
            data.mkdir()
            workflow.write_jsonl(data / "items.jsonl", [{"id": "1", "question_type": "test", "difficulty": "easy", "question": "Which?", "images": []}])
            workflow.write_jsonl(data / "keys.jsonl", [{"id": "1", "answer": "A", "thought": ""}])
            (data / "sample.json").write_text(json.dumps({"ids": ["1"], "sets": {"A": ["1"], "B": []}}))
            with patch.object(workflow, "DATA", data), patch.object(workflow, "RUNS", runs), patch.object(workflow, "REPORTS", reports):
                with workflow.db_connect() as conn:
                    conn.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?)", ("R1", "1", "A", 3, 10, "", "", 0))
                    conn.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?)", ("R2", "1", "A", 4, 11, "", "", 0))
                workflow.report(Namespace(provider="compatible", model="test-model"))
                summary = json.loads((reports / "summary.json").read_text())
                self.assertEqual(summary["overall"]["human_accuracy"], 1)
                self.assertEqual(summary["categories"]["test"]["human_correct_model_wrong_n"], 0)
                self.assertEqual(summary["sets"]["A"]["n"], 1)

    def test_sample_balances_difficulty_and_reviewer_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            items = [{"id": f"{cat}-{difficulty}-{n}", "question_type": cat,
                      "difficulty": difficulty} for cat in ("first", "second")
                     for difficulty in workflow.DIFFICULTIES for n in range(4)]
            workflow.write_jsonl(data / "items.jsonl", items)
            with patch.object(workflow, "DATA", data), patch.object(workflow, "ROOT", root):
                workflow.sample(Namespace(seed=777, reviewers=["R1", "R2", "R3", "R4"], force=False))
                sample = json.loads((data / "sample.json").read_text())
                self.assertEqual((root / "protocol/sample.json").read_bytes(), (data / "sample.json").read_bytes())
                self.assertEqual(len(sample["ids"]), 12)
                self.assertEqual(len(set(sample["ids"])), 12)
                lookup = {item["id"]: item for item in items}
                for set_name, reviewers in (("A", ["R1", "R2"]), ("B", ["R3", "R4"])):
                    self.assertEqual(len(sample["sets"][set_name]), 6)
                    for cat in ("first", "second"):
                        for difficulty in workflow.DIFFICULTIES:
                            matches = [item_id for item_id in sample["sets"][set_name]
                                       if lookup[item_id]["question_type"] == cat and lookup[item_id]["difficulty"] == difficulty]
                            self.assertEqual(len(matches), 1)
                            self.assertEqual(sample["assignments"][matches[0]], reviewers)

    def test_review_export_import_matches_fixed_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            sample = {"reviewers": ["R1", "R2", "R3", "R4"], "assignments": {"1": ["R1", "R2"]}, "ids": ["1"]}
            (data / "sample.json").write_text(json.dumps(sample))
            out = Path(tmp) / "R1.json"
            with patch.object(workflow, "DATA", data):
                with workflow.db_connect() as conn:
                    conn.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?)", ("R1", "1", "B", 3, 10, "1", "note", 0))
                workflow.export_review(Namespace(reviewer="R1", out=str(out), force=False))
                with workflow.db_connect() as conn:
                    conn.execute("DELETE FROM answers")
                workflow.import_review(Namespace(path=str(out)))
                workflow.import_review(Namespace(path=str(out)))
                with workflow.db_connect() as conn:
                    self.assertEqual(conn.execute("SELECT answer FROM answers WHERE reviewer='R1'").fetchone()[0], "B")


if __name__ == "__main__":
    unittest.main()
