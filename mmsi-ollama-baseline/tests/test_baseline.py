from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline import (
    OFFICIAL_POST_PROMPT,
    build_claim_verification_request,
    build_inspection_request,
    build_request,
    extract_official_choice,
    extract_inspection_answer,
    extract_numbered_claims,
    official_prompt,
    prompt_with_image_placeholders,
    select_records,
    summarize,
)
from run import completed_keys, latest_rows
from chat_example import image_manifest, rebuild_messages, request_payload


class OfficialCompatibilityTest(unittest.TestCase):
    def test_prompt_matches_official_form(self) -> None:
        self.assertEqual(official_prompt("Q"), f"Q\n{OFFICIAL_POST_PROMPT}")

    def test_official_answer_extraction(self) -> None:
        self.assertEqual(extract_official_choice("``C``"), "C")
        self.assertEqual(extract_official_choice("The answer is B."), "B")
        self.assertIsNone(extract_official_choice("A bicycle"))
        self.assertIsNone(extract_official_choice(None))

    def test_request_matches_paper_generation_settings(self) -> None:
        request = build_request("m", "p", ["image"])
        self.assertEqual(request["options"]["temperature"], 0)
        self.assertEqual(request["options"]["num_predict"], 2048)
        self.assertEqual(request["options"]["num_ctx"], 8192)
        self.assertNotIn("format", request)
        self.assertTrue(request["messages"][0]["content"].startswith("IMAGE 1:\n[img]\n\np"))

    def test_inspection_defaults_to_official_prompt_with_transport_wrapper(self) -> None:
        request = build_inspection_request("m", "question only", ["a", "b"])
        content = request["messages"][0]["content"]
        self.assertTrue(content.endswith(official_prompt("question only")))
        self.assertTrue(content.startswith("IMAGE 1:\n[img]\n\nIMAGE 2:\n[img]"))
        self.assertNotIn("indexed 0 through 1", content)
        self.assertFalse(request["think"])
        self.assertTrue(request["stream"])
        self.assertNotIn("format", request)
        self.assertEqual(request["options"]["num_predict"], 2048)

    def test_natural_inspection_is_explicitly_non_official(self) -> None:
        request = build_inspection_request(
            "m", "question only", ["a", "b"], rationale=True, temperature=0.6
        )
        content = request["messages"][0]["content"]
        self.assertIn("first supplied image is Image 1", content)
        self.assertIn("exactly 2 separate images", content)
        self.assertIn("inspect and report on every image", content)
        self.assertNotIn("indexed 0 through", content)
        self.assertNotIn("reference rationale", content.lower())
        self.assertIn("FINAL ANSWER: ``X``", content)
        image_observations = content.index("IMAGE-BY-IMAGE OBSERVATIONS")
        common_aspects = content.index("COMMON ASPECTS")
        inferences = content.index("INFERENCES")
        self.assertLess(image_observations, common_aspects)
        self.assertLess(common_aspects, inferences)
        self.assertIn("finishing it before examining Image 2", content)
        self.assertIn("Do not make spatial inferences in these sections", content)
        self.assertEqual(request["options"]["temperature"], 0.6)
        self.assertEqual(request["options"]["top_p"], 0.95)
        self.assertEqual(request["options"]["top_k"], 20)

    def test_same_size_image_workaround_separates_placeholders(self) -> None:
        prompt = prompt_with_image_placeholders("question", 2)
        self.assertEqual(prompt.count("[img]"), 2)
        self.assertIn("IMAGE 1:\n[img]\n\nIMAGE 2:\n[img]", prompt)

    def test_claim_audit_is_separate_from_solver(self) -> None:
        solver = build_inspection_request("m", "q", ["a"], rationale=True)
        self.assertNotIn("gold", solver["messages"][0]["content"].lower())
        audit = build_claim_verification_request("m", "q", ["a"], "claim", "D", "reference")
        content = audit["messages"][0]["content"]
        self.assertIn("CORRECT ANSWER: D", content)
        self.assertIn("HUMAN REFERENCE REASONING", content)
        self.assertIn("exactly one verdict for every numbered claim", content)
        self.assertIn("claims", audit["format"]["properties"])

    def test_numbered_claim_extraction(self) -> None:
        output = "1. [Observation | Image 1] toilet visible\n2. [Inference] behind viewer\nFINAL ANSWER: D"
        self.assertEqual(
            extract_numbered_claims(output),
            [
                {"claim_number": 1, "text": "[Observation | Image 1] toilet visible"},
                {"claim_number": 2, "text": "[Inference] behind viewer"},
            ],
        )

    def test_chat_rebuild_retains_images_and_feedback_history(self) -> None:
        turns = [{
            "assistant": "1. initial claim\nFINAL ANSWER: A",
            "feedback": "Claim 1 is wrong.",
        }]
        messages = rebuild_messages("q", ["image-1", "image-2"], turns)
        self.assertEqual(messages[1]["images"], ["image-1", "image-2"])
        self.assertIn("IMAGE 1:\n[img]\n\nIMAGE 2:\n[img]", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["content"], "Claim 1 is wrong.")
        payload = request_payload("m", messages, num_ctx=16384, temperature=0.6)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["options"]["num_ctx"], 16384)
        self.assertEqual(payload["options"]["temperature"], 0.6)
        self.assertEqual(payload["options"]["top_p"], 0.95)
        self.assertEqual(payload["options"]["top_k"], 20)

    def test_image_manifest_preserves_order(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one").write_bytes(b"first")
            (root / "two").write_bytes(b"second")
            manifest = image_manifest(root, ["one", "two"])
            self.assertEqual([item["image_number"] for item in manifest], [1, 2])
            self.assertEqual([item["bytes"] for item in manifest], [5, 6])

    def test_natural_inspection_answer_extraction_uses_final_marker(self) -> None:
        text = "Option A seems plausible, but image 1 reverses it.\nFINAL ANSWER: **C**"
        self.assertEqual(extract_inspection_answer(text), "C")
        self.assertIsNone(extract_inspection_answer("I considered A and C."))

    def test_structured_inspection_remains_optional(self) -> None:
        request = build_inspection_request("m", "q", ["a"], structured=True)
        self.assertIn("cross_view_inferences", request["format"]["properties"])

    def test_stratified_selection_is_reproducible(self) -> None:
        records = [
            {"id": i, "category": f"c{i % 2}", "difficulty": "easy"}
            for i in range(12)
        ]
        kwargs = dict(limit=5, categories=None, difficulties=None, ids=None, sampling="stratified", seed=7)
        self.assertEqual(
            [x["id"] for x in select_records(records, **kwargs)],
            [x["id"] for x in select_records(records, **kwargs)],
        )

    def test_small_stratified_sample_is_not_alphabetically_biased(self) -> None:
        records = [
            {"id": i, "category": category, "difficulty": difficulty}
            for i, (category, difficulty) in enumerate(
                (category, difficulty)
                for category in ("a", "b", "c", "d")
                for difficulty in ("easy", "hard", "medium")
            )
        ]
        sample = select_records(
            records, limit=4, categories=None, difficulties=None, ids=None,
            sampling="stratified", seed=7,
        )
        self.assertGreater(len({item["category"] for item in sample}), 1)

    def test_summary(self) -> None:
        rows = [{
            "category": "c", "difficulty": "easy", "correct": i == 0,
            "extracted_answer": "A", "wall_seconds": 1.0,
            "ollama_metrics": {"eval_count": 2},
        } for i in range(2)]
        self.assertEqual(summarize(rows)["overall"]["accuracy"], 0.5)

    def test_failed_rows_are_retried_and_latest_attempt_wins(self) -> None:
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            base = {"id": 1, "model": {"digest": "d"}, "protocol_version": "p"}
            failed = {**base, "error": "failed", "correct": False}
            success = {**base, "error": None, "correct": True}
            path.write_text(json.dumps(failed) + "\n", encoding="utf-8")
            self.assertFalse(completed_keys(path))
            path.write_text(json.dumps(failed) + "\n" + json.dumps(success) + "\n", encoding="utf-8")
            self.assertEqual(len(latest_rows(path)), 1)
            self.assertTrue(latest_rows(path)[0]["correct"])


if __name__ == "__main__":
    unittest.main()
