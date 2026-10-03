import unittest

from fvpg import (
    build_ollama_request,
    parse_structured_response,
    stratified_sample,
    validate_proof_payload,
)


class VLMHarnessTest(unittest.TestCase):
    def test_direct_request_does_not_include_reference_rationale(self) -> None:
        request = build_ollama_request("model", "Question Options: A: x, B: y", ["image"], "direct")
        self.assertEqual(request["messages"][1]["images"], ["image"])
        self.assertNotIn("thought", request["messages"][1]["content"].lower())
        self.assertEqual(request["format"]["properties"]["answer"]["enum"], ["A", "B", "C", "D"])

    def test_proof_claim_status_is_forced_unknown(self) -> None:
        request = build_ollama_request("model", "Question", ["image"], "proof")
        status = request["format"]["properties"]["proofs"]["items"]["properties"]["claims"]["items"]["properties"]["status"]
        self.assertEqual(status["enum"], ["unknown"])
        self.assertIn("never image/view IDs", request["messages"][1]["content"])

    def test_parse_answer(self) -> None:
        self.assertEqual(parse_structured_response('{"answer":"B","confidence":0.5}')["answer"], "B")

    def test_stratified_sample_is_deterministic(self) -> None:
        records = [
            {"id": index, "category": "c", "difficulty": "easy", "images": ["x"]}
            for index in range(10)
        ]
        first = [r["id"] for r in stratified_sample(records, 4)]
        second = [r["id"] for r in stratified_sample(records, 4)]
        self.assertEqual(first, second)

    def test_proof_semantics_reject_duplicate_options_and_bad_views(self) -> None:
        proof = {
            "option": "A", "conclusion": "x", "critical_claim_id": "c1",
            "claims": [{
                "id": "c1", "predicate": "VIEW_OVERLAP", "arguments": [],
                "frame": "image", "depends_on": [], "status": "unknown",
                "evidence_request": {"verifier": "correspondence", "views": [0], "test": "x"},
            }],
        }
        payload = {
            "entities": [{"name": "x", "role": "target", "views": [2]}],
            "proofs": [proof, proof, proof, proof],
        }
        errors = validate_proof_payload(payload, image_count=2)
        self.assertTrue(any("exactly once" in error for error in errors))
        self.assertTrue(any("invalid view" in error for error in errors))
        self.assertTrue(any("two distinct views" in error for error in errors))

    def test_proof_semantics_reject_wrong_verifier(self) -> None:
        payload = {
            "proofs": [{
                "option": option, "conclusion": "x", "critical_claim_id": f"c{option}",
                "claims": [{
                    "id": f"c{option}", "predicate": "DEPTH_ORDER", "arguments": [],
                    "frame": "camera", "depends_on": [], "status": "unknown",
                    "evidence_request": {
                        "verifier": "correspondence", "views": [0], "test": "x"
                    },
                }],
            } for option in "ABCD"]
        }
        errors = validate_proof_payload(payload, image_count=1)
        self.assertTrue(any("incompatible verifier" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
