import unittest

from fvpg import compile_proof_program, validate_proof_payload


def payload_with_bad_scheduling():
    return {
        "entities": [],
        "proofs": [{
            "option": option,
            "conclusion": f"camera faces {option}",
            "critical_claim_id": f"c{option}",
            "claims": [{
                "id": f"c{option}",
                "predicate": "CAMERA_ROTATION",
                "arguments": ["view0", "view1"],
                "frame": "geographic",
                "depends_on": ["view_0"],
                "evidence_request": {
                    "verifier": "geometry_checker",
                    "views": [0],
                    "test": "estimate relative rotation",
                },
                "status": "unknown",
            }],
        } for option in "ABCD"],
        "answer": "A",
        "confidence": 0.5,
    }


class ProofProgramCompilerTest(unittest.TestCase):
    def test_repairs_only_execution_metadata(self) -> None:
        raw = payload_with_bad_scheduling()
        compiled, repairs = compile_proof_program(raw, image_count=2)
        claim = compiled["proofs"][0]["claims"][0]
        self.assertEqual(claim["predicate"], "CAMERA_ROTATION")
        self.assertEqual(compiled["proofs"][0]["conclusion"], "camera faces A")
        self.assertEqual(compiled["answer"], "A")
        self.assertEqual(claim["evidence_request"]["verifier"], "camera_pose")
        self.assertEqual(claim["evidence_request"]["views"], [0, 1])
        self.assertEqual(claim["depends_on"], [])
        self.assertTrue(repairs)
        self.assertFalse(validate_proof_payload(compiled, image_count=2))

    def test_does_not_repair_semantic_structure(self) -> None:
        raw = payload_with_bad_scheduling()
        raw["proofs"][1]["option"] = "A"
        compiled, _ = compile_proof_program(raw, image_count=2)
        self.assertTrue(validate_proof_payload(compiled, image_count=2))


if __name__ == "__main__":
    unittest.main()
