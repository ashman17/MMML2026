import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvpg import (
    Claim,
    Derivation,
    Evidence,
    Frame,
    MMSIProofCompiler,
    ProofChecker,
    ProofGraph,
    Status,
)


class ProofCheckerTest(unittest.TestCase):
    def test_valid_inverse_relation(self) -> None:
        graph = ProofGraph()
        graph.add_claim(Claim("a", "left_of", ("chair", "curtain"), Frame.GLOBAL))
        graph.add_claim(Claim("b", "right_of", ("curtain", "chair"), Frame.GLOBAL))
        graph.add_evidence(Evidence(
            id="e1",
            verifier="geometry",
            status=Status.SUPPORTED,
            support_probability=0.9,
            contradiction_probability=0.05,
            unknown_probability=0.05,
            provenance={"image_ids": ["1", "2"], "model": "test"},
        ), "a")
        graph.add_derivation(Derivation("d1", "inverse_relation", ("a",), "b"))

        result = ProofChecker().check(graph)
        self.assertTrue(result.valid)
        self.assertIn("b", result.supported_claims)

    def test_frame_change_is_rejected(self) -> None:
        graph = ProofGraph()
        graph.add_claim(Claim("a", "left_of", ("chair", "curtain"), Frame.IMAGE, status=Status.SUPPORTED))
        graph.add_claim(Claim("b", "right_of", ("curtain", "chair"), Frame.GLOBAL))
        graph.add_derivation(Derivation("d1", "inverse_relation", ("a",), "b"))

        result = ProofChecker().check(graph)
        self.assertFalse(result.valid)
        self.assertTrue(any("coordinate frame" in error for error in result.errors))

    def test_position_does_not_imply_orientation(self) -> None:
        graph = ProofGraph()
        graph.add_claim(Claim("a", "behind", ("chair", "bin"), Frame.GLOBAL, status=Status.SUPPORTED))
        graph.add_claim(Claim("b", "left_of", ("bin", "curtain"), Frame.GLOBAL, status=Status.SUPPORTED))
        graph.add_claim(Claim("c", "facing", ("chair", "curtain"), Frame.GLOBAL))
        graph.add_derivation(Derivation("d1", "position_implies_orientation", ("a", "b"), "c"))

        result = ProofChecker().check(graph)
        self.assertFalse(result.valid)
        self.assertIn("c", result.unresolved_claims)
        self.assertTrue(any("unregistered rule" in error for error in result.errors))

    def test_evidence_requires_provenance(self) -> None:
        with self.assertRaises(ValueError):
            Evidence(
                id="e1",
                verifier="vlm",
                status=Status.UNKNOWN,
                support_probability=0.2,
                contradiction_probability=0.2,
                unknown_probability=0.6,
                provenance={},
            )


class MMSIProofCompilerTest(unittest.TestCase):
    def test_cardinal_options_use_geographic_frame(self) -> None:
        record = {
            "id": 352,
            "category": "Positional Relationship (Cam.–Cam.)",
            "question": (
                "Assuming that the camera was facing the west side of the room when the first "
                "picture was taken, which direction is the camera facing in the room when the "
                "second picture is taken? Options: A: East, B: South, C: North, D: West"
            ),
        }
        compiled = MMSIProofCompiler().compile_record(record)
        self.assertEqual(compiled.frame, Frame.GEOGRAPHIC)
        self.assertEqual(compiled.graph.claims["option_c"].predicate, "north_of")

    def test_chair_curtain_compiles_orientation_as_unresolved(self) -> None:
        record = {
            "id": 178,
            "category": "Positional Relationship (Obj.–Obj.)",
            "question": (
                "When you are sitting on the green chair, where is the green curtain "
                "located in relation to you? Options: A: Front right, B: Directly to "
                "the left, C: Directly in front, D: Front left"
            ),
        }
        compiled = MMSIProofCompiler().compile_record(record)

        self.assertEqual(compiled.target, "green curtain")
        self.assertEqual(compiled.reference, "green chair")
        self.assertEqual(compiled.frame, Frame.OBJECT)
        self.assertEqual(len(compiled.hypotheses), 4)
        self.assertIn("observer_orientation", compiled.graph.claims)
        self.assertEqual(compiled.graph.claims["observer_orientation"].status, Status.UNKNOWN)
        self.assertTrue(all(
            "observer_orientation" in hypothesis.required_claim_ids
            for hypothesis in compiled.hypotheses
        ))

    def test_camera_object_uses_camera_frame(self) -> None:
        record = {
            "id": 704,
            "category": "Positional Relationship (Cam.–Obj.)",
            "question": (
                "When you take the photo in Image 2, where is the green chair from "
                "Image 1 located relative to you? Options: A: Directly in front, "
                "B: On the right, C: Directly behind, D: On the left"
            ),
        }
        compiled = MMSIProofCompiler().compile_record(record)

        self.assertEqual(compiled.frame, Frame.CAMERA)
        self.assertEqual(len(compiled.hypotheses), 4)
        self.assertFalse(compiled.warnings)

    def test_generic_camera_object_parse_is_marked_unverified(self) -> None:
        record = {
            "id": 1,
            "category": "Positional Relationship (Cam.–Obj.)",
            "question": (
                "When you took the second photo, where was the toilet in relation "
                "to you? Options: A: Left, B: Right, C: Front, D: Behind"
            ),
        }
        compiled = MMSIProofCompiler().compile_record(record)

        self.assertEqual(compiled.target, "toilet")
        self.assertEqual(compiled.reference, "observer_at_question_pose")
        self.assertTrue(any("semantic verification" in warning for warning in compiled.warnings))


if __name__ == "__main__":
    unittest.main()
