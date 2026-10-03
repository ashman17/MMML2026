import unittest

from fvpg import Status, one_sided_support_evidence


class EvidenceAdapterTest(unittest.TestCase):
    def test_failed_match_is_unknown_not_contradicted(self) -> None:
        evidence = one_sided_support_evidence(
            evidence_id="e", verifier="lightglue", score=1.0,
            support_threshold=2.0, calibrated_precision=0.95,
            provenance={"images": ["a", "b"]}, reliability_group="correspondence-v1",
        )
        self.assertEqual(evidence.status, Status.UNKNOWN)
        self.assertEqual(evidence.contradiction_probability, 0.0)

    def test_pass_carries_calibration_and_provenance(self) -> None:
        evidence = one_sided_support_evidence(
            evidence_id="e", verifier="lightglue", score=3.0,
            support_threshold=2.0, calibrated_precision=0.95,
            provenance={"images": ["a", "b"]}, reliability_group="correspondence-v1",
        )
        self.assertEqual(evidence.status, Status.SUPPORTED)
        self.assertEqual(evidence.support_probability, 0.95)
        self.assertEqual(evidence.provenance["raw_score"], 3.0)


if __name__ == "__main__":
    unittest.main()
