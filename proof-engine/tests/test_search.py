import unittest

from fvpg import (
    ClaimTest,
    expected_information_gain,
    posterior_for_outcome,
    select_next_test,
)


class ActiveSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test = ClaimTest(
            "pose", frozenset({"A"}), frozenset({"B", "C", "D"}),
            true_positive_rate=0.9, false_positive_rate=0.05,
            unknown_probability=0.1, cost=2.0, reliability_group="pose-v1",
        )

    def test_support_increases_supported_option(self) -> None:
        posterior = posterior_for_outcome({x: 0.25 for x in "ABCD"}, self.test, "supported")
        self.assertGreater(posterior["A"], 0.8)

    def test_discriminative_test_has_information_gain(self) -> None:
        gain = expected_information_gain({x: 0.25 for x in "ABCD"}, self.test)
        self.assertGreater(gain, 0.4)

    def test_selector_respects_cost_and_correlation(self) -> None:
        expensive = ClaimTest(
            "expensive", frozenset({"A"}), frozenset({"B", "C", "D"}),
            0.9, 0.05, 0.1, 20.0, "other",
        )
        choice = select_next_test({"A": 0.6, "B": 0.2, "C": 0.1, "D": 0.1}, [expensive, self.test])
        self.assertEqual(choice.test_id, "pose")
        self.assertTrue(choice.targets_leader)
        blocked = select_next_test(
            {x: 0.25 for x in "ABCD"}, [self.test], used_reliability_groups={"pose-v1"}
        )
        self.assertIsNone(blocked)


if __name__ == "__main__":
    unittest.main()
