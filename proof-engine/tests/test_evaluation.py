import unittest

import numpy as np

from fvpg import (
    PairItem,
    binary_auc,
    classification_metrics,
    cohen_kappa,
    component_split,
    select_threshold,
)


class EvaluationTest(unittest.TestCase):
    def test_kappa_perfect_agreement(self) -> None:
        self.assertEqual(cohen_kappa(["a", "b", "a"], ["a", "b", "a"]), 1.0)

    def test_auc(self) -> None:
        self.assertEqual(binary_auc(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])), 1.0)

    def test_threshold_respects_false_positive_limit(self) -> None:
        labels = np.array([1, 1, 0, 0], dtype=bool)
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        threshold = select_threshold(labels, scores, maximum_false_positive_rate=0.0)
        self.assertEqual(threshold, 0.8)
        metrics = classification_metrics(labels, scores, threshold)
        self.assertEqual(metrics["true_positive_rate"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 0.0)

    def test_component_split_prevents_question_leakage(self) -> None:
        items = [
            PairItem("a", 1, 2),
            PairItem("b", 2, 3),
            PairItem("c", 4, 5),
        ]
        split = component_split(items)
        self.assertEqual(split["a"], split["b"])


if __name__ == "__main__":
    unittest.main()
