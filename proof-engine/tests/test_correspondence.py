import unittest

import numpy as np

from fvpg import measure_patch_correspondence


class CorrespondenceMeasurementTest(unittest.TestCase):
    def test_identical_patch_descriptors_support_overlap(self) -> None:
        descriptors = np.eye(16, dtype=np.float64)
        result = measure_patch_correspondence(descriptors, descriptors, (4, 4))
        self.assertEqual(result.matches, 16)
        self.assertEqual(result.affine_inliers, 16)
        self.assertAlmostEqual(result.affine_inlier_ratio, 1.0)
        self.assertEqual(result.provisional_status, "supported")

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            measure_patch_correspondence(np.ones((4, 2)), np.ones((5, 2)), (2, 2))


if __name__ == "__main__":
    unittest.main()
