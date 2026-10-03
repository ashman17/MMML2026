import math
import unittest

import numpy as np

from fvpg import cardinal_after_yaw, inverse_yaw_consistent, relative_camera_motion


def camera_from_world(center, yaw_degrees):
    angle = math.radians(yaw_degrees)
    # Camera-to-world rotation for a positive visual right turn.
    camera_to_world = np.array([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ])
    world_to_camera = camera_to_world.T
    translation = -world_to_camera @ np.asarray(center, dtype=float)
    return np.column_stack((world_to_camera, translation))


class CameraGeometryTest(unittest.TestCase):
    def test_right_turn_and_translation(self) -> None:
        first = camera_from_world([0, 0, 0], 0)
        second = camera_from_world([1, 0, 2], 90)
        motion = relative_camera_motion(first, second)
        self.assertAlmostEqual(motion.yaw_degrees, 90.0)
        np.testing.assert_allclose(motion.translation_in_first_camera, [1, 0, 2])
        self.assertEqual(cardinal_after_yaw("west", motion.yaw_degrees), "north")

    def test_non_cardinal_yaw_is_unknown(self) -> None:
        self.assertIsNone(cardinal_after_yaw("north", 44.0, maximum_angular_error=20.0))

    def test_reverse_order_consistency(self) -> None:
        self.assertTrue(inverse_yaw_consistent(91, -88, tolerance_degrees=5))
        self.assertFalse(inverse_yaw_consistent(90, -40, tolerance_degrees=5))


if __name__ == "__main__":
    unittest.main()
