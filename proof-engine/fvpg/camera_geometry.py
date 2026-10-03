from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RelativeCameraMotion:
    yaw_degrees: float
    forward_vertical_component: float
    translation_in_first_camera: tuple[float, float, float]
    translation_norm: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.asarray(extrinsic, dtype=float)
    if matrix.shape != (3, 4):
        raise ValueError("extrinsic must have shape [3, 4]")
    result = np.eye(4)
    result[:3] = matrix
    return result


def relative_camera_motion(
    first_camera_from_world: np.ndarray,
    second_camera_from_world: np.ndarray,
) -> RelativeCameraMotion:
    """Interpret VGGT-style OpenCV camera-from-world extrinsics.

    Axes are x-right, y-down, z-forward. The returned translation is the second
    camera center expressed in the first camera frame. Positive yaw means the
    second camera looks to the first camera's right.
    """
    first = _homogeneous(first_camera_from_world)
    second = _homogeneous(second_camera_from_world)
    first_rotation = first[:3, :3]
    second_rotation = second[:3, :3]
    first_center_world = -first_rotation.T @ first[:3, 3]
    second_center_world = -second_rotation.T @ second[:3, 3]
    translation = first_rotation @ (second_center_world - first_center_world)

    second_forward_in_first = first_rotation @ second_rotation.T @ np.array([0.0, 0.0, 1.0])
    horizontal_norm = math.hypot(second_forward_in_first[0], second_forward_in_first[2])
    if horizontal_norm < 1e-8:
        raise ValueError("second camera forward axis has no horizontal component")
    yaw = math.degrees(math.atan2(second_forward_in_first[0], second_forward_in_first[2]))
    return RelativeCameraMotion(
        yaw_degrees=float(yaw),
        forward_vertical_component=float(second_forward_in_first[1]),
        translation_in_first_camera=tuple(float(value) for value in translation),
        translation_norm=float(np.linalg.norm(translation)),
    )


def cardinal_after_yaw(
    initial_heading: str,
    yaw_degrees: float,
    maximum_angular_error: float = 30.0,
) -> str | None:
    """Compose a known initial heading with a relative camera yaw."""
    headings = ("north", "east", "south", "west")
    try:
        initial_index = headings.index(initial_heading.lower())
    except ValueError as exception:
        raise ValueError(f"unsupported heading: {initial_heading}") from exception
    quarter_turns = round(yaw_degrees / 90.0)
    residual = abs(yaw_degrees - quarter_turns * 90.0)
    if residual > maximum_angular_error:
        return None
    # In OpenCV coordinates, positive yaw is a visual right turn. Compass
    # heading indices above advance clockwise.
    return headings[(initial_index + quarter_turns) % 4]


def inverse_yaw_consistent(
    forward_order_yaw: float,
    reverse_order_yaw: float,
    tolerance_degrees: float = 15.0,
) -> bool:
    """Check two separately inferred image orders without wraparound errors."""
    disagreement = (forward_order_yaw + reverse_order_yaw + 180.0) % 360.0 - 180.0
    return abs(disagreement) <= tolerance_degrees
