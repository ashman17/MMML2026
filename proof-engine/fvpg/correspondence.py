from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools

import numpy as np


@dataclass(frozen=True)
class CorrespondenceMeasurement:
    matches: int
    median_similarity: float
    affine_inliers: int
    affine_inlier_ratio: float
    median_affine_residual: float | None
    source_coverage: float
    target_coverage: float
    geometric_score: float
    provisional_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def measure_patch_correspondence(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    grid_shape: tuple[int, int],
    minimum_similarity: float = 0.55,
    residual_threshold: float = 0.10,
) -> CorrespondenceMeasurement:
    """Measure mutual patch matches and robust affine consistency.

    The provisional status is a coverage diagnostic, not a calibrated claim
    probability and not a substitute for LightGlue/geometric verification.
    """
    a = _normalize(descriptors_a)
    b = _normalize(descriptors_b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Descriptor arrays must be [patches, dimensions] with matching dimensions")
    expected = grid_shape[0] * grid_shape[1]
    if a.shape[0] != expected or b.shape[0] != expected:
        raise ValueError("Descriptor count does not match grid_shape")

    similarity = a @ b.T
    a_to_b = similarity.argmax(axis=1)
    b_to_a = similarity.argmax(axis=0)
    pairs = [
        (i, int(j), float(similarity[i, j]))
        for i, j in enumerate(a_to_b)
        if b_to_a[j] == i and similarity[i, j] >= minimum_similarity
    ]
    if not pairs:
        return CorrespondenceMeasurement(0, 0.0, 0, 0.0, None, 0.0, 0.0, 0.0, "unknown")

    coordinates = _grid_coordinates(grid_shape)
    source = np.asarray([coordinates[i] for i, _, _ in pairs])
    target = np.asarray([coordinates[j] for _, j, _ in pairs])
    similarities = np.asarray([score for _, _, score in pairs])
    inliers, residuals = _robust_affine(source, target, residual_threshold)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(pairs)
    source_coverage = len(set(i for i, _, _ in pairs)) / expected
    target_coverage = len(set(j for _, j, _ in pairs)) / expected
    median_residual = float(np.median(residuals[inliers])) if inlier_count else None
    geometric_score = inlier_count * inlier_ratio * float(np.median(similarities))

    supported = (
        len(pairs) >= 8
        and inlier_count >= 6
        and inlier_ratio >= 0.40
        and min(source_coverage, target_coverage) >= 0.03
    )
    return CorrespondenceMeasurement(
        len(pairs),
        float(np.median(similarities)),
        inlier_count,
        inlier_ratio,
        median_residual,
        source_coverage,
        target_coverage,
        geometric_score,
        "supported" if supported else "unknown",
    )


def _normalize(descriptors: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float64)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / np.maximum(norms, 1e-12)


def _grid_coordinates(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.meshgrid(
        (np.arange(height) + 0.5) / height,
        (np.arange(width) + 0.5) / width,
        indexing="ij",
    )
    return np.stack((x, y), axis=-1).reshape(-1, 2)


def _fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((source, np.ones(len(source))))
    transform, *_ = np.linalg.lstsq(design, target, rcond=None)
    return transform


def _robust_affine(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(source)
    if count < 3:
        return np.zeros(count, dtype=bool), np.full(count, np.inf)

    candidates = itertools.islice(itertools.combinations(range(count), 3), 512)
    best = np.zeros(count, dtype=bool)
    best_error = np.inf
    for indices in candidates:
        sample = source[list(indices)]
        design = np.column_stack((sample, np.ones(3)))
        if abs(np.linalg.det(design)) < 1e-6:
            continue
        transform = _fit_affine(sample, target[list(indices)])
        residuals = np.linalg.norm(
            np.column_stack((source, np.ones(count))) @ transform - target,
            axis=1,
        )
        inliers = residuals <= threshold
        error = float(residuals[inliers].sum()) if inliers.any() else np.inf
        if inliers.sum() > best.sum() or (inliers.sum() == best.sum() and error < best_error):
            best = inliers
            best_error = error

    if best.sum() >= 3:
        transform = _fit_affine(source[best], target[best])
        residuals = np.linalg.norm(
            np.column_stack((source, np.ones(count))) @ transform - target,
            axis=1,
        )
        best = residuals <= threshold
        return best, residuals
    return best, np.full(count, np.inf)
