from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from teeth3ds_dmm.errors import Teeth3DSDMMError
from teeth3ds_dmm.reason_codes import ReasonCode
from teeth3ds_dmm.schemas.alignment import SimilarityTransform


RIGHT = frozenset(range(11, 18))
LEFT = frozenset(range(21, 28))
ANTERIOR = frozenset((11, 12, 21, 22))
POSTERIOR = frozenset((16, 17, 26, 27))


@dataclass(frozen=True)
class AlignmentCorrespondence:
    labels: tuple[int, ...]
    source: NDArray[np.float64]
    target: NDArray[np.float64]


def centroid_correspondence(
    source: dict[int, tuple[float, float, float]],
    target: dict[int, tuple[float, float, float]],
    *,
    min_common_teeth: int,
    require_bilateral: bool,
    require_anterior_posterior: bool,
) -> AlignmentCorrespondence:
    labels = tuple(sorted(set(source) & set(target)))
    valid = len(labels) >= min_common_teeth
    if require_bilateral:
        valid = valid and bool(set(labels) & RIGHT) and bool(set(labels) & LEFT)
    if require_anterior_posterior:
        valid = valid and bool(set(labels) & ANTERIOR) and bool(set(labels) & POSTERIOR)
    if not valid:
        raise Teeth3DSDMMError(
            ReasonCode.E_INSUFFICIENT_COMMON_TEETH,
            "Common centroid labels do not satisfy the frozen alignment coverage contract",
            context={"common_labels": list(labels), "min_common_teeth": min_common_teeth},
        )
    source_points = np.asarray([source[label] for label in labels], dtype=np.float64)
    target_points = np.asarray([target[label] for label in labels], dtype=np.float64)
    if not np.isfinite(source_points).all() or not np.isfinite(target_points).all():
        raise ValueError("Centroid correspondence contains non-finite coordinates")
    return AlignmentCorrespondence(labels, source_points, target_points)


def solve_similarity(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    determinant_tolerance: float = 1e-8,
    orthogonality_tolerance: float = 1e-8,
) -> SimilarityTransform:
    """Solve x' = s R x + t using the v1.3 SVD/reflection contract."""

    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("Similarity source/target must have identical shapes [N,3], N>=3")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Similarity input contains non-finite values")
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    centered_x = x - mu_x
    centered_y = y - mu_y
    denominator = float(np.sum(centered_x * centered_x))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Similarity source configuration is degenerate")
    u, singular_values, vt = np.linalg.svd(centered_x.T @ centered_y)
    v = vt.T
    raw_det = float(np.linalg.det(v @ u.T))
    correction = np.eye(3, dtype=np.float64)
    if raw_det < 0:
        correction[-1, -1] = -1.0
    rotation = v @ correction @ u.T
    scale = float(np.sum(singular_values * np.diag(correction)) / denominator)
    translation = mu_y - scale * (rotation @ mu_x)
    result = SimilarityTransform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        singular_values=singular_values,
        reflection_corrected=raw_det < 0,
    )
    if abs(result.det_rotation - 1.0) > determinant_tolerance:
        raise Teeth3DSDMMError(
            ReasonCode.E_ALIGNMENT_REFLECTION,
            "Similarity rotation determinant is not +1 after reflection correction",
            context={"det_rotation": result.det_rotation},
        )
    if result.orthogonality_error > orthogonality_tolerance:
        raise Teeth3DSDMMError(
            ReasonCode.E_ALIGNMENT_QC_FAILED,
            "Similarity rotation is not orthogonal within tolerance",
            context={"orthogonality_error": result.orthogonality_error},
        )
    return result


def align_centroid_maps(
    source: dict[int, tuple[float, float, float]],
    target: dict[int, tuple[float, float, float]],
    *,
    min_common_teeth: int,
    require_bilateral: bool = True,
    require_anterior_posterior: bool = True,
) -> tuple[SimilarityTransform, AlignmentCorrespondence, dict[int, tuple[float, float, float]]]:
    correspondence = centroid_correspondence(
        source,
        target,
        min_common_teeth=min_common_teeth,
        require_bilateral=require_bilateral,
        require_anterior_posterior=require_anterior_posterior,
    )
    transform = solve_similarity(correspondence.source, correspondence.target)
    aligned = transform.apply(
        np.asarray([source[label] for label in sorted(source)], dtype=np.float64)
    )
    aligned_map = {
        label: tuple(float(value) for value in point)
        for label, point in zip(sorted(source), aligned, strict=True)
    }
    return transform, correspondence, aligned_map
