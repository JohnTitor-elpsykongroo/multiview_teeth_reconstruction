from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teeth3ds_dmm.schemas.alignment import CanonicalFrame, SimilarityTransform


def _group_mean(
    centroids: dict[int, tuple[float, float, float]], labels: tuple[int, ...], name: str
) -> NDArray[np.float64]:
    present = [centroids[label] for label in labels if label in centroids]
    if not present:
        raise ValueError(f"Reference has no centroid in canonical {name} group")
    return np.asarray(present, dtype=np.float64).mean(axis=0)


def _normalize(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError(f"Canonical {name} axis is degenerate")
    return value / norm


def build_canonical_frame(
    *,
    reference_case_id: str,
    centroids: dict[int, tuple[float, float, float]],
    mesh_vertices: NDArray[np.float64],
    right_group: tuple[int, ...],
    left_group: tuple[int, ...],
    anterior_group: tuple[int, ...],
    posterior_group: tuple[int, ...],
    reference_bbox_margin: float,
    manual_orientation_confirmed: bool,
) -> CanonicalFrame:
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Reference mesh vertices must have shape [N,3]")
    if not np.isfinite(vertices).all():
        raise ValueError("Reference mesh vertices contain non-finite values")
    x_axis = _normalize(
        _group_mean(centroids, left_group, "left")
        - _group_mean(centroids, right_group, "right"),
        "X",
    )
    y_initial = (
        _group_mean(centroids, anterior_group, "anterior")
        - _group_mean(centroids, posterior_group, "posterior")
    )
    y_axis = _normalize(y_initial - np.dot(y_initial, x_axis) * x_axis, "Y")
    z_axis = _normalize(np.cross(x_axis, y_axis), "Z")
    basis = np.column_stack((x_axis, y_axis, z_axis))
    rotation = basis.T
    vertices_in_axes = vertices @ basis
    bbox_min = vertices_in_axes.min(axis=0)
    bbox_max = vertices_in_axes.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    half_extent = 0.5 * float(np.max(bbox_max - bbox_min))
    if not np.isfinite(half_extent) or half_extent <= np.finfo(np.float64).eps:
        raise ValueError("Reference mesh bounding box is degenerate")
    scale = float(reference_bbox_margin / half_extent)
    translation = -scale * bbox_center
    transform = SimilarityTransform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        singular_values=np.zeros(3, dtype=np.float64),
        reflection_corrected=False,
    )
    labels = sorted(centroids)
    points = transform.apply(np.asarray([centroids[label] for label in labels], dtype=np.float64))
    canonical_centroids = {
        label: tuple(float(value) for value in point)
        for label, point in zip(labels, points, strict=True)
    }
    return CanonicalFrame(
        schema_version="1.1",
        reference_case_id=reference_case_id,
        basis_x=x_axis,
        basis_y=y_axis,
        basis_z=z_axis,
        bbox_center_in_axes=bbox_center,
        bbox_half_extent=half_extent,
        reference_bbox_margin=reference_bbox_margin,
        transform=transform,
        reference_centroids=canonical_centroids,
        manual_orientation_confirmed=manual_orientation_confirmed,
    )
