from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teeth3ds_dmm.constants import TOOTH_LABELS


TRANSFORM_CONVENTION = "column_vector_xprime=sRx+t"


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    reflection_corrected: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Similarity scale must be finite and positive")
        if self.rotation.shape != (3, 3) or self.translation.shape != (3,):
            raise ValueError("Similarity rotation/translation must have shapes [3,3]/[3]")
        if self.singular_values.shape != (3,):
            raise ValueError("Similarity singular_values must have shape [3]")
        if not all(
            np.isfinite(value).all()
            for value in (self.rotation, self.translation, self.singular_values)
        ):
            raise ValueError("Similarity transform contains a non-finite value")

    @property
    def det_rotation(self) -> float:
        return float(np.linalg.det(self.rotation))

    @property
    def orthogonality_error(self) -> float:
        return float(
            np.max(np.abs(self.rotation.T @ self.rotation - np.eye(3, dtype=np.float64)))
        )

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim == 1:
            if values.shape != (3,):
                raise ValueError("A point must have shape [3]")
            return self.scale * (self.rotation @ values) + self.translation
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Points must have shape [N,3]")
        return self.scale * (values @ self.rotation.T) + self.translation

    def to_dict(self) -> dict[str, Any]:
        return {
            "convention": TRANSFORM_CONVENTION,
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "det_rotation": self.det_rotation,
            "orthogonality_error": self.orthogonality_error,
            "singular_values": self.singular_values.tolist(),
            "reflection_corrected": self.reflection_corrected,
        }


@dataclass(frozen=True)
class CanonicalFrame:
    schema_version: str
    reference_case_id: str
    basis_x: NDArray[np.float64]
    basis_y: NDArray[np.float64]
    basis_z: NDArray[np.float64]
    bbox_center_in_axes: NDArray[np.float64]
    bbox_half_extent: float
    reference_bbox_margin: float
    transform: SimilarityTransform
    reference_centroids: dict[int, tuple[float, float, float]]
    manual_orientation_confirmed: bool

    def __post_init__(self) -> None:
        if self.schema_version != "1.1" or not self.reference_case_id:
            raise ValueError("Canonical frame schema/reference is invalid")
        basis = np.column_stack((self.basis_x, self.basis_y, self.basis_z))
        if basis.shape != (3, 3) or not np.isfinite(basis).all():
            raise ValueError("Canonical basis is invalid")
        if np.max(np.abs(basis.T @ basis - np.eye(3))) > 1e-8:
            raise ValueError("Canonical basis must be orthonormal")
        if abs(float(np.linalg.det(basis)) - 1.0) > 1e-8:
            raise ValueError("Canonical basis must be right handed")
        if self.bbox_center_in_axes.shape != (3,):
            raise ValueError("bbox_center_in_axes must have shape [3]")
        if not np.isfinite(self.bbox_half_extent) or self.bbox_half_extent <= 0:
            raise ValueError("bbox_half_extent must be finite and positive")
        if not 0 < self.reference_bbox_margin <= 1:
            raise ValueError("reference_bbox_margin must be in (0,1]")
        if set(self.reference_centroids) - set(TOOTH_LABELS):
            raise ValueError("Canonical frame contains an unknown FDI label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_case_id": self.reference_case_id,
            "axis_definition": {
                "+x": "patient_right_to_patient_left",
                "+y": "posterior_to_anterior",
                "+z": "crown_occlusal_direction",
                "handedness": "right_handed",
            },
            "basis_x": self.basis_x.tolist(),
            "basis_y": self.basis_y.tolist(),
            "basis_z": self.basis_z.tolist(),
            "bbox_center_in_axes": self.bbox_center_in_axes.tolist(),
            "bbox_half_extent": self.bbox_half_extent,
            "reference_bbox_margin": self.reference_bbox_margin,
            "canonical_transform": self.transform.to_dict(),
            "reference_centroids": {
                str(label): list(value)
                for label, value in sorted(self.reference_centroids.items())
            },
            "manual_orientation_confirmed": self.manual_orientation_confirmed,
        }


@dataclass(frozen=True)
class TransformRecord:
    case_id: str
    centroid_method: str
    alignment_method: str
    transform: SimilarityTransform
    common_labels: tuple[int, ...]
    rmse_before: float
    rmse_after: float
    max_residual: float
    reference_case_id: str
    canonical_frame_hash: str
    status: str
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.1",
            "case_id": self.case_id,
            "centroid_method": self.centroid_method,
            "alignment_method": self.alignment_method,
            **self.transform.to_dict(),
            "common_labels": list(self.common_labels),
            "rmse_before": self.rmse_before,
            "rmse_after": self.rmse_after,
            "max_residual": self.max_residual,
            "reference_case_id": self.reference_case_id,
            "canonical_frame_hash": self.canonical_frame_hash,
            "canonical_frame_version": "1.1",
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }
