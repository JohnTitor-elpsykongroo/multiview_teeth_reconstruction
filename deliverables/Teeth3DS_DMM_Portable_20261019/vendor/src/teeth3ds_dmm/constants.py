from __future__ import annotations

PREPROCESSING_SPEC_VERSION = "1.3"
DEFAULT_GLOBAL_SEED = 20221121

GUM_LABEL = 0
TOOTH_LABELS = (11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27)
WISDOM_LABELS = (18, 28)
MODEL_LABELS = (GUM_LABEL, *TOOTH_LABELS)

CENTROID_METHODS = frozenset({"strict_surface_area_weighted", "vertex_mean"})
ALIGNMENT_METHODS = frozenset({"fixed_reference", "iterative_gpa"})
FREE_SPACE_METHODS = frozenset({"offsurface_compat", "temporary_watertight_proxy"})

