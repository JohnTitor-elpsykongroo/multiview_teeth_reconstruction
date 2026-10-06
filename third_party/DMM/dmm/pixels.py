"""Explicit image coordinates. Old manifests retain integer-center semantics."""
from dmm.validation import require

INTEGER = "integer_centers"
EDGE = "edge_origin_centers_at_half"


def pixel_offset(metadata):
    convention = metadata.get("pixel_convention", INTEGER)
    require(convention in (INTEGER, EDGE), "unsupported pixel_convention")
    return .5 if convention == EDGE else 0.


def image_center(width, height, metadata):
    offset = pixel_offset(metadata)
    return [(width - 1) / 2 + offset, (height - 1) / 2 + offset]
