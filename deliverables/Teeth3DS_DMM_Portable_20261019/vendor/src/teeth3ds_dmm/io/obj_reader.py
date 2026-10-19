from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from teeth3ds_dmm.errors import Teeth3DSDMMError
from teeth3ds_dmm.reason_codes import ReasonCode


@dataclass(frozen=True)
class ObjMesh:
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    vertex_colors: NDArray[np.float64] | None
    colored_vertex_count: int
    has_vn_directives: bool

    @property
    def has_partial_vertex_colors(self) -> bool:
        return 0 < self.colored_vertex_count < len(self.vertices)


def read_obj(path: Path) -> ObjMesh:
    """Read positions, optional RGB, and triangle indices without reindexing vertices."""

    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float] | None] = []
    faces: list[tuple[int, int, int]] = []
    has_vn_directives = False

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.partition("#")[0].strip()
                if not line:
                    continue
                fields = line.split()
                directive = fields[0]
                if directive == "v":
                    if len(fields) not in {4, 7}:
                        _raise_obj_error(
                            path,
                            line_number,
                            "Vertex line must be 'v x y z' or 'v x y z r g b'",
                        )
                    try:
                        vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
                        colors.append(
                            (float(fields[4]), float(fields[5]), float(fields[6]))
                            if len(fields) == 7
                            else None
                        )
                    except ValueError as exc:
                        _raise_obj_error(path, line_number, "Vertex coordinates/colors must be numeric", exc)
                elif directive == "f":
                    if len(fields) != 4:
                        raise Teeth3DSDMMError(
                            ReasonCode.E_NON_TRIANGULAR_FACE,
                            "Only triangle faces are accepted by the v1.3 source contract",
                            context={"path": str(path), "line": line_number},
                        )
                    try:
                        faces.append(
                            tuple(_parse_vertex_index(token, len(vertices)) for token in fields[1:4])
                        )
                    except ValueError as exc:
                        _raise_obj_error(path, line_number, "Face vertex index is invalid", exc)
                elif directive == "vn":
                    # v1.3 does not use or require OBJ normals. Presence is only recorded.
                    has_vn_directives = True
    except UnicodeError as exc:
        raise Teeth3DSDMMError(
            ReasonCode.E_OBJ_PARSE_ERROR,
            "OBJ is not valid UTF-8 text",
            context={"path": str(path)},
        ) from exc
    except OSError as exc:
        raise Teeth3DSDMMError(
            ReasonCode.E_OBJ_PARSE_ERROR,
            "OBJ cannot be read",
            context={"path": str(path)},
        ) from exc

    try:
        vertex_array = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
        face_array = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    except (OverflowError, ValueError) as exc:
        raise Teeth3DSDMMError(
            ReasonCode.E_OBJ_PARSE_ERROR,
            "OBJ numeric values cannot be represented by the parser dtypes",
            context={"path": str(path)},
        ) from exc
    colored_vertex_count = sum(color is not None for color in colors)
    color_array: NDArray[np.float64] | None = None
    if colored_vertex_count:
        color_array = np.asarray(
            [color if color is not None else (np.nan, np.nan, np.nan) for color in colors],
            dtype=np.float64,
        ).reshape((-1, 3))
    return ObjMesh(
        vertices=vertex_array,
        faces=face_array,
        vertex_colors=color_array,
        colored_vertex_count=colored_vertex_count,
        has_vn_directives=has_vn_directives,
    )


def _parse_vertex_index(token: str, current_vertex_count: int) -> int:
    vertex_token = token.split("/", maxsplit=1)[0]
    raw_index = int(vertex_token)
    if raw_index == 0:
        raise ValueError("OBJ indices are one-based and cannot be zero")
    return raw_index - 1 if raw_index > 0 else current_vertex_count + raw_index


def _raise_obj_error(
    path: Path,
    line_number: int,
    message: str,
    cause: Exception | None = None,
) -> None:
    error = Teeth3DSDMMError(
        ReasonCode.E_OBJ_PARSE_ERROR,
        message,
        context={"path": str(path), "line": line_number},
    )
    if cause is None:
        raise error
    raise error from cause
