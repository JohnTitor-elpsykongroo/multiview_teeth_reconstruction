"""Known-camera geometry in mm and edge-origin pixels; never offset K."""
import numpy as np


def rigid(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError("expected finite rigid 4x4 transform, without scale/reflection")
    return t


def project(points, K, T_camera_from_world):
    t = rigid(T_camera_from_world)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx3")
    camera = points @ t[:3, :3].T + t[:3, 3]
    homogeneous = camera @ np.asarray(K).T
    uv = np.full((len(points), 2), np.nan)
    positive = camera[:, 2] > 0
    uv[positive] = homogeneous[positive, :2] / homogeneous[positive, 2:]
    return uv, camera[:, 2]


def ray(uv, K, T_camera_from_world):
    t = rigid(T_camera_from_world)
    direction = t[:3, :3].T @ np.linalg.solve(K, [*uv, 1.0])
    return -t[:3, :3].T @ t[:3, 3], direction / np.linalg.norm(direction)


def intersect_rays(origins, directions, max_condition=1e6):
    """Least-squares diagnostic, NOT anatomical correspondence or an arch pose."""
    o, d = np.asarray(origins, float), np.asarray(directions, float)
    if len(o) < 2:
        return {"status": "INSUFFICIENT_RAYS"}
    if o.shape != d.shape or o.shape[1:] != (3,) or not np.isfinite([o, d]).all():
        raise ValueError("rays must be finite Nx3")
    if np.any(np.linalg.norm(d, axis=1) == 0):
        raise ValueError("zero ray direction")
    d = d / np.linalg.norm(d, axis=1)[:, None]
    baseline = float(np.max(np.linalg.norm(o[:, None] - o[None], axis=-1)))
    p = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    a = p.sum(0)
    condition = float(np.linalg.cond(a))
    if baseline < 1e-6 or not np.isfinite(condition) or condition > max_condition:
        return {"status": "DEGENERATE_RAYS", "max_baseline_mm": baseline}
    x = np.linalg.solve(a, np.einsum("nij,nj->i", p, o))
    depths = np.einsum("ni,ni->n", x - o, d)
    residual = np.linalg.norm(np.einsum("nij,nj->ni", p, x - o), axis=1)
    return {"status": "CENTROID_HEURISTIC" if np.all(depths > 0) else "BEHIND_CAMERA",
            "point_world_mm": x.tolist(), "condition": condition,
            "rms_ray_distance_mm": float(np.sqrt(np.mean(residual ** 2))),
            "max_baseline_mm": baseline}


def estimate_rigid(source_mm, target_mm):
    """Kabsch foundation for independently supplied homologous landmarks.

    Not called automatically on mask centroids. No scale is estimated.
    """
    a, b = np.asarray(source_mm, float), np.asarray(target_mm, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3 or not np.isfinite([a, b]).all():
        raise ValueError("need >=3 paired finite 3D landmarks")
    ac, bc = a - a.mean(0), b - b.mean(0)
    if np.linalg.matrix_rank(ac) < 2 or np.linalg.matrix_rank(bc) < 2:
        raise ValueError("collinear landmarks do not constrain rigid pose")
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
    r = vt.T @ correction @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, b.mean(0) - r @ a.mean(0)
    return t, float(np.sqrt(np.mean(np.sum((a @ r.T + t[:3, 3] - b) ** 2, axis=1))))
