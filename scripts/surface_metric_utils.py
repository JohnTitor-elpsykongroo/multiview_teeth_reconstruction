"""Finite point-to-triangle distances, including collapsed triangle edges."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def closest_distance_fallback(triangles, points):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge_distances = []
    for first, second in [(a, b), (b, c), (c, a)]:
        edge = second-first
        squared = np.einsum('ij,ij->i', edge, edge)
        numerator = np.einsum('ij,ij->i', points-first, edge)
        parameter = np.clip(np.divide(numerator, squared, out=np.zeros_like(numerator), where=squared>0), 0, 1)
        closest = first+parameter[:, None]*edge
        edge_distances.append(np.linalg.norm(points-closest, axis=1))
    distance = np.min(edge_distances, axis=0)
    normal = np.cross(b-a, c-a)
    squared_normal = np.einsum('ij,ij->i', normal, normal)
    regular = squared_normal > 0
    if regular.any():
        n, aa, bb, cc, pp = normal[regular], a[regular], b[regular], c[regular], points[regular]
        factor = np.einsum('ij,ij->i', pp-aa, n)/squared_normal[regular]
        projection = pp-factor[:, None]*n
        v = np.einsum('ij,ij->i', np.cross(projection-aa, cc-aa), n)/squared_normal[regular]
        w = np.einsum('ij,ij->i', np.cross(bb-aa, projection-aa), n)/squared_normal[regular]
        inside = (v>=0) & (w>=0) & (v+w<=1)
        index = np.flatnonzero(regular)[inside]
        distance[index] = np.minimum(distance[index], np.abs(factor[inside])*np.sqrt(squared_normal[index]))
    if not np.isfinite(distance).all():
        raise ValueError('nonfinite fallback distance')
    return distance


def finite_nearest_triangle_distances(points, mesh, diagnostics=None):
    triangles = mesh[0][mesh[1]].astype(float)
    if not np.isfinite(triangles).all() or not np.isfinite(points).all() or not len(triangles):
        raise ValueError('invalid mesh or sampled points')
    tree = cKDTree(triangles.mean(axis=1))
    result, repairs = [], 0
    k = min(32, len(triangles))
    for start in range(0, len(points), 512):
        part = points[start:start+512]
        _, indices = tree.query(part, k=k)
        candidates = triangles[indices].reshape(-1, 3, 3)
        expanded = np.repeat(part, k, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            closest = trimesh.triangles.closest_point(candidates, expanded)
        distances = np.linalg.norm(closest-expanded, axis=1)
        bad = ~np.isfinite(distances)
        if bad.any():
            distances[bad] = closest_distance_fallback(candidates[bad], expanded[bad])
            repairs += int(bad.sum())
        result.append(distances.reshape(-1, k).min(axis=1))
    answer = np.concatenate(result)
    if not np.isfinite(answer).all():
        raise ValueError('nonfinite nearest-triangle distance')
    if diagnostics is not None:
        diagnostics['nonfinite_candidate_distances_repaired'] = repairs
        diagnostics['zero_area_target_triangles'] = int((np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1)==0).sum())
    return answer
