"""Analytical geometry checks for the nonfinite distance repair."""
import numpy as np
from surface_metric_utils import closest_distance_fallback, finite_nearest_triangle_distances

triangles = np.array([
    [[0,0,0],[1,0,0],[0,1,0]],
    [[0,0,0],[1,0,0],[0,1,0]],
    [[0,0,0],[1,0,0],[0,1,0]],
    [[0,0,0],[0,0,0],[1,0,0]],
    [[0,0,0],[0,0,0],[0,0,0]],
], dtype=float)
points = np.array([[.25,.25,1],[1,1,0],[-1,0,0],[.5,1,0],[0,0,2]], dtype=float)
expected = [1, np.sqrt(.5), 1, 1, 2]
np.testing.assert_allclose(closest_distance_fallback(triangles, points), expected, atol=1e-14, rtol=1e-14)
mesh = (triangles[3], np.array([[0,1,2],[0,0,0]]))
diagnostics = {}
actual = finite_nearest_triangle_distances(np.array([[.5,1,0],[2,0,0]],dtype=float), mesh, diagnostics)
np.testing.assert_allclose(actual, [1,1], atol=1e-14)
assert diagnostics['nonfinite_candidate_distances_repaired'] > 0
print({'status': 'ANALYTICAL_DISTANCE_CHECK_PASS', 'cases': 7, 'diagnostics': diagnostics})
