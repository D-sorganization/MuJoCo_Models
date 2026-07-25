import sys

# filepath_test_polygon
filepath_test_polygon = "tests/unit/optimization/test_polygon_geometry.py"
with open(filepath_test_polygon, "r") as f:
    content = f.read()

content += """
def _point_to_segment_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    abx = bx - ax
    aby = by - ay

    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-12:
        apx = px - ax
        apy = py - ay
        return apx * apx + apy * apy

    t = ((px - ax) * abx + (py - ay) * aby) / ab_sq
    if t < 0.0:
        dx = px - ax
        dy = py - ay
    elif t > 1.0:
        dx = px - bx
        dy = py - by
    else:
        dx = px - (ax + t * abx)
        dy = py - (ay + t * aby)

    return dx * dx + dy * dy

class TestSquaredDistanceToPolygonFallback:
    def test_point_to_segment_clamping(self) -> None:
        dist = _point_to_segment_sq(5.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == 16.0

        dist = _point_to_segment_sq(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == 1.0

    def test_point_to_segment_clamping_zero(self) -> None:
        dist = _point_to_segment_sq(2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert dist == 4.0
"""
with open(filepath_test_polygon, "w") as f:
    f.write(content)


# filepath_test_traj
filepath_test_traj = "tests/unit/optimization/test_trajectory_optimizer.py"
with open(filepath_test_traj, "r") as f:
    content = f.read()

content += """
class TestComputeBarPathCostFallback:
    def test_point_to_segment_clamping(self) -> None:
        from tests.unit.optimization.test_polygon_geometry import _point_to_segment_sq
        dist = _point_to_segment_sq(5.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == 16.0

        dist = _point_to_segment_sq(-1.0, 0.0, 0.0, 1.0, 2.0, 0.0)
        assert dist == 5.0

    def test_point_to_segment_clamping_zero(self) -> None:
        from tests.unit.optimization.test_polygon_geometry import _point_to_segment_sq
        dist = _point_to_segment_sq(2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert dist == 4.0
"""
with open(filepath_test_traj, "w") as f:
    f.write(content)
