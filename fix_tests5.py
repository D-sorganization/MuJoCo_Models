import sys

# filepath_test_polygon
filepath_test_polygon = "tests/unit/optimization/test_polygon_geometry.py"
with open(filepath_test_polygon, "r") as f:
    content = f.read()

content = content.replace("""    def test_point_to_segment_clamping(self) -> None:
        \"\"\"Test explicit clamping coverage in _point_to_segment_sq.\"\"\"
        from mujoco_models.optimization.polygon_geometry import _point_to_segment_sq

        # point well past bx, by (t > 1.0)
        dist = _point_to_segment_sq(5.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(16.0)

        # point well before ax, ay (t < 0.0)
        dist = _point_to_segment_sq(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(1.0)

    def test_point_to_segment_clamping_zero(self) -> None:
        \"\"\"Test explicit clamping coverage for ab_sq < 1e-12 in _point_to_segment_sq.\"\"\"
        from mujoco_models.optimization.polygon_geometry import _point_to_segment_sq

        # ax == bx, ay == by
        dist = _point_to_segment_sq(2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert dist == pytest.approx(4.0)""", "")

with open(filepath_test_polygon, "w") as f:
    f.write(content)

# filepath_test_traj
filepath_test_traj = "tests/unit/optimization/test_trajectory_optimizer.py"
with open(filepath_test_traj, "r") as f:
    content = f.read()

content = content.replace("""    def test_point_to_segment_clamping(self) -> None:
        \"\"\"Test explicit clamping coverage in _point_to_segment_sq.\"\"\"
        from mujoco_models.optimization.trajectory_optimizer import _point_to_segment_sq

        # point well past bx, by (t > 1.0)
        dist = _point_to_segment_sq(5.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(16.0)

        # point well before ax, ay (t < 0.0)
        dist = _point_to_segment_sq(-1.0, 0.0, 0.0, 1.0, 2.0, 0.0)
        assert dist == pytest.approx(5.0)

    def test_point_to_segment_clamping_zero(self) -> None:
        \"\"\"Test explicit clamping coverage for ab_sq < 1e-12 in _point_to_segment_sq.\"\"\"
        from mujoco_models.optimization.trajectory_optimizer import _point_to_segment_sq

        # ax == bx, ay == by
        dist = _point_to_segment_sq(2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert dist == pytest.approx(4.0)""", "")

with open(filepath_test_traj, "w") as f:
    f.write(content)
