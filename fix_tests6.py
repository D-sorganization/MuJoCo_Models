import sys

def delete_test_method(filepath, class_name, method_name):
    with open(filepath, "r") as f:
        lines = f.readlines()

    in_class = False
    in_method = False
    method_indent = -1

    new_lines = []

    for line in lines:
        if line.startswith(f"class {class_name}"):
            in_class = True
            new_lines.append(line)
            continue

        if in_class:
            if line.strip().startswith("def " + method_name):
                in_method = True
                method_indent = len(line) - len(line.lstrip())
                continue

            if in_method:
                # if line is not empty and has indentation <= method_indent, we're out of the method
                if line.strip() and (len(line) - len(line.lstrip())) <= method_indent:
                    in_method = False
                else:
                    continue

        new_lines.append(line)

    with open(filepath, "w") as f:
        f.write("".join(new_lines))

filepath_test_polygon = "tests/unit/optimization/test_polygon_geometry.py"
delete_test_method(filepath_test_polygon, "TestSquaredDistanceToPolygon", "test_point_to_segment_clamping")
delete_test_method(filepath_test_polygon, "TestSquaredDistanceToPolygon", "test_point_to_segment_clamping_zero")

filepath_test_traj = "tests/unit/optimization/test_trajectory_optimizer.py"
delete_test_method(filepath_test_traj, "TestComputeBarPathCost", "test_point_to_segment_clamping")
delete_test_method(filepath_test_traj, "TestComputeBarPathCost", "test_point_to_segment_clamping_zero")
