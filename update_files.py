import sys

# filepath_polygon
filepath_polygon = "src/mujoco_models/optimization/polygon_geometry.py"
with open(filepath_polygon, "r") as f:
    content = f.read()

content = content.replace("""    xj, yj = poly_list[-1][0], poly_list[-1][1]

    for i in range(n):
        xi, yi = poly_list[i][0], poly_list[i][1]

        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside

        xj, yj = xi, yi""", """    xj, yj = poly_list[-1]

    for xi, yi in poly_list:
        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        xj, yj = xi, yi""")

content = content.replace("""    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        dist_sq = _point_to_segment_sq(
            px,
            py,
            poly_list[i][0],
            poly_list[i][1],
            poly_list[j][0],
            poly_list[j][1],
        )
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq""", """    xj, yj = poly_list[-1]

    for xi, yi in poly_list:
        dist_sq = _point_to_segment_sq(px, py, xj, yj, xi, yi)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
        xj, yj = xi, yi""")

with open(filepath_polygon, "w") as f:
    f.write(content)

# filepath_traj
filepath_traj = "src/mujoco_models/optimization/trajectory_optimizer.py"
with open(filepath_traj, "r") as f:
    content = f.read()

content = content.replace("""    xj, yj = poly_list[-1][0], poly_list[-1][1]

    for i in range(n):
        xi, yi = poly_list[i][0], poly_list[i][1]

        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside

        xj, yj = xi, yi""", """    xj, yj = poly_list[-1]

    for xi, yi in poly_list:
        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        xj, yj = xi, yi""")

content = content.replace("""    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        dist_sq = _point_to_segment_sq(
            px,
            py,
            poly_list[i][0],
            poly_list[i][1],
            poly_list[j][0],
            poly_list[j][1],
        )
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq""", """    xj, yj = poly_list[-1]

    for xi, yi in poly_list:
        dist_sq = _point_to_segment_sq(px, py, xj, yj, xi, yi)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
        xj, yj = xi, yi""")

with open(filepath_traj, "w") as f:
    f.write(content)
