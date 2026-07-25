import sys

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
            min_dist_sq = dist_sq
    return min_dist_sq


# OPTIMIZATION: Replaced numpy array operations with scalar arithmetic
# to avoid array creation overhead for 2D vectors.
def _point_to_segment_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    \"\"\"Squared distance from a point to a line segment.\"\"\"
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

    return dx * dx + dy * dy""", """    xj, yj = poly_list[-1]

    for xi, yi in poly_list:
        abx = xi - xj
        aby = yi - yj

        ab_sq = abx * abx + aby * aby
        if ab_sq < 1e-12:
            apx = px - xj
            apy = py - yj
            dist_sq = apx * apx + apy * apy
        else:
            t = ((px - xj) * abx + (py - yj) * aby) / ab_sq
            if t < 0.0:
                dx = px - xj
                dy = py - yj
            elif t > 1.0:
                dx = px - xi
                dy = py - yi
            else:
                dx = px - (xj + t * abx)
                dy = py - (yj + t * aby)
            dist_sq = dx * dx + dy * dy

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

        xj, yj = xi, yi
    return min_dist_sq""")

with open(filepath_polygon, "w") as f:
    f.write(content)

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
            min_dist_sq = dist_sq
    return min_dist_sq


# OPTIMIZATION: Replaced numpy array operations with scalar arithmetic
# to avoid array creation overhead for 2D vectors.
def _point_to_segment_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    \"\"\"Squared distance from a point to a line segment.\"\"\"
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

    return dx * dx + dy * dy""", """    xj, yj = poly_list[-1]

    for xi, yi in poly_list:
        abx = xi - xj
        aby = yi - yj

        ab_sq = abx * abx + aby * aby
        if ab_sq < 1e-12:
            apx = px - xj
            apy = py - yj
            dist_sq = apx * apx + apy * apy
        else:
            t = ((px - xj) * abx + (py - yj) * aby) / ab_sq
            if t < 0.0:
                dx = px - xj
                dy = py - yj
            elif t > 1.0:
                dx = px - xi
                dy = py - yi
            else:
                dx = px - (xj + t * abx)
                dy = py - (yj + t * aby)
            dist_sq = dx * dx + dy * dy

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

        xj, yj = xi, yi
    return min_dist_sq""")

with open(filepath_traj, "w") as f:
    f.write(content)
