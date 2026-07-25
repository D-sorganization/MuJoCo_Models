import sys

filepath = "src/mujoco_models/optimization/polygon_geometry.py"
with open(filepath, "r") as f:
    content = f.read()

content = content.replace("""    for xi, yi in poly_list:
        dist_sq = _point_to_segment_sq(px, py, xj, yj, xi, yi)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
        xj, yj = xi, yi
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

    return dx * dx + dy * dy
""", """    for xi, yi in poly_list:
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
    return min_dist_sq
""")

with open(filepath, "w") as f:
    f.write(content)
