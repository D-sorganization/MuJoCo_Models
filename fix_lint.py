import sys

# filepath_polygon
filepath_polygon = "src/mujoco_models/optimization/polygon_geometry.py"
with open(filepath_polygon, "r") as f:
    content = f.read()

content = content.replace("    n = len(polygon)\n", "")

with open(filepath_polygon, "w") as f:
    f.write(content)

# filepath_traj
filepath_traj = "src/mujoco_models/optimization/trajectory_optimizer.py"
with open(filepath_traj, "r") as f:
    content = f.read()

content = content.replace("    n = len(poly_list)\n", "")

with open(filepath_traj, "w") as f:
    f.write(content)
