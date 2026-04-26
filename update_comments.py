import re

with open("src/mujoco_models/optimization/polygon_geometry.py", "r") as f:
    content = f.read()

content = content.replace("def _point_to_segment_sq(", "# OPTIMIZATION: Replaced numpy array operations with scalar arithmetic to avoid array creation overhead for 2D vectors.\ndef _point_to_segment_sq(")

with open("src/mujoco_models/optimization/polygon_geometry.py", "w") as f:
    f.write(content)

with open("src/mujoco_models/optimization/trajectory_optimizer.py", "r") as f:
    content = f.read()

content = content.replace("def _point_to_segment_sq(", "# OPTIMIZATION: Replaced numpy array operations with scalar arithmetic to avoid array creation overhead for 2D vectors.\ndef _point_to_segment_sq(")

with open("src/mujoco_models/optimization/trajectory_optimizer.py", "w") as f:
    f.write(content)
