import sys

filepath = "SPEC.md"
with open(filepath, "r") as f:
    content = f.read()

if "## Recent Updates" not in content:
    content = content.replace("## Future Optimizations", "## Recent Updates\n\n- Inlined `_point_to_segment_sq` into `squared_distance_to_polygon` and `_squared_distance_to_polygon` to avoid function call overhead during iterative geometry calculations.\n\n## Future Optimizations")

with open(filepath, "w") as f:
    f.write(content)
